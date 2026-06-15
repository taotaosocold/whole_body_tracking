# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
    Convert SOMA-retargeter CSV motion to NPZ format for CASBOT 25-DOF.

    Usage:
        python scripts/casbot_soma_to_npz.py -f path_to_input.csv --input_fps 120 --headless
"""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert SOMA CSV motion to NPZ format.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="The path to the input motion CSV file.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="frame range: START END (both inclusive, frame index starts from 1).",
)
parser.add_argument("--output_name", type=str, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.output_name:
    args_cli.output_name = (
        "/".join(args_cli.input_file.split("/")[:-1])
        + "/"
        + args_cli.input_file.split("/")[-1].replace(".csv", ".npz")
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_slerp,
    euler_xyz_from_quat,
)

from whole_body_tracking.robots.casbot_02 import CASBOT_02_25DOF_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = CASBOT_02_25DOF_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the SOMA CSV file."""
        motion_csv = np.loadtxt(self.motion_file, delimiter=",", skiprows=1)

        if self.frame_range is not None:
            start_idx = self.frame_range[0] - 1
            end_idx = self.frame_range[1]
            motion_csv = motion_csv[start_idx:end_idx]

        # columns 1-3: root translation (cm → m)
        root_pos = torch.from_numpy(motion_csv[:, 1:4] / 100.0).float()
        # columns 4-6: root rotation (Euler XYZ degrees → quaternion wxyz)
        euler_rad = torch.from_numpy(np.radians(motion_csv[:, 4:7])).float()
        root_rot = quat_from_euler_xyz(euler_rad[:, 0], euler_rad[:, 1], euler_rad[:, 2])
        # columns 7-31: joint angles (degrees → radians, first 25 DOFs)
        dof_pos = torch.from_numpy(np.radians(motion_csv[:, 7:32])).float()

        self.motion_base_poss_input = root_pos.to(self.device)
        self.motion_base_rots_input = root_rot.to(self.device)
        self.motion_dof_poss_input = dof_pos.to(self.device)

        self.input_frames = self.motion_base_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration:.2f} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps},"
            f" output frames: {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            body_pos_w = robot.data.body_pos_w[0]
            body_quat_w = robot.data.body_quat_w[0]
            body_lin_vel_w = robot.data.body_lin_vel_w[0]
            body_ang_vel_w = robot.data.body_ang_vel_w[0]

            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(body_pos_w.cpu().numpy().copy())
            log["body_quat_w"].append(body_quat_w.cpu().numpy().copy())
            log["body_lin_vel_w"].append(body_lin_vel_w.cpu().numpy().copy())
            log["body_ang_vel_w"].append(body_ang_vel_w.cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)
            np.savez(args_cli.output_name, **log)
            print("[INFO]: Motion npz file saved to", args_cli.output_name)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(
        sim,
        scene,
        joint_names=[
            "left_leg_pelvic_pitch_joint",
            "left_leg_pelvic_roll_joint",
            "left_leg_pelvic_yaw_joint",
            "left_leg_knee_pitch_joint",
            "left_leg_ankle_pitch_joint",
            "left_leg_ankle_roll_joint",
            "right_leg_pelvic_pitch_joint",
            "right_leg_pelvic_roll_joint",
            "right_leg_pelvic_yaw_joint",
            "right_leg_knee_pitch_joint",
            "right_leg_ankle_pitch_joint",
            "right_leg_ankle_roll_joint",
            "waist_yaw_joint",
            "head_yaw_joint",
            "head_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
