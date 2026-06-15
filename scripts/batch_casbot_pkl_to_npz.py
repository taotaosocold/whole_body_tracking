"""
    批量将 pkl 文件夹转为 npz 文件，使用 25dof 单腰 CASBOT 机器人配置。
    python scripts/batch_casbot_pkl_to_npz.py --input_folder /path/to/pkls --outpt_folder /path/to/npzs --input_fps 30 --headless
"""

import argparse
import numpy as np
import os as _os
import sys
from isaaclab.app import AppLauncher
import pickle

parser = argparse.ArgumentParser(description="Batch convert pkl files to npz files.")
parser.add_argument("--input_folder", type=str, required=True, help="The path to the folder containing pkl files.")
parser.add_argument("--input_fps", type=int, default=60, help="The fps of the input motion.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--output_folder", type=str, default=None, help="The folder to save output npz files (default: same as input_folder).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

input_folder = _os.path.abspath(args_cli.input_folder)
output_folder = _os.path.abspath(args_cli.output_folder) if args_cli.output_folder else input_folder
_os.makedirs(output_folder, exist_ok=True)

pkl_files = sorted(f for f in _os.listdir(input_folder) if f.endswith(".pkl"))
if not pkl_files:
    print(f"[ERROR] No .pkl files found in {input_folder}")
    sys.exit(1)
print(f"[INFO] Found {len(pkl_files)} .pkl files in {input_folder}")
print(f"[INFO] Output directory: {output_folder}")

# launch omniverse app
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
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp, quat_apply, euler_xyz_from_quat, quat_from_euler_xyz

from whole_body_tracking.robots.casbot_02 import CASBOT_02_25DOF_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
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
    def __init__(self, motion_file, input_fps, output_fps, device):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if np.__version__ < '2.0.0':
            sys.modules['numpy._core'] = np.core
            from numpy.core import multiarray
            sys.modules['numpy._core.multiarray'] = multiarray
        with open(self.motion_file, 'rb') as f:
            motion = pickle.load(f)
            motion['root_pos'] = torch.from_numpy(motion['root_pos'])
            motion['root_rot'] = torch.from_numpy(motion['root_rot'])
            motion['dof_pos'] = torch.from_numpy(motion['dof_pos'])
        self.motion_base_poss_input = motion['root_pos'].float().to(self.device)
        self.motion_base_rots_input = motion['root_rot'].float().to(self.device)
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]].float().to(self.device)
        self.motion_dof_poss_input = motion['dof_pos'].float().to(self.device)
        self.input_frames = self.motion_base_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"  loaded {_os.path.basename(self.motion_file)} — {self.duration:.1f}s, {self.input_frames} frames")

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(self.motion_base_poss_input[index_0], self.motion_base_poss_input[index_1], blend.unsqueeze(1))
        self.motion_base_rots = self._slerp(self.motion_base_rots_input[index_0], self.motion_base_rots_input[index_1], blend)
        self.motion_dof_poss = self._lerp(self.motion_dof_poss_input[index_0], self.motion_dof_poss_input[index_1], blend.unsqueeze(1))

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        slerped = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        done = self.current_idx >= self.output_frames
        if done:
            self.current_idx = 0
        return state, done


JOINT_NAMES = [
    "left_leg_pelvic_pitch_joint", "left_leg_pelvic_roll_joint", "left_leg_pelvic_yaw_joint",
    "left_leg_knee_pitch_joint", "left_leg_ankle_pitch_joint", "left_leg_ankle_roll_joint",
    "right_leg_pelvic_pitch_joint", "right_leg_pelvic_roll_joint", "right_leg_pelvic_yaw_joint",
    "right_leg_knee_pitch_joint", "right_leg_ankle_pitch_joint", "right_leg_ankle_roll_joint",
    "waist_yaw_joint", "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_wrist_yaw_joint",
]


def process_one_file(sim, scene, robot_joint_indexes, pkl_path, output_path):
    """Load a pkl, step through all frames, save npz."""
    robot = scene["robot"]

    motion = MotionLoader(pkl_path, args_cli.input_fps, args_cli.output_fps, sim.device)

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [], "joint_vel": [],
        "body_pos_w": [], "body_quat_w": [],
        "body_lin_vel_w": [], "body_ang_vel_w": [],
    }

    sim_dt = sim.get_physics_dt()

    for _ in range(motion.output_frames):
        state, done = motion.get_next_state()
        (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel) = state

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = base_rot
        root_states[:, 7:10] = base_lin_vel
        root_states[:, 10:] = base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = dof_pos
        joint_vel[:, robot_joint_indexes] = dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim_dt)

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)

    np.savez(output_path, **log)
    print(f"  saved → {output_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]

    print(f"[INFO] Processing {len(pkl_files)} files...")
    for i, fname in enumerate(pkl_files):
        pkl_path = _os.path.join(input_folder, fname)
        output_path = _os.path.join(output_folder, fname.replace(".pkl", ".npz"))
        print(f"[{i+1}/{len(pkl_files)}] {fname}")
        process_one_file(sim, scene, robot_joint_indexes, pkl_path, output_path)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
