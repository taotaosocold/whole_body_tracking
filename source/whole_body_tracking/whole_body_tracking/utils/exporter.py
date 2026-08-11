# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os

import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


def export_terrain_policy_as_onnx(
    actor: object,
    path: str,
    filename="policy.onnx",
    verbose=False,
):
    """Export a terrain-locomotion actor including its CNN and attention encoder."""
    required_attributes = ("obs_normalizer", "cnns", "proprio_embedding", "map_scan_size")
    missing_attributes = [name for name in required_attributes if not hasattr(actor, name)]
    if missing_attributes:
        raise TypeError(
            "Terrain policy exporter received an incompatible actor; "
            f"missing attributes: {missing_attributes}"
        )
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxTerrainPolicyExporter(actor, verbose)
    policy_exporter.export(path, filename)


def export_parkour_policy_as_onnx(
    actor: object,
    path: str,
    filename="policy.onnx",
    verbose=False,
):
    """Export a parkour actor with normalization, terrain CNN and cross-attention."""
    required_attributes = (
        "obs_normalizer",
        "cnns",
        "proprio_embedding",
        "map_scan_size",
        "mlp",
    )
    missing_attributes = [name for name in required_attributes if not hasattr(actor, name)]
    if missing_attributes:
        raise TypeError(
            "Parkour policy exporter received an incompatible actor; "
            f"missing attributes: {missing_attributes}"
        )
    os.makedirs(path, exist_ok=True)
    _OnnxParkourPolicyExporter(actor, verbose).export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


class _OnnxTerrainPolicyExporter(torch.nn.Module):
    """ONNX wrapper for the terrain-locomotion actor."""

    def __init__(self, actor, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.normalizer = copy.deepcopy(actor.obs_normalizer)
        self.map_cnn = copy.deepcopy(actor.cnns["map_cnn"])
        self.mha = copy.deepcopy(actor.cnns["mha"])
        self.proprio_embedding = copy.deepcopy(actor.proprio_embedding)
        self.actor = copy.deepcopy(actor.mlp)
        self.deterministic_output = (
            actor.distribution.as_deterministic_output_module()
            if actor.distribution is not None
            else torch.nn.Identity()
        )
        self.input_size = actor.obs_dim
        self.map_scan_size = actor.map_scan_size
        self.map_length = actor.L
        self.map_width = actor.W
        self.coord_dim = actor.coord_dim

    def forward(self, x):
        x = self.normalizer(x)
        proprio = x[:, :-self.map_scan_size]
        terrain = x[:, -self.map_scan_size :].reshape(
            -1, self.map_width, self.map_length, self.coord_dim
        )
        terrain = terrain.permute(0, 3, 1, 2)
        terrain_tokens = self.map_cnn(terrain)
        terrain_tokens = terrain_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        query = self.proprio_embedding(proprio).unsqueeze(1)
        terrain_feature, _ = self.mha(
            query=query,
            key=terrain_tokens,
            value=terrain_tokens,
            need_weights=True,
        )
        latent = torch.cat((proprio, terrain_feature.squeeze(1)), dim=-1)
        return self.deterministic_output(self.actor(latent))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        obs = torch.zeros(1, self.input_size)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        )


class _OnnxParkourPolicyExporter(_OnnxTerrainPolicyExporter):
    """ONNX wrapper for parkour policies using the terrain encoder architecture."""

    pass


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)
    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": getattr(env.scene["robot"].data, "default_joint_pos_nominal", env.scene["robot"].data.default_joint_pos[0]).cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
