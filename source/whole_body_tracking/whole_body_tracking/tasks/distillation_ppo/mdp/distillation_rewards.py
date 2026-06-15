"""ONNX teacher policy + distillation reward for PPO-based BC."""

import os

import torch

_teacher_cache: dict[str, "_TeacherPolicy"] = {}


class _TeacherPolicy:
    """Cached ONNX teacher model loaded via onnxruntime."""

    def __init__(self, onnx_path: str):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Teacher ONNX not found: {onnx_path}")
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def get_action(self, obs: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        """Run teacher inference, return actions tensor on the same device as obs."""
        device = obs.device
        obs_np = obs.float().cpu().numpy()
        ts_np = time_step.float().cpu().numpy()
        outputs = self.session.run(None, {"obs": obs_np, "time_step": ts_np})
        return torch.tensor(outputs[0], device=device)


def _get_teacher(onnx_path: str) -> _TeacherPolicy:
    if onnx_path not in _teacher_cache:
        _teacher_cache[onnx_path] = _TeacherPolicy(onnx_path)
    return _teacher_cache[onnx_path]


def distillation_action_l2(
    env,
    sigma: float = 0.1,
    teacher_onnx_path: str = "",
    teacher_mode: str = "abs",
):
    """Gaussian kernel reward: exp(-||student_action - teacher_action||^2 / (2*sigma^2)).

    Args:
        sigma: Gaussian kernel width.
        teacher_onnx_path: Path to teacher ONNX model.
        teacher_mode: "abs" (teacher outputs absolute actions) or
                      "residual_to_abs" (teacher outputs residuals; convert to
                      absolute equivalent before comparison).
    """
    if not teacher_onnx_path:
        return torch.zeros(env.num_envs, device=env.device)

    teacher = _get_teacher(teacher_onnx_path)

    # student actions (processed = ready-to-use target)
    student_actions = env.action_manager.get_term("joint_pos").processed_actions

    # policy observation + motion time step for teacher
    obs = env.observation_manager.get_obs("policy")
    time_step = env.command_manager.get_term("motion").time_steps.unsqueeze(-1)

    with torch.no_grad():
        teacher_actions = teacher.get_action(obs, time_step)

    if teacher_mode == "residual_to_abs":
        # Teacher ONNX was residual-trained: residual * scale + motion_ref = target
        # Student (absolute): abs * scale + default_pos = target
        # → abs = residual + (motion_ref - default_pos) / scale
        motion_ref = env.command_manager.get_term("motion").joint_pos
        data = env.scene["robot"].data
        default_pos = getattr(data, "default_joint_pos_nominal", data.default_joint_pos[0])
        action_scale = env.action_manager.get_term("joint_pos")._scale[0]
        teacher_actions = teacher_actions + (motion_ref - default_pos) / action_scale

    diff = student_actions - teacher_actions
    return torch.exp(-torch.sum(diff**2, dim=-1) / (2.0 * sigma**2))
