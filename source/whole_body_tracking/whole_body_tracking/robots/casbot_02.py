import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# fmt: off
# Armature values per joint group
ARMATURE_LEG_PITCH = 0.06999046   # leg_pelvic_pitch, leg_knee_pitch
ARMATURE_LEG_ROLL  = 0.06999046   # leg_pelvic_roll
ARMATURE_LEG_YAW   = 0.03959369   # leg_pelvic_yaw
ARMATURE_ANKLE     = 0.03959369   # ankle_pitch, ankle_roll

ARMATURE_ARM_PITCH = 0.03298028   # shoulder_pitch, elbow_pitch
ARMATURE_ARM_ROLL  = 0.03298028   # shoulder_roll
ARMATURE_ARM_YAW   = 0.02452611   # shoulder_yaw, wrist_yaw
ARMATURE_WAIST_YAW = 0.06999046
ARMATURE_HEAD = 0.03298028

NATURAL_FREQ   = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO  = 2.0

def _stiffness(armature): return armature * NATURAL_FREQ**2
def _damping(armature):   return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ

# fmt: on

CASBOT_02_25DOF_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/casbot_skeleton_description/urdf/casbot_skeleton_25dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.875),
        joint_pos={
            ".*_leg_pelvic_pitch_joint": -0.1,
            ".*_leg_knee_pitch_joint":    0.5,
            ".*_leg_ankle_pitch_joint":  -0.175,
            ".*_leg_ankle_roll_joint":    0.0,
            "left_shoulder_pitch_joint":  0.0,
            "right_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint":   0.0,
            "right_shoulder_roll_joint":  0.0,
            ".*_elbow_pitch_joint":       -0.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_pelvic_pitch_joint",
                ".*_leg_pelvic_roll_joint",
                ".*_leg_pelvic_yaw_joint",
                ".*_leg_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_leg_pelvic_pitch_joint": 150.0,
                ".*_leg_pelvic_roll_joint":  150.0,
                ".*_leg_pelvic_yaw_joint":    60.0,
                ".*_leg_knee_pitch_joint":   150.0,
            },
            velocity_limit_sim=14.0,
            stiffness={
                ".*_leg_pelvic_pitch_joint": _stiffness(ARMATURE_LEG_PITCH),
                ".*_leg_pelvic_roll_joint":  _stiffness(ARMATURE_LEG_ROLL),
                ".*_leg_pelvic_yaw_joint":   _stiffness(ARMATURE_LEG_YAW),
                ".*_leg_knee_pitch_joint":   _stiffness(ARMATURE_LEG_PITCH),
            },
            damping={
                ".*_leg_pelvic_pitch_joint": _damping(ARMATURE_LEG_PITCH),
                ".*_leg_pelvic_roll_joint":  _damping(ARMATURE_LEG_ROLL),
                ".*_leg_pelvic_yaw_joint":   _damping(ARMATURE_LEG_YAW),
                ".*_leg_knee_pitch_joint":   _damping(ARMATURE_LEG_PITCH),
            },
            armature={
                ".*_leg_pelvic_pitch_joint": ARMATURE_LEG_PITCH,
                ".*_leg_pelvic_roll_joint":  ARMATURE_LEG_ROLL,
                ".*_leg_pelvic_yaw_joint":   ARMATURE_LEG_YAW,
                ".*_leg_knee_pitch_joint":   ARMATURE_LEG_PITCH,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_ankle_pitch_joint",
                ".*_leg_ankle_roll_joint",
            ],
            effort_limit_sim=60.0,
            velocity_limit_sim=14.0,
            stiffness=_stiffness(ARMATURE_ANKLE),
            damping=_damping(ARMATURE_ANKLE),
            armature=ARMATURE_ANKLE,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=14.0,
            stiffness=_stiffness(ARMATURE_WAIST_YAW),
            damping=_damping(ARMATURE_WAIST_YAW),
            armature=ARMATURE_WAIST_YAW,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 75.0,
                ".*_shoulder_roll_joint":  75.0,
                ".*_shoulder_yaw_joint":   36.0,
                ".*_elbow_pitch_joint":    75.0,
                ".*_wrist_yaw_joint":      36.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 12.2,
                ".*_shoulder_roll_joint":  12.2,
                ".*_shoulder_yaw_joint":    9.3,
                ".*_elbow_pitch_joint":    12.2,
                ".*_wrist_yaw_joint":       9.3,
            },
            stiffness={
                ".*_shoulder_pitch_joint": _stiffness(ARMATURE_ARM_PITCH),
                ".*_shoulder_roll_joint":  _stiffness(ARMATURE_ARM_ROLL),
                ".*_shoulder_yaw_joint":   _stiffness(ARMATURE_ARM_YAW),
                ".*_elbow_pitch_joint":    _stiffness(ARMATURE_ARM_PITCH),
                ".*_wrist_yaw_joint":      _stiffness(ARMATURE_ARM_YAW),
            },
            damping={
                ".*_shoulder_pitch_joint": _damping(ARMATURE_ARM_PITCH),
                ".*_shoulder_roll_joint":  _damping(ARMATURE_ARM_ROLL),
                ".*_shoulder_yaw_joint":   _damping(ARMATURE_ARM_YAW),
                ".*_elbow_pitch_joint":    _damping(ARMATURE_ARM_PITCH),
                ".*_wrist_yaw_joint":      _damping(ARMATURE_ARM_YAW),
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_ARM_PITCH,
                ".*_shoulder_roll_joint":  ARMATURE_ARM_ROLL,
                ".*_shoulder_yaw_joint":   ARMATURE_ARM_YAW,
                ".*_elbow_pitch_joint":    ARMATURE_ARM_PITCH,
                ".*_wrist_yaw_joint":      ARMATURE_ARM_YAW,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_yaw_joint",
                "head_pitch_joint",
            ],
            effort_limit_sim=36.0,
            velocity_limit_sim=9.3,
            stiffness=0.0,
            damping=0.0,
            armature=ARMATURE_HEAD,
        ),
    },
)

CASBOT_02_25DOF_ACTION_SCALE = {}
for _a in CASBOT_02_25DOF_CYLINDER_CFG.actuators.values():
    _e = _a.effort_limit_sim
    _s = _a.stiffness
    _names = _a.joint_names_expr
    if not isinstance(_e, dict):
        _e = {n: _e for n in _names}
    if not isinstance(_s, dict):
        _s = {n: _s for n in _names}
    for _n in _names:
        if _n in _e and _n in _s and _s[_n]:
            CASBOT_02_25DOF_ACTION_SCALE[_n] = 0.25 * _e[_n] / _s[_n]


CASBOT_02_25DOF_CYLINDER_WITH_HANDS_CFG = CASBOT_02_25DOF_CYLINDER_CFG.replace(
    spawn=CASBOT_02_25DOF_CYLINDER_CFG.spawn.replace(
        asset_path=f"{ASSET_DIR}/casbot_skeleton_description/urdf/casbot_skeleton_25dof_with_hands.urdf",
    ),
)

# fmt: off
# Explicit stiffness/damping per joint group (numerically same as formula, but hardcoded for direct tuning)
_S_LEG_PITCH = 276.311
_S_LEG_ROLL  = 276.311
_S_LEG_YAW   = 156.310
_S_KNEE      = 276.311
_S_ANKLE     = 156.310
_S_WAIST_YAW = 276.311
_S_SHOULDER_PITCH = 130.201
_S_SHOULDER_ROLL  = 130.201
_S_SHOULDER_YAW   = 96.825
_S_ELBOW      = 130.201
_S_WRIST_YAW  = 96.825

_D_LEG_PITCH = 17.591
_D_LEG_ROLL  = 17.591
_D_LEG_YAW   = 9.951
_D_KNEE      = 17.591
_D_ANKLE     = 9.951
_D_WAIST_YAW = 17.591
_D_SHOULDER_PITCH = 8.289
_D_SHOULDER_ROLL  = 8.289
_D_SHOULDER_YAW   = 6.164
_D_ELBOW      = 8.289
_D_WRIST_YAW  = 6.164
# fmt: on

CASBOT_02_25DOF_CYLINDER_DIRECT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/casbot_skeleton_description/urdf/casbot_skeleton_25dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.875),
        joint_pos={
            ".*_leg_pelvic_pitch_joint": -0.1,
            ".*_leg_knee_pitch_joint":    0.5,
            ".*_leg_ankle_pitch_joint":  -0.175,
            ".*_leg_ankle_roll_joint":    0.0,
            "left_shoulder_pitch_joint":  0.0,
            "right_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint":   0.0,
            "right_shoulder_roll_joint":  0.0,
            ".*_elbow_pitch_joint":       -0.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_pelvic_pitch_joint",
                ".*_leg_pelvic_roll_joint",
                ".*_leg_pelvic_yaw_joint",
                ".*_leg_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_leg_pelvic_pitch_joint": 150.0,
                ".*_leg_pelvic_roll_joint":  150.0,
                ".*_leg_pelvic_yaw_joint":    60.0,
                ".*_leg_knee_pitch_joint":   150.0,
            },
            velocity_limit_sim=14.0,
            stiffness={
                ".*_leg_pelvic_pitch_joint": _S_LEG_PITCH,
                ".*_leg_pelvic_roll_joint":  _S_LEG_ROLL,
                ".*_leg_pelvic_yaw_joint":   _S_LEG_YAW,
                ".*_leg_knee_pitch_joint":   _S_KNEE,
            },
            damping={
                ".*_leg_pelvic_pitch_joint": _D_LEG_PITCH,
                ".*_leg_pelvic_roll_joint":  _D_LEG_ROLL,
                ".*_leg_pelvic_yaw_joint":   _D_LEG_YAW,
                ".*_leg_knee_pitch_joint":   _D_KNEE,
            },
            armature={
                ".*_leg_pelvic_pitch_joint": ARMATURE_LEG_PITCH,
                ".*_leg_pelvic_roll_joint":  ARMATURE_LEG_ROLL,
                ".*_leg_pelvic_yaw_joint":   ARMATURE_LEG_YAW,
                ".*_leg_knee_pitch_joint":   ARMATURE_LEG_PITCH,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_ankle_pitch_joint",
                ".*_leg_ankle_roll_joint",
            ],
            effort_limit_sim=60.0,
            velocity_limit_sim=14.0,
            stiffness=_S_ANKLE,
            damping=_D_ANKLE,
            armature=ARMATURE_ANKLE,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=14.0,
            stiffness=_S_WAIST_YAW,
            damping=_D_WAIST_YAW,
            armature=ARMATURE_WAIST_YAW,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 75.0,
                ".*_shoulder_roll_joint":  75.0,
                ".*_shoulder_yaw_joint":   36.0,
                ".*_elbow_pitch_joint":    75.0,
                ".*_wrist_yaw_joint":      36.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 12.2,
                ".*_shoulder_roll_joint":  12.2,
                ".*_shoulder_yaw_joint":    9.3,
                ".*_elbow_pitch_joint":    12.2,
                ".*_wrist_yaw_joint":       9.3,
            },
            stiffness={
                ".*_shoulder_pitch_joint": _S_SHOULDER_PITCH,
                ".*_shoulder_roll_joint":  _S_SHOULDER_ROLL,
                ".*_shoulder_yaw_joint":   _S_SHOULDER_YAW,
                ".*_elbow_pitch_joint":    _S_ELBOW,
                ".*_wrist_yaw_joint":      _S_WRIST_YAW,
            },
            damping={
                ".*_shoulder_pitch_joint": _D_SHOULDER_PITCH,
                ".*_shoulder_roll_joint":  _D_SHOULDER_ROLL,
                ".*_shoulder_yaw_joint":   _D_SHOULDER_YAW,
                ".*_elbow_pitch_joint":    _D_ELBOW,
                ".*_wrist_yaw_joint":      _D_WRIST_YAW,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_ARM_PITCH,
                ".*_shoulder_roll_joint":  ARMATURE_ARM_ROLL,
                ".*_shoulder_yaw_joint":   ARMATURE_ARM_YAW,
                ".*_elbow_pitch_joint":    ARMATURE_ARM_PITCH,
                ".*_wrist_yaw_joint":      ARMATURE_ARM_YAW,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_yaw_joint",
                "head_pitch_joint",
            ],
            effort_limit_sim=36.0,
            velocity_limit_sim=9.3,
            stiffness=0.0,
            damping=0.0,
            armature=ARMATURE_HEAD,
        ),
    },
)

CASBOT_02_25DOF_DIRECT_ACTION_SCALE = {
    "left_leg_pelvic_pitch_joint":  0.136,
    "right_leg_pelvic_pitch_joint": 0.136,
    "left_leg_pelvic_roll_joint":   0.136,
    "right_leg_pelvic_roll_joint":  0.136,
    "left_leg_pelvic_yaw_joint":    0.096,
    "right_leg_pelvic_yaw_joint":   0.096,
    "left_leg_knee_pitch_joint":    0.136,
    "right_leg_knee_pitch_joint":   0.136,
    "left_leg_ankle_pitch_joint":   0.096,
    "right_leg_ankle_pitch_joint":  0.096,
    "left_leg_ankle_roll_joint":    0.096,
    "right_leg_ankle_roll_joint":   0.096,
    "waist_yaw_joint":              0.054,
    "left_shoulder_pitch_joint":    0.144,
    "right_shoulder_pitch_joint":   0.144,
    "left_shoulder_roll_joint":     0.144,
    "right_shoulder_roll_joint":    0.144,
    "left_shoulder_yaw_joint":      0.093,
    "right_shoulder_yaw_joint":     0.093,
    "left_elbow_pitch_joint":       0.144,
    "right_elbow_pitch_joint":      0.144,
    "left_wrist_yaw_joint":         0.093,
    "right_wrist_yaw_joint":        0.093,
    "head_yaw_joint":               1.0,
    "head_pitch_joint":             1.0,
}
