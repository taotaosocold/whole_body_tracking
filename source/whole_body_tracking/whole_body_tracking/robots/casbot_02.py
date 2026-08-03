import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots.g1 import (
    ARMATURE_4010 as G1_ARMATURE_4010,
    ARMATURE_5020 as G1_ARMATURE_5020,
    ARMATURE_7520_14 as G1_ARMATURE_7520_14,
    ARMATURE_7520_22 as G1_ARMATURE_7520_22,
    DAMPING_4010 as G1_DAMPING_4010,
    DAMPING_5020 as G1_DAMPING_5020,
    DAMPING_7520_14 as G1_DAMPING_7520_14,
    DAMPING_7520_22 as G1_DAMPING_7520_22,
    STIFFNESS_4010 as G1_STIFFNESS_4010,
    STIFFNESS_5020 as G1_STIFFNESS_5020,
    STIFFNESS_7520_14 as G1_STIFFNESS_7520_14,
    STIFFNESS_7520_22 as G1_STIFFNESS_7520_22,
)

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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.875),
        joint_pos={
            ".*_leg_pelvic_pitch_joint": -0.1,
            ".*_leg_knee_pitch_joint": 0.5,
            ".*_leg_ankle_pitch_joint": -0.175,
            ".*_leg_ankle_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            ".*_elbow_pitch_joint": -0.5,
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
            stiffness={"head_yaw_joint": 49.159, "head_pitch_joint": 58.668},
            damping={"head_yaw_joint": 3.130, "head_pitch_joint": 3.735},
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
            CASBOT_02_25DOF_ACTION_SCALE[_n] = 0.75 * _e[_n] / _s[_n]

CASBOT_02_25DOF_ACTION_SCALE["head_yaw_joint"] = 0.15
CASBOT_02_25DOF_ACTION_SCALE["head_pitch_joint"] = 0.15


CASBOT_02_25DOF_CYLINDER_WITH_HANDS_CFG = CASBOT_02_25DOF_CYLINDER_CFG.replace(
    spawn=CASBOT_02_25DOF_CYLINDER_CFG.spawn.replace(
        asset_path=f"{ASSET_DIR}/casbot_skeleton_description/urdf/casbot_skeleton_25dof_with_hands.urdf",
    ),
)

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
            "left_leg_pelvic_pitch_joint": -0.1,
            "right_leg_pelvic_pitch_joint": -0.1,
            "waist_yaw_joint": 0.0,
            "left_leg_pelvic_roll_joint": 0.0,
            "right_leg_pelvic_roll_joint": 0.0,
            "head_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.1,
            "right_shoulder_pitch_joint": 0.1,
            "left_leg_pelvic_yaw_joint": 0.3,
            "right_leg_pelvic_yaw_joint": -0.3,
            "head_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.4,
            "right_shoulder_roll_joint": -0.4,
            "left_leg_knee_pitch_joint": 0.2,
            "right_leg_knee_pitch_joint": 0.2,
            "left_shoulder_yaw_joint": -0.4,
            "right_shoulder_yaw_joint": 0.4,
            "left_leg_ankle_pitch_joint": -0.1,
            "right_leg_ankle_pitch_joint": -0.1,
            "left_elbow_pitch_joint": -0.8,
            "right_elbow_pitch_joint": -0.8,
            "left_leg_ankle_roll_joint": 0.0,
            "right_leg_ankle_roll_joint": 0.0,
            "left_wrist_yaw_joint": 0.16,
            "right_wrist_yaw_joint": -0.16,
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
                "left_leg_pelvic_pitch_joint": 143.097,
                "right_leg_pelvic_pitch_joint": 143.117,
                "left_leg_pelvic_roll_joint": 141.995,
                "right_leg_pelvic_roll_joint": 141.996,
                "left_leg_pelvic_yaw_joint": 90.141,
                "right_leg_pelvic_yaw_joint": 90.137,
                "left_leg_knee_pitch_joint": 245.677,
                "right_leg_knee_pitch_joint": 245.799,
            },
            damping={
                "left_leg_pelvic_pitch_joint": 9.110,
                "right_leg_pelvic_pitch_joint": 9.111,
                "left_leg_pelvic_roll_joint": 9.040,
                "right_leg_pelvic_roll_joint": 9.040,
                "left_leg_pelvic_yaw_joint": 5.739,
                "right_leg_pelvic_yaw_joint": 5.738,
                "left_leg_knee_pitch_joint": 15.640,
                "right_leg_knee_pitch_joint": 15.648,
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
            stiffness={
                "left_leg_ankle_pitch_joint": 78.167,
                "right_leg_ankle_pitch_joint": 78.167,
                "left_leg_ankle_roll_joint": 81.054,
                "right_leg_ankle_roll_joint": 81.475,
            },
            damping={
                "left_leg_ankle_pitch_joint": 4.976,
                "right_leg_ankle_pitch_joint": 4.976,
                "left_leg_ankle_roll_joint": 5.160,
                "right_leg_ankle_roll_joint": 5.187,
            },
            armature=ARMATURE_ANKLE,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=14.0,
            stiffness=223.930,
            damping=14.256,
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
                "left_shoulder_pitch_joint": 65.792,
                "right_shoulder_pitch_joint": 65.796,
                "left_shoulder_roll_joint": 67.208,
                "right_shoulder_roll_joint": 67.207,
                "left_shoulder_yaw_joint": 48.653,
                "right_shoulder_yaw_joint": 48.653,
                "left_elbow_pitch_joint": 75.965,
                "right_elbow_pitch_joint": 75.965,
                "left_wrist_yaw_joint": 48.361,
                "right_wrist_yaw_joint": 48.674,
            },
            damping={
                "left_shoulder_pitch_joint": 4.188,
                "right_shoulder_pitch_joint": 4.189,
                "left_shoulder_roll_joint": 4.279,
                "right_shoulder_roll_joint": 4.279,
                "left_shoulder_yaw_joint": 3.097,
                "right_shoulder_yaw_joint": 3.097,
                "left_elbow_pitch_joint": 4.835,
                "right_elbow_pitch_joint": 4.836,
                "left_wrist_yaw_joint": 3.079,
                "right_wrist_yaw_joint": 3.099,
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
            stiffness={"head_yaw_joint": 49.159, "head_pitch_joint": 58.668},
            damping={"head_yaw_joint": 3.130, "head_pitch_joint": 3.735},
            armature=ARMATURE_HEAD,
        ),
    },
)

CASBOT_02_25DOF_DIRECT_ACTION_SCALE = {
    "left_leg_pelvic_pitch_joint": 0.5,
    "right_leg_pelvic_pitch_joint": 0.5,
    "waist_yaw_joint": 0.5,
    "left_leg_pelvic_roll_joint": 0.5,
    "right_leg_pelvic_roll_joint": 0.5,
    "head_yaw_joint": 0.15,
    "left_shoulder_pitch_joint": 0.4,
    "right_shoulder_pitch_joint": 0.4,
    "left_leg_pelvic_yaw_joint": 0.5,
    "right_leg_pelvic_yaw_joint": 0.5,
    "head_pitch_joint": 0.15,
    "left_shoulder_roll_joint": 0.4,
    "right_shoulder_roll_joint": 0.4,
    "left_leg_knee_pitch_joint": 0.5,
    "right_leg_knee_pitch_joint": 0.5,
    "left_shoulder_yaw_joint": 0.4,
    "right_shoulder_yaw_joint": 0.4,
    "left_leg_ankle_pitch_joint": 0.5,
    "right_leg_ankle_pitch_joint": 0.5,
    "left_elbow_pitch_joint": 0.4,
    "right_elbow_pitch_joint": 0.4,
    "left_leg_ankle_roll_joint": 0.5,
    "right_leg_ankle_roll_joint": 0.5,
    "left_wrist_yaw_joint": 0.2,
    "right_wrist_yaw_joint": 0.2,
}


CASBOT_02_25DOF_CYLINDER_G1_CFG = CASBOT_02_25DOF_CYLINDER_CFG.replace(
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_pelvic_pitch_joint",
                ".*_leg_pelvic_roll_joint",
                ".*_leg_pelvic_yaw_joint",
                ".*_leg_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_leg_pelvic_pitch_joint": 88.0,
                ".*_leg_pelvic_roll_joint": 139.0,
                ".*_leg_pelvic_yaw_joint": 88.0,
                ".*_leg_knee_pitch_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_leg_pelvic_pitch_joint": 32.0,
                ".*_leg_pelvic_roll_joint": 20.0,
                ".*_leg_pelvic_yaw_joint": 32.0,
                ".*_leg_knee_pitch_joint": 20.0,
            },
            stiffness={
                ".*_leg_pelvic_pitch_joint": G1_STIFFNESS_7520_14,
                ".*_leg_pelvic_roll_joint": G1_STIFFNESS_7520_22,
                ".*_leg_pelvic_yaw_joint": G1_STIFFNESS_7520_14,
                ".*_leg_knee_pitch_joint": G1_STIFFNESS_7520_22,
            },
            damping={
                ".*_leg_pelvic_pitch_joint": G1_DAMPING_7520_14,
                ".*_leg_pelvic_roll_joint": G1_DAMPING_7520_22,
                ".*_leg_pelvic_yaw_joint": G1_DAMPING_7520_14,
                ".*_leg_knee_pitch_joint": G1_DAMPING_7520_22,
            },
            armature={
                ".*_leg_pelvic_pitch_joint": G1_ARMATURE_7520_14,
                ".*_leg_pelvic_roll_joint": G1_ARMATURE_7520_22,
                ".*_leg_pelvic_yaw_joint": G1_ARMATURE_7520_14,
                ".*_leg_knee_pitch_joint": G1_ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_leg_ankle_pitch_joint", ".*_leg_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * G1_STIFFNESS_5020,
            damping=2.0 * G1_DAMPING_5020,
            armature=2.0 * G1_ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=G1_STIFFNESS_7520_14,
            damping=G1_DAMPING_7520_14,
            armature=G1_ARMATURE_7520_14,
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
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_pitch_joint": 25.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_pitch_joint": 37.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": G1_STIFFNESS_5020,
                ".*_shoulder_roll_joint": G1_STIFFNESS_5020,
                ".*_shoulder_yaw_joint": G1_STIFFNESS_5020,
                ".*_elbow_pitch_joint": G1_STIFFNESS_5020,
                ".*_wrist_yaw_joint": G1_STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": G1_DAMPING_5020,
                ".*_shoulder_roll_joint": G1_DAMPING_5020,
                ".*_shoulder_yaw_joint": G1_DAMPING_5020,
                ".*_elbow_pitch_joint": G1_DAMPING_5020,
                ".*_wrist_yaw_joint": G1_DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": G1_ARMATURE_5020,
                ".*_shoulder_roll_joint": G1_ARMATURE_5020,
                ".*_shoulder_yaw_joint": G1_ARMATURE_5020,
                ".*_elbow_pitch_joint": G1_ARMATURE_5020,
                ".*_wrist_yaw_joint": G1_ARMATURE_4010,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim=36.0,
            velocity_limit_sim=9.3,
            stiffness={"head_yaw_joint": 49.159, "head_pitch_joint": 58.668},
            damping={"head_yaw_joint": 3.130, "head_pitch_joint": 3.735},
            armature=ARMATURE_HEAD,
        ),
    },
)


CASBOT_02_25DOF_G1_ACTION_SCALE = {
    "left_leg_pelvic_pitch_joint": 0.25 * 88.0 / G1_STIFFNESS_7520_14,
    "right_leg_pelvic_pitch_joint": 0.25 * 88.0 / G1_STIFFNESS_7520_14,
    "left_leg_pelvic_roll_joint": 0.25 * 139.0 / G1_STIFFNESS_7520_22,
    "right_leg_pelvic_roll_joint": 0.25 * 139.0 / G1_STIFFNESS_7520_22,
    "left_leg_pelvic_yaw_joint": 0.25 * 88.0 / G1_STIFFNESS_7520_14,
    "right_leg_pelvic_yaw_joint": 0.25 * 88.0 / G1_STIFFNESS_7520_14,
    "left_leg_knee_pitch_joint": 0.25 * 139.0 / G1_STIFFNESS_7520_22,
    "right_leg_knee_pitch_joint": 0.25 * 139.0 / G1_STIFFNESS_7520_22,
    "left_leg_ankle_pitch_joint": 0.25 * 50.0 / (2.0 * G1_STIFFNESS_5020),
    "right_leg_ankle_pitch_joint": 0.25 * 50.0 / (2.0 * G1_STIFFNESS_5020),
    "left_leg_ankle_roll_joint": 0.25 * 50.0 / (2.0 * G1_STIFFNESS_5020),
    "right_leg_ankle_roll_joint": 0.25 * 50.0 / (2.0 * G1_STIFFNESS_5020),
    "waist_yaw_joint": 0.25 * 88.0 / G1_STIFFNESS_7520_14,
    "left_shoulder_pitch_joint": 0.25 * 25.0 / G1_STIFFNESS_5020,
    "right_shoulder_pitch_joint": 0.25 * 25.0 / G1_STIFFNESS_5020,
    "left_shoulder_roll_joint": 0.25 * 25.0 / G1_STIFFNESS_5020,
    "right_shoulder_roll_joint": 0.25 * 25.0 / G1_STIFFNESS_5020,
    "left_shoulder_yaw_joint": 0.25 * 25.0 / G1_STIFFNESS_5020,
    "right_shoulder_yaw_joint": 0.25 * 25.0 / G1_STIFFNESS_5020,
    "left_elbow_pitch_joint": 0.25 * 25.0 / G1_STIFFNESS_5020,
    "right_elbow_pitch_joint": 0.25 * 25.0 / G1_STIFFNESS_5020,
    "left_wrist_yaw_joint": 0.25 * 5.0 / G1_STIFFNESS_4010,
    "right_wrist_yaw_joint": 0.25 * 5.0 / G1_STIFFNESS_4010,
    "head_yaw_joint": 0.15,
    "head_pitch_joint": 0.15,
}
