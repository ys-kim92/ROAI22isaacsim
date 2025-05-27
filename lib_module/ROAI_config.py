import numpy as np
from isaacsim.core.utils.rotations import euler_angles_to_quat

settings = {
    # --- 상태 초기값 ---
    "default_state": {
        "_robots": [],
        "_tasks": [],
        "_task_params": [],
        "_controllers": [],
        "_articulation_controllers": [],
        "_success_goal": {},
        "_log_path": "",
        "_current_robot_index": 0,
        "_current_target_index": -1,
        "_reached_flag": False,
        "_any360_reached_flag": False,
        "_goal_rotation_done": False,
        "_current_Z_rotation_angle": 0,
        "_fsm_timer": None,
        "_ini_time_sim": 0,
        "_ini_time_real": 0,
        "_fsm_finished": False,
    },

    # --- 설정 값 ---
    "configuration": {
        "_robot_poses": [
            # pos                 # rot (yaw) -> quaternion
            (np.array([1.5, 0.8, 0]), euler_angles_to_quat(np.array([0, 0, np.pi]))),
            (np.array([1.5, -0.8, 0]), euler_angles_to_quat(np.array([0, 0, np.pi]))),
            (np.array([-1.5, 0.8, 0]), euler_angles_to_quat(np.array([0, 0, 0]))),
            (np.array([-1.5, -0.8, 0]), euler_angles_to_quat(np.array([0, 0, 0]))),
        ], # 추후 urdf에 robot initial position 입력받으면 필요없는 코드
        "_target_360_resolution": 90,
        "_goal_move_timeout": 1,        # deg
        "_planning_mode": 1,            # 0: RMPflow, 1: RRT
    }
}