from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class RoaiDefaultConfig:
    # --- 초기값 ---
    _robots: List = field(default_factory=list)
    _tasks: List = field(default_factory=list)
    _task_params: List = field(default_factory=list)
    _controllers: List = field(default_factory=list)
    _articulation_controllers: List = field(default_factory=list)
    _success_goal: Dict = field(default_factory=dict)
    _log_path: str = ""
    _current_robot_index: int = 0
    _current_target_index: int = -1
    _reached_flag: bool = False
    _any360_reached_flag: bool = False
    _goal_rotation_done: bool = False
    _current_Z_rotation_angle: float = 0
    _fsm_timer: Optional[float] = None
    _ini_time_sim: float = 0
    _ini_time_real: float = 0
    _fsm_finished: bool = False

    # 초기값 override 불가능 설정
    PROTECTED = [
        "_robots", "_tasks", "_task_params", "_controllers", "_articulation_controllers",
        "_success_goal", "_log_path", "_current_robot_index", "_current_target_index",
        "_reached_flag", "_any360_reached_flag", "_goal_rotation_done", "_current_Z_rotation_angle",
        "_fsm_timer", "_ini_time_sim", "_ini_time_real", "_fsm_finished"
    ]

    # --- 설정값, override 가능 ---
    _num_of_tasks: int = 4                  # 로봇 대수
    _target_360_resolution: int = 90        # deg
    _goal_move_timeout: float = 1           # IK 성공 시 GUI 모션 대기 시간
    _planning_mode: int = 1                 # 0: RMPflow, 1: RRT
    