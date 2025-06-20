# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
from isaacsim.examples.interactive.base_sample import RoaiBaseSample
from isaacsim.examples.interactive.lib_robot.example.tasks import Stacking
from isaacsim.examples.interactive.path_planning.roai_path_planning_controller import FrankaRrtController
from isaacsim.examples.interactive.path_planning.roai_path_planning_task import FrankaPathPlanningTask
from isaacsim.examples.interactive.lib_robot.example.controllers.stacking_controller import StackingController
from isaacsim.examples.interactive.lib_robot.example.tasks import FollowTarget as FollowTargetTask
from isaacsim.examples.interactive.lib_robot.example.controllers.rmpflow_controller import RMPFlowController
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.api.objects import VisualCone
import time
import json

#+++++ Custom 모듈
from isaacsim.examples.interactive.lib_module.data_io import DataIO
from isaacsim.examples.interactive.lib_module.goal_related import *
from isaacsim.examples.interactive.lib_module.default_config import RoaiDefaultConfig


class GoalValidation(RoaiBaseSample):
    def __init__(self) -> None:
        super().__init__()

        # 파라미터 초기화
        default_config = RoaiDefaultConfig()
        
        for key, value in default_config.__dict__.items():
            setattr(self, key, value)

        # 아래부터는 추후 urdf에서 robot initial position 입력받으면 필요없는 코드
        self._robot_poses = [
            # pos                 # rot (yaw) -> quaternion
            (np.array([1.5, 0.8, 0]), euler_angles_to_quat(np.array([0, 0, np.pi]))),      
            (np.array([1.5, -0.8, 0]), euler_angles_to_quat(np.array([0, 0, np.pi]))),     
            (np.array([-1.5, 0.8, 0]), euler_angles_to_quat(np.array([0, 0, 0]))),    
            (np.array([-1.5, -0.8, 0]), euler_angles_to_quat(np.array([0, 0, 0]))),  
        ]   
        self._num_of_tasks = len(self._robot_poses)  # 로봇 대수
        return
    
    #+++++ Scene build

    def setup_scene(self):
        world = self.get_world()

        # 상세 config override
        base_dir = os.path.dirname(__file__)
        override_config_file = os.path.join(base_dir, "..", "lib_module", "override_config.json")
        abs_path = os.path.abspath(override_config_file)

        with open(abs_path, "r") as f:
            update_config = json.load(f)

        for key, value in update_config.items():
            target_key = f"_{key}"
            if hasattr(self, target_key) and (target_key not in RoaiDefaultConfig.PROTECTED):
                setattr(self, target_key, value)

        # goal 추가
        self._goal_sequence = GoalRelated._visualize_every_goals(world.scene, filename="goals.json")
        self._num_of_goals = len(self._goal_sequence)

        self._shared_target_path = "/World/SharedTarget"
        self._shared_target_name = "SharedTarget"

        if not is_prim_path_valid(self._shared_target_path):
            target = VisualCone(
                name=self._shared_target_name,
                prim_path=self._shared_target_path,
                position=np.array([0, 0, 0.7]),  # 초기 위치
                color=np.array([0, 1, 0]),          # green
                orientation=euler_angles_to_quat(np.array([0, 0, 0])),
                height=0.05,                  
                radius=0.02
            )
            world.scene.add(target)

        # 로봇 추가
        for i in range(self._num_of_tasks):
            if self._planning_mode == 0:
                task = FollowTargetTask(
                    name="task" + str(i),
                    target_prim_path=self._shared_target_path,
                    target_name=self._shared_target_name
                )
            elif self._planning_mode == 1:
                task = FrankaPathPlanningTask(
                    name="task" + str(i),
                    target_prim_path=self._shared_target_path,
                    target_name=self._shared_target_name
                )

            task.sample = self      # task나 controller에서 접근하기 위해 sample 변수 선언

            task._init_pose = self._robot_poses[i]
            world.add_task(task)
        return

    async def setup_post_load(self):
        # 로봇별 task 셋업
        for i in range(self._num_of_tasks):
            task = self._world.get_task(name="task" + str(i))
            self._tasks.append(task)
            params = task.get_params()
            self._task_params.append(params)

            robot = self._world.scene.get_object(params["robot_name"]["value"])
            self._robots.append(robot)

            task._physics_isim_view = self._world.physics_sim_view
            task._robot.initialize(task._physics_sim_view)

            if self._planning_mode == 0:
                controller = RMPFlowController(
                    name="target_follower_controller"+ str(i), 
                    robot_articulation=robot
                )
            elif self._planning_mode == 1:
                controller = FrankaRrtController(name="target_follower_controller"+ str(i), robot_articulation=robot)


            self._controllers.append(controller)
            articulation_controller = robot.get_articulation_controller()
            self._articulation_controllers.append(articulation_controller)
        return

    #+++++ Start 버튼 동작 후 실행

    async def _on_task_event_async(self, val, log_path):
        world = self.get_world()
        self._log_path = log_path

        if val:
            world.add_physics_callback("sim_step", self._physics_step)
            await world.play_async()
            self._ini_time_sim = self._world.current_time
            self._ini_time_real = time.time()
        else:
            world.remove_physics_callback("sim_step")
            DataIO._on_save_data_event(self,log_path)
            print("+++++++ Goal validation stop +++++++")
            self.world_cleanup()
        return
    
    def _physics_step(self, step_size):
        if self._fsm_finished:
            DataIO._on_save_data_event(self, self._log_path)
            print("+++++++ Goal validation finisih +++++++")
            self.world_cleanup()
            return
    
        current_time = self._world.current_time
        self._goal_rotation_done = False

        # shared target 초기 위치
        if self._current_target_index == -1:
            GoalRelated._move_to_next_target(self)
            self._fsm_timer = current_time
            return

        # robot 제어
        observations = self._world.get_observations()

        target_position = observations["SharedTarget"]["position"]
        target_orientation = observations["SharedTarget"]["orientation"]    # quaternian wxyz 순서

        robot = self._robots[self._current_robot_index]
        controller = self._controllers[self._current_robot_index]
        articulation_controller = self._articulation_controllers[self._current_robot_index]
           
        if self._planning_mode == 0:
            local_pos = target_position
            local_ori = target_orientation

            actions = controller.forward(
                target_end_effector_position=local_pos,
                target_end_effector_orientation=local_ori,
            )

            articulation_controller.apply_action(actions)

            if controller._success_flag:
                self._any360_reached_flag = True
                if (current_time - self._fsm_timer > self._goal_move_timeout):
                    DataIO._on_logging_event(self, 0)
                    self._goal_rotation_done = True
            else:
                self._goal_rotation_done = True

        elif self._planning_mode == 1:
            # base_position, base_orientation = robot.get_world_pose()  # quaternian wxyz 순서
            if not self._goal_rotation_done:
                GoalRelated._goal_Z_rotation(self)

        if self._goal_rotation_done:
            # shared target 위치 변경
            GoalRelated._move_to_next_target(self)
        
            if self._planning_mode == 1:
                controller.reset()
                controller._success_flag = False
                self._reached_flag = False
                self._any360_reached_flag = False
                self._fsm_timer = current_time
                self._current_Z_rotation_angle = 0

        return

    #+++++ Reset 버튼 관련 실행

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        return
    
    async def setup_post_reset(self):
        # reset 후 task 재설정, start 가능
        await self.setup_post_load()
        return

    def world_cleanup(self):
        self._robots = []
        self._tasks = []
        self._task_params = []
        self._controllers = []
        self._articulation_controllers = []
        self._current_robot_index = 0
        self._current_target_index = -1
        self._reached_flag = False
        self._any360_reached_flag = False
        self._goal_rotation_done = False
        self._current_Z_rotation_angle = 0

        self._world.stop()
        self._world.clear_all_callbacks()
 
        return