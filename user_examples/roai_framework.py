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

#+++++ Custom 모듈
from isaacsim.examples.interactive.lib_module.data_io import DataIO
from isaacsim.examples.interactive.lib_module.goal_related import *


class GoalValidation(RoaiBaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._robots = []
        self._tasks = []
        self._task_params = []
        self._controllers = []
        self._articulation_controllers = []

        # 설정값
        self._log_freq = 10     # FPS/n
        self._robot_poses = [
            # pos                 # rot (yaw) -> quaternion
            (np.array([1.5, 0.8, 0]), euler_angles_to_quat(np.array([0, 0, np.pi]))),      
            (np.array([1.5, -0.8, 0]), euler_angles_to_quat(np.array([0, 0, np.pi]))),     
            (np.array([-1.5, 0.8, 0]), euler_angles_to_quat(np.array([0, 0, 0]))),    
            (np.array([-1.5, -0.8, 0]), euler_angles_to_quat(np.array([0, 0, 0]))),  
        ]
        self._num_of_tasks = len(self._robot_poses)  # 로봇 대수
        self._target_reach_threshold = 0.05
        self._teleport_timeout = 3
        self._planning_mode = 0         # 0: RMPflow, 1: RRT
        return
    
    #+++++ Scene build

    def setup_scene(self):
        world = self.get_world()

        # goal 추가
        GoalRelated._visualize_every_goals(world.scene, filename="goals.json")

        self._shared_target_path = "/World/SharedTarget"
        self._shared_target_name = "shared_target"

        if not is_prim_path_valid(self._shared_target_path):
            target = VisualCuboid(
                name=self._shared_target_name,
                prim_path=self._shared_target_path,
                position=np.array([0, 0, 0.7]),  # 초기 위치
                orientation=euler_angles_to_quat(np.array([0, 0, 0])),
                color=np.array([0, 1, 0]),
                size=1.0,
                scale=np.array([0.03, 0.03, 0.03]) / get_stage_units(),
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

            task._init_pose = self._robot_poses[i]
            world.add_task(task)

        # Target goal 추가
        self._goal_sequence = GoalLoader._load_goals_from_file("goals.json")
        self._num_of_goals = len(self._goal_sequence)
        self._current_target_index = -1
        self._fsm_timer = None
        self._fsm_finished = False
        return

    async def setup_post_load(self):
        # 로봇별 셋업
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
        if val:
            world.add_physics_callback("sim_step", self._physics_step)
            await world.play_async()
            DataIO._on_logging_event(self, self._log_freq)
        else:
            world.remove_physics_callback("sim_step")
            DataIO._on_save_data_event(self,log_path)
        return
    
    def _physics_step(self, step_size):
        if self._fsm_finished:
            return
    
        current_time = self._world.current_time

        # shared target 위치 변경
        if self._current_target_index == -1:
            GoalRelated._teleport_next_target(self)
            #return

        if (current_time - self._fsm_timer > self._teleport_timeout):
            GoalRelated._teleport_next_target(self)

            if self._planning_mode == 1:
                for controller in self._controllers:
                    controller.reset()

        # robot 제어
        observations = self._world.get_observations()

        target_position = observations["shared_target"]["position"]
        target_orientation = observations["shared_target"]["orientation"]

        for i in range(self._num_of_tasks):
            if self._planning_mode == 0:
                local_pos = target_position
                local_ori = target_orientation

            elif self._planning_mode == 1:
                base_position, base_orientation = self._robots[i].get_world_pose()

                local_pos, local_ori = GoalRelated.transform_pose_to_local_frame(
                    target_position,
                    target_orientation,
                    base_position,
                    base_orientation
                )

            actions = self._controllers[i].forward(
                target_end_effector_position=local_pos,
                target_end_effector_orientation=local_ori,
            )
            if self._planning_mode == 1:
                kps, kds = self._tasks[i].get_custom_gains()
                self._articulation_controllers[i].set_gains(kps, kds)
            self._articulation_controllers[i].apply_action(actions)
        return

    #+++++ Reset 버튼 동작 후 실행

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
            for i in range(len(self._controllers)):
                self._controllers[i].reset()
        return

    def world_cleanup(self):
        self._robots = []
        self._tasks = []
        self._task_params = []
        self._controllers = []
        self._articulation_controllers = []
        return
