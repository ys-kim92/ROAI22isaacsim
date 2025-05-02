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
from isaacsim.examples.interactive.lib_robot.example.controllers.stacking_controller import StackingController
from isaacsim.examples.interactive.lib_robot.example.tasks import FollowTarget as FollowTargetTask
from isaacsim.examples.interactive.lib_robot.example.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat


#+++++ Custom 모듈
from isaacsim.examples.interactive.lib_module.data_io import DataIO


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
            (np.array([0, 0, 0]), euler_angles_to_quat(np.array([0, 0, 0]))),      
            (np.array([0, 2, 0]), euler_angles_to_quat(np.array([0, 0, 0]))),     
            (np.array([2.5, 0, 0]), euler_angles_to_quat(np.array([0, 0, np.pi]))),    
            (np.array([2.5, 2, 0]), euler_angles_to_quat(np.array([0, 0, np.pi]))),  
        ]
        self._num_of_tasks = len(self._robot_poses)  # 로봇 대수
        return
    
    #+++++ Scene build

    def setup_scene(self):
        world = self.get_world()

        # Scene & Task 추가
        for i in range(self._num_of_tasks):
            task = FollowTargetTask(name="task" + str(i))
            task._init_pose = self._robot_poses[i]
            
            world.add_task(task)
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

            controller = RMPFlowController(
                name="target_follower_controller"+ str(i), 
                robot_articulation=robot
            )
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
        observations = self._world.get_observations()

        for i in range(self._num_of_tasks):
            target_name = self._task_params[i]["target_name"]["value"]
            target_position = observations[target_name]["position"]
            target_orientation = observations[target_name]["orientation"]

            actions = self._controllers[i].forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=target_orientation,
            )
            self._articulation_controllers[i].apply_action(actions)

            #actions = self._controllers[i].forward(observations=observations, end_effector_offset=np.array([0, 0, 0]))
            #self._articulation_controllers[i].apply_action(actions)
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
