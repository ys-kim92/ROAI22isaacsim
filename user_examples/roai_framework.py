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
from isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller import StackingController
from isaacsim.robot.manipulators.examples.franka.tasks import Stacking

#+++++ Custom 모듈
from isaacsim.examples.interactive.user_examples.data_io import DataIO


class GoalValidation(RoaiBaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._robots = []
        self._tasks = []
        self._controllers = []
        self._articulation_controllers = []

        # 설정값
        self._num_of_tasks = 4
        return
    
    #+++++ Scene build

    def setup_scene(self):
        world = self.get_world()

        # Task 추가 (*Follow target 반영)
        for i in range(self._num_of_tasks):
            task = Stacking(name="task" + str(i), offset=np.array([0, (i * 2) - 3, 0]))
            world.add_task(task)
        return

    async def setup_post_load(self):
        for i in range(self._num_of_tasks):
            self._tasks.append(self._world.get_task(name="task" + str(i)))
        for i in range(self._num_of_tasks):
            self._robots.append(self._world.scene.get_object(self._tasks[i].get_params()["robot_name"]["value"]))
            
            # Controller 추가 (*Follow target 반영)
            self._controllers.append(
                StackingController(
                    name="stacking_controller",
                    gripper=self._robots[i].gripper,
                    robot_articulation=self._robots[i],
                    picking_order_cube_names=self._tasks[i].get_cube_names(),
                    robot_observation_name=self._robots[i].name,
                )
            )
        for i in range(self._num_of_tasks):
            self._articulation_controllers.append(self._robots[i].get_articulation_controller())
        return

    #+++++ Start 버튼 동작 후 실행

    async def _on_task_event_async(self, val, log_path):
        world = self.get_world()
        if val:
            world.add_physics_callback("sim_step", self._physics_step)
            await world.play_async()
            DataIO._on_logging_event(self)
        else:
            world.remove_physics_callback("sim_step")
            DataIO._on_save_data_event(self,log_path)
        return
    
    def _physics_step(self, step_size):
        observations = self._world.get_observations()
        for i in range(self._num_of_tasks):
            actions = self._controllers[i].forward(observations=observations, end_effector_offset=np.array([0, 0, 0]))
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
        self._controllers = []
        self._articulation_controllers = []
        return
