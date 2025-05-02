# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional

import isaacsim.core.api.tasks as tasks
import numpy as np
from isaacsim.examples.interactive.lib_robot.example import Franka
from isaacsim.core.api.objects import VisualCuboid

from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.string import find_unique_string_name


class FollowTarget(tasks.FollowTarget):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to "franka_follow_target".
        target_prim_path (Optional[str], optional): [description]. Defaults to None.
        target_name (Optional[str], optional): [description]. Defaults to None.
        target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        target_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        franka_prim_path (Optional[str], optional): [description]. Defaults to None.
        franka_robot_name (Optional[str], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str = "franka_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        franka_prim_path: Optional[str] = None,
        franka_robot_name: Optional[str] = None,
    ) -> None:
        
        if target_prim_path is not None and target_name is None:
            target_name = target_prim_path.split("/")[-1]
        
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=None,
            target_orientation=None,
            offset=None,
        )

        self._franka_prim_path = franka_prim_path
        self._franka_robot_name = franka_robot_name

        self._physics_sim_view = None
        return
    
    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        self._target = self.scene.get_object(self._target_name)
        self._task_objects[self._target_name] = self._target
        self._target_visual_material = self._target.get_applied_visual_material()

        self._robot = self.set_robot()
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()

    def set_params(self, *args, **kwargs):
        return

    def set_robot(self) -> Franka:
        if self._franka_prim_path is None:
            self._franka_prim_path = f"/World/Franka_{self.name}"  # e.g. /World/Franka_task0
        if self._franka_robot_name is None:
            self._franka_robot_name = f"my_franka_{self.name}"      # e.g. my_franka_task0

        robot = Franka(prim_path=self._franka_prim_path, name=self._franka_robot_name)

        if hasattr(self, "_init_pose"):
            pos, ori = self._init_pose
            robot.set_world_pose(position=pos, orientation=ori)

        return robot
