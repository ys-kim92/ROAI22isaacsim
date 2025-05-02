from isaacsim.examples.interactive.base_sample import RoaiBaseSample
from isaacsim.core.utils.rotations import euler_angles_to_quat
import json
import os
import numpy as np
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.prims import create_prim



class GoalRelated(RoaiBaseSample):
    def _teleport_next_target(self):
        self._current_target_index += 1
        if self._current_target_index >= self._num_of_goals:
            self._fsm_finished = True
            return

        goal = self._goal_sequence[self._current_target_index]
        pos = np.array(goal["position"])
        rpy = np.array(goal["orientation"])
        quat = euler_angles_to_quat(rpy)

        target = self._world.scene.get_object("shared_target")
        target.set_world_pose(position=pos, orientation=quat)

        self._fsm_timer = self._world.current_time
        print(f"[FSM] Teleported to goal {self._current_target_index} → pos: {pos}, rpy: {rpy}")


    def _visualize_every_goals(scene, filename="goals.json"):
        base_dir = os.path.dirname(__file__)
        abs_path = os.path.join(base_dir, filename)

        with open(abs_path, "r") as f:
            goal_points = json.load(f)

        for i, entry in enumerate(goal_points):
            pos = np.array(entry["position"])
            rpy = np.array(entry["orientation"])  # roll, pitch, yaw
            quat = euler_angles_to_quat(rpy)

            folder_path = "/World/GoalCubes"
            if not is_prim_path_valid(folder_path):
                create_prim(prim_path=folder_path, prim_type="Xform")
                
            cube = VisualCuboid(
                name=f"goal_{i}",
                prim_path=f"{folder_path}/GoalCube_{i}",
                position=pos,
                orientation=quat,
                color=np.array([0, 0, 0]),
                size=0.5,
                scale=np.array([0.03, 0.03, 0.03]) / get_stage_units(),
            )
            scene.add(cube)

class GoalLoader:
    @staticmethod
    def _load_goals_from_file(filename="goals.json"):
        base_dir = os.path.dirname(__file__)
        abs_path = os.path.join(base_dir, filename)

        with open(abs_path, "r") as f:
            goal_points = json.load(f)

        return goal_points