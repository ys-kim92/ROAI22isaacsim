from isaacsim.examples.interactive.base_sample import RoaiBaseSample
from isaacsim.core.utils.rotations import euler_angles_to_quat
import json
import os
import numpy as np
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.stage import get_stage_units


class GoalRelated(RoaiBaseSample):
    def _teleport_next_target(self):
        self._current_target_index += 1
        if self._current_target_index >= len(self._target_waypoints):
            self._fsm_finished = True
            return

        new_pos = self._target_waypoints[self._current_target_index]
        target = self._world.scene.get_object("shared_target")
        target.set_world_pose(position=new_pos, orientation=euler_angles_to_quat([0, 0, 0]))
        self._fsm_timer = self._world.current_time
        print(f"[FSM] Moved to waypoint {self._current_target_index}: {new_pos}")

    def load_every_goals(scene, json_path="goals.json"):
        base_dir = os.path.dirname(__file__)
        abs_path = os.path.join(base_dir, json_path)

        with open(abs_path, "r") as f:
            goal_points = json.load(f)

        for i, entry in enumerate(goal_points):
            pos = np.array(entry["position"])
            rpy = np.array(entry["orientation"])  # roll, pitch, yaw
            quat = euler_angles_to_quat(rpy)

            cube = VisualCuboid(
                name=f"goal_{i}",
                prim_path=f"/World/GoalCube_{i}",
                position=pos,
                orientation=quat,
                color=np.array([1, 0, 0]),
                size=0.5,
                scale=np.array([0.03, 0.03, 0.03]) / get_stage_units(),
            )
            scene.add(cube)