from isaacsim.examples.interactive.base_sample import RoaiBaseSample
from isaacsim.core.utils.rotations import euler_angles_to_quat
import json
import os
import numpy as np
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.prims import create_prim
from scipy.spatial.transform import Rotation as R
import carb




class GoalRelated(RoaiBaseSample):
    def _move_to_next_target(self):
        self._current_target_index += 1

        if self._current_target_index >= self._num_of_goals:
            self._fsm_finished = True
            return

        goal = self._goal_sequence[self._current_target_index]
        pos = np.array(goal["position"])
        rpy = np.array(goal["orientation"])
        quat = euler_angles_to_quat(rpy)        # wxyz로 만듦

        target = self._world.scene.get_object("shared_target")
        target.set_world_pose(position=pos, orientation=quat)

        self._fsm_timer = self._world.current_time
        #carb.log_info(f"[FSM] Move to goal #{self._current_target_index} → pos: {pos}, rpy: {rpy}")
        print("--------------------------------------------------------")
        print(f"[FSM] Move to goal #{self._current_target_index} → pos: {pos}, rpy: {rpy}")

    @staticmethod
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

        return goal_points


    @staticmethod
    def _transform_goal_to_local_frame(target_position, target_orientation, base_position, base_orientation):
        base_ori_xyzw = wxyz_to_xyzw(base_orientation)
        target_ori_xyzw = wxyz_to_xyzw(target_orientation)

        # 위치 변환
        delta = target_position - base_position
        base_rot = R.from_quat(base_ori_xyzw)
        local_position = base_rot.inv().apply(delta)

        # 회전 변환
        target_rot = R.from_quat(target_ori_xyzw)
        local_rot = base_rot.inv() * target_rot
        local_orientation = local_rot.as_quat()
        
        local_orientation = xyzw_to_wxyz(local_orientation)

        return local_position, local_orientation
       
    

@staticmethod
def wxyz_to_xyzw(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]])

@staticmethod
def xyzw_to_wxyz(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]])