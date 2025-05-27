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

#+++++ Custom 모듈
from isaacsim.examples.interactive.lib_module.data_io import DataIO



class GoalRelated(RoaiBaseSample):
    def _move_to_next_target(self):
        self._current_target_index += 1

        if self._current_target_index >= self._num_of_goals:
            self._current_target_index = 0
            self._current_robot_index += 1
            
            if self._current_robot_index >= self._num_of_tasks:
                self._current_robot_index -= 1
                self._fsm_finished = True
                return
            
        goal = self._goal_sequence[self._current_target_index]
        pos = np.array(goal["position"])
        rpy = np.array(goal["orientation"])
        quat = euler_angles_to_quat(rpy)        # wxyz로 만듦

        target = self._world.scene.get_object("SharedTarget")
        target.set_world_pose(position=pos, orientation=quat)

        #carb.log_info(f"[FSM] Move to goal #{self._current_target_index} → pos: {pos}, rpy: {rpy}")
        print("--------------------------------------------------------")
        print(f"[FSM] Move to goal #{self._current_target_index} → pos: {pos}, rpy: {rpy}")

    def _goal_Z_rotation(self):
        # 회전 완료 조건
        if self._current_Z_rotation_angle >= 360:
            self._goal_rotation_done = True
            print("Rotation complete. Moving to next target.")
            return
        
        current_time = self._world.current_time
        angle = self._current_Z_rotation_angle

        # 로봇 제어를 위한 기본 정보 가져오기
        robot = self._robots[self._current_robot_index]
        controller = self._controllers[self._current_robot_index]
        articulation_controller = self._articulation_controllers[self._current_robot_index]
        observations = self._world.get_observations()
        target_position = observations["SharedTarget"]["position"]
        target_orientation = observations["SharedTarget"]["orientation"]
        base_position, base_orientation = robot.get_world_pose()

        # local_ori 보정 (이론상 그대로 가야 맞는데, 지금 franka hand 좌표계로 인해 y축 기준 180도 뒤집어야 함)
        y_axis_rotation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        rotated_local_ori = quaternion_multiply(y_axis_rotation, target_orientation)

        # 회전 각도 계산 및 상대좌표 설정
        yaw_rotation = euler_angles_to_quat(np.array([0,0,np.radians(angle)]))
        rotated_orientation = quaternion_multiply(rotated_local_ori,yaw_rotation)

        local_pos, local_ori = GoalRelated._transform_goal_to_local_frame(
            target_position, rotated_orientation, base_position, base_orientation
        )

        # 컨트롤러 동작
        actions = controller.forward(
            target_index=self._current_target_index,
            target_end_effector_position=local_pos,
            target_end_effector_orientation=local_ori,
        )
        kps, kds = self._tasks[self._current_robot_index].get_custom_gains()
        articulation_controller.set_gains(kps, kds)
        articulation_controller.apply_action(actions)

        # 성공 여부 체크
        if controller._success_flag:
            self._any360_reached_flag = True
            if (current_time - self._fsm_timer > self._goal_move_timeout):
                DataIO._on_logging_event(self, angle)
                print(f"Z rot angle: {angle} - Success")
                self._current_Z_rotation_angle = angle + self._target_360_resolution

                self._controllers[self._current_robot_index].reset()
                self._reached_flag = False
                self._fsm_timer = current_time
        else:
            print(f"Z rot angle: {angle} - Fail")
            self._current_Z_rotation_angle = angle + self._target_360_resolution

            # self._controllers[self._current_robot_index].reset()
            self._fsm_timer = current_time

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

            folder_path = "/World/GoalList"
            if not is_prim_path_valid(folder_path):
                create_prim(prim_path=folder_path, prim_type="Xform")
                
            cube = VisualCuboid(
                name=f"Goal_{i}",
                prim_path=f"{folder_path}/Goal_{i}",
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

@staticmethod
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])