from isaacsim.examples.interactive.base_sample import RoaiBaseSample
import os
import json
import numpy as np
from collections import defaultdict
from isaacsim.core.utils.rotations import euler_angles_to_quat
import time


class DataIO(RoaiBaseSample):
    def _on_logging_event(self):
        scene = self._world.scene
        robot_key = f"robot #{self._current_robot_index}"

        if robot_key not in self._success_goal:
            self._success_goal[robot_key] = []
            
        # 로깅할 데이터
        target_name = self._shared_target_name
        target = scene.get_object(target_name)
        target_pos, target_ori = target.get_world_pose()

        goal_data = {
            "goal_number": f"Goal_{self._current_target_index}",
            "goal_position": target_pos.tolist(),
            "goal_orientation": target_ori.tolist()
        }
            
        self._success_goal[robot_key].append(goal_data)


    def _on_save_data_event(self, log_path):
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, "success_goal.json"), "w") as f:
            json.dump(self._success_goal, f, indent=2)
        
        DataIO.report_goal_validation(self)
        # 저장 후 버퍼 비움
        self._success_goal.clear()

        return

    def report_goal_validation(self):
        if not hasattr(self, "_success_goal") or not isinstance(self._success_goal, dict):
            print("[ERROR] Wrong format of self._success_goal")
            return

        unique_goal_numbers = set()
        robot_goal_counts = {}
        robot_goal_lists = defaultdict(list)


        # 결과 분석
        working_duration_sim = self._world.current_time - self._ini_time_sim
        working_duration_real = time.time() - self._ini_time_real
        m_sim, s_sim = divmod(working_duration_sim, 60)
        m_real, s_real = divmod(working_duration_real, 60)

        for robot_key, goal_list in self._success_goal.items():
            robot_goal_counts[robot_key] = len(goal_list)

            for goal in goal_list:
                goal_number = goal["goal_number"]

                unique_goal_numbers.add(goal_number)
                robot_goal_lists[robot_key].append(goal_number)

                # scene에 반영
                target = self._world.scene.get_object(goal_number)      # prim_path가 아닌 name 찾음
                self._target_visual_material = target.get_applied_visual_material()
                self._target_visual_material.set_color(color=np.array([0, 1.0, 0]))   # green

        target = self._world.scene.get_object("SharedTarget")
        target.set_world_pose(position=np.array([0, 0, 0.7]), orientation=euler_angles_to_quat(np.array([0, 0, 0])))


        # 결과 print
        print("\n+++++++ Result report +++++++")
        print(f"* Number of success goals: {len(unique_goal_numbers)} / {self._num_of_goals}")
        print(f"* Total calculation time - real (MM:SS): {int(m_real):02d}:{int(s_real):02d}")
        print(f"                         - sim  (MM:SS): {int(m_sim):02d}:{int(s_sim):02d}")

        for robot_key, goal_list in self._success_goal.items():
            print(f"* Success goals for each {robot_key}: {robot_goal_counts[robot_key]} goals")
            print(f"    {robot_goal_lists[robot_key]}")