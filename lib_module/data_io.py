from isaacsim.examples.interactive.base_sample import RoaiBaseSample
import os
import json


class DataIO(RoaiBaseSample):
    def _on_logging_event(self):
        scene = self._world.scene
        robot_key = f"robot #{self._current_robot_index}"
        
        if self._current_target_index == 0:
            self._success_goal[robot_key] = []

        if robot_key not in self._success_goal:
            self._success_goal[robot_key] = []
            
        # 로깅할 데이터
        target_name = self._shared_target_name
        target = scene.get_object(target_name)
        target_pos, target_ori = target.get_world_pose()

        goal_data = {
            "goal_number": f"goal #{self._current_target_index}",
            "goal_position": target_pos.tolist(),
            "goal_orientation": target_ori.tolist()
        }
            
        self._success_goal[robot_key].append(goal_data)



    def _on_save_data_event(self, log_path):
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, "success_goal.json"), "w") as f:
            json.dump(self._success_goal, f, indent=2)
        
        # 저장 후 버퍼 비움
        self._success_goal.clear()
        
        return
