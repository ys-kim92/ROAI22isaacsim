from isaacsim.examples.interactive.base_sample import RoaiBaseSample


class DataIO(RoaiBaseSample):
    def _on_logging_event(self, log_freq):
        world = self.get_world()
        data_logger = world.get_data_logger()

        if not world.get_data_logger().is_started():
            def frame_logging_func(tasks, scene):               
                data = {}

                for i, params in enumerate(self._task_params):
                    target_name = params["target_name"]["value"]
                    target = scene.get_object(target_name)
                    target_pos, target_ori = target.get_world_pose()
                    robot_name = params["robot_name"]["value"]
                    robot = scene.get_object(robot_name)
                    
                    data[f"robot{i}_goal_position"] = target_pos.tolist()
                    data[f"robot{i}_goal_orientation"] = target_ori.tolist()
                    data[f"robot{i}_current_joint_positions"] = robot.get_joint_positions().tolist()
                    data[f"robot{i}_applied_joint_positions"] = robot.get_applied_action().joint_positions.tolist()

                return data

            data_logger.add_data_frame_logging_func(frame_logging_func)
            data_logger.start()


    def _on_save_data_event(self, log_path):
        world = self.get_world()
        data_logger = world.get_data_logger()
        data_logger.save(log_path=log_path)
        data_logger.reset()
        return
