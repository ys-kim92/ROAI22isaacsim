# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import os
import carb.settings

import omni.ext
import omni.ui as ui
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import RoaiBaseSampleUITemplate
from isaacsim.examples.interactive.user_examples import GoalValidation
from isaacsim.gui.components.ui_utils import btn_builder, get_style, setup_ui_headers, state_btn_builder, str_builder


class GoalValidationExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Goal Validation"
        self.category = "ROAI"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Goal validation framework",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_adding_multiple_robots.html",
            "overview": "This Example shows how to run multiple tasks in the same scene.\n\nPress 'LOAD' to load the scene, \npress 'START validation' to start goal validation.\n\nPress the 'Open in IDE' button to view the source code.",
            "sample": GoalValidation(),
        }

        ui_handle = GoalValidationUI(**ui_kwargs)

        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)
        return


class GoalValidationUI(RoaiBaseSampleUITemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extra_frames(self):
        extra_stacks = self.get_extra_frames_handle()
        self.task_ui_elements = {}

        with extra_stacks:
            with ui.CollapsableFrame(
                title="Data Log Option",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                # style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                self.build_data_logging_ui()

            with ui.CollapsableFrame(
                title="Task Control",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                # style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                self.build_task_controls_ui()

    def post_reset_button_event(self):
        self.task_ui_elements["Start"].enabled = True
        return

    def post_load_button_event(self):
        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", "PathTracing")

        self.task_ui_elements["Start"].enabled = True
        return

    def post_clear_button_event(self):
        self.task_ui_elements["Start"].enabled = False
        return
    
    #++++ 실제 버튼 UI
    
    def build_data_logging_ui(self):
        with ui.VStack(spacing=5):
            dict = {
                "label": "Output Directory",
                "type": "stringfield",
                "default_val": os.path.join(os.getcwd()),
                "tooltip": "Output Directory",
                "on_clicked_fn": None,
                "use_folder_picker": False,
                "read_only": False,
            }
            self.task_ui_elements["Output Directory"] = str_builder(**dict)
        return

    def build_task_controls_ui(self):
        with ui.VStack(spacing=5):

            dict = {
                "label": "Goal validation",
                "type": "button",
                "a_text": "START",
                "b_text": "STOP",
                "tooltip": "Start goal validation",
                "on_clicked_fn": self._on_task_button_event,
            }

            self.task_ui_elements["Start"] = state_btn_builder(**dict)
            self.task_ui_elements["Start"].enabled = False

    #+++++ 실제 버튼 기능

    def _on_task_button_event(self, val):
        asyncio.ensure_future(self.sample._on_task_event_async(val,self.task_ui_elements["Output Directory"].get_value_as_string()))
        #self.task_ui_elements["Start"].enabled = False
        return