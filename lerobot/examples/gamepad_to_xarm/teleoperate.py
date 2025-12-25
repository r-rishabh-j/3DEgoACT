# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specif

import time

from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.teleoperators.gamepad.gamepad_utils import XboxXarmController
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
import gymnasium as gym
import gym_xarm
import mujoco
import mujoco.viewer
import numpy as np

FPS = 50

# Initialize the robot and teleoperator
# Create environment
env = gym.make("gym_xarm/XarmLift-v0", obs_type="state", observation_width=128, observation_height=128)
observation, info = env.reset()

# Access MuJoCo model and data
model = env.unwrapped.model
data = env.unwrapped.data
teleoperator = XboxXarmController()

teleoperator.start()

delta_scale = 0.5
def create_robot_action(state):
    action = np.zeros(4, dtype=np.float32)
    action[0]= -state['axis_3'] * delta_scale  # Invert Y axis
    action[1]= (state['axis_2']) * delta_scale
    action[2]= (-state['axis_1']) * delta_scale  # Right - Left trigger
    if state['axis_4']>0.2:  # A button
        action[3] = 1.0
    if state['axis_5']>0.2:  # B button       
        action[3] = -1.0
    return action

# Init rerun viewer
init_rerun(session_name="lerobot_teletop")

print("Starting teleop loop. Move your phone to teleoperate the robot...")
while True:
    t0 = time.perf_counter()

    # Get robot observation
    robot_obs = robot.get_observation()

    # Get teleop action
    phone_obs = teleop_device.get_action()

    # Phone -> EE pose -> Joints transition
    joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))

    # Send action to robot
    _ = robot.send_action(joint_action)

    # Visualize
    log_rerun_data(observation=phone_obs, action=joint_action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
