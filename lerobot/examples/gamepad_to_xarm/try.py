# import mujoco
# import numpy as np
# import pygame
# from pygame.locals import *

# # Load your MuJoCo model (replace with your XML file path)
# model = mujoco.MjModel.from_xml_path('path/to/your/model.xml')  # e.g., 'humanoid.xml'
# data = mujoco.MjData(model)

# # Create offscreen renderer
# renderer = mujoco.Renderer(model, height=480, width=640)  # Adjust resolution as needed

# # Initialize Pygame
# pygame.init()
# screen = pygame.display.set_mode((640, 480))  # Match renderer width/height
# pygame.display.set_caption('MuJoCo Teleoperation with Pygame')
# clock = pygame.time.Clock()

# # Main loop
# running = True
# while running:
#     # Handle Pygame events (teleoperation input)
#     for event in pygame.event.get():
#         if event.type == QUIT:
#             running = False
#         elif event.type == KEYDOWN:
#             # Example teleoperation: Apply forces/torques based on keys
#             if event.key == K_UP:
#                 data.ctrl[0] = 1.0  # e.g., forward movement
#             elif event.key == K_DOWN:
#                 data.ctrl[0] = -1.0
#             # Add more controls (e.g., joystick with pygame.joystick)
#         elif event.type == KEYUP:
#             data.ctrl[0] = 0.0  # Reset controls

#     # Step the simulation
#     mujoco.mj_step(model, data)

#     # Render offscreen
#     renderer.update_scene(data, camera="fixed")  # Use a camera name or ID from your model
#     rgb = renderer.render()  # Returns (height, width, 3) uint8 NumPy array

#     # Convert to Pygame surface and display
#     surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))  # Transpose for Pygame (width, height, 3)
#     screen.blit(surface, (0, 0))
#     pygame.display.flip()

#     # Cap FPS
#     clock.tick(60)  # e.g., 60 FPS

# # Cleanup
# pygame.quit()

import os
import mujoco as mj
from mujoco import viewer
import gym_xarm
import gymnasium as gym


# CRITICAL: Set BEFORE importing pygame (dummy driver prevents Cocoa video conflicts)
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"  # Allows input w/o focus

import pygame  # Now safe
pygame.display.set_mode((1, 1))
pygame.joystick.init()

# Load MuJoCo model
# model = mj.MjModel.from_xml_path('path/to/your/model.xml')
# data = mj.MjData(model)
env = gym.make("gym_xarm/XarmLift-v0", obs_type="state", observation_width=128, observation_height=128)
observation, info = env.reset()

# Access MuJoCo model and data
model = env.unwrapped.model
data = env.unwrapped.data

# Setup joystick (index 0 for first/only controller)
if pygame.joystick.get_count() == 0:
    raise ValueError("No joystick detected. Connect Xbox controller.")
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Controller: {joystick.get_name()} | Axes: {joystick.get_numaxes()} | Buttons: {joystick.get_numbuttons()}")

# Launch MuJoCo passive viewer (handles display on main thread)
with viewer.launch_passive(model, data) as v:
    while v.is_running():
        # MINIMAL: Pump events to update joystick state (safe w/ dummy driver)
        pygame.event.pump()

        # Poll current state (updated by pump)
        left_x = joystick.get_axis(0)    # Left stick X (-1 left to 1 right)
        left_y = joystick.get_axis(1)    # Left stick Y (-1 up to 1 down)
        right_x = joystick.get_axis(2)   # Right stick X
        right_y = joystick.get_axis(3)   # Right stick Y (often turn)
        lt = joystick.get_axis(4)        # Left trigger (~ -1 off to 1 full)
        rt = joystick.get_axis(5)        # Right trigger
        a_button = joystick.get_button(0)  # A button (True if pressed)
        # print(left_x, left_y, right_x, right_y, lt, rt, a_button)

        # Map to MuJoCo controls (example for humanoid/cartpole/etc.)
        # data.ctrl[0] = -left_y  # Forward/back
        # data.ctrl[1] = left_x   # Strafe left/right
        # data.ctrl[2] = right_x  # Rotate/turn
        # if a_button:
        #     data.ctrl[3] = 1.0   # Jump/action

        # Step simulation
        mj.mj_step(model, data)

        # Sync viewer
        v.sync()