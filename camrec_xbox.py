
from lerobot.utils.robot_utils import busy_wait
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import time
import gym_xarm
import os
from point_cloud_wrapper import MujocoPointcloudXArm
from pointcloud import visualize_pointclouds_video

# CRITICAL: Set BEFORE importing pygame (dummy driver prevents Cocoa video conflicts)
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"  # Allows input w/o focus

import pygame  # Now safe
pygame.display.set_mode((1, 1))
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise ValueError("No joystick detected. Connect Xbox controller.")
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Controller: {joystick.get_name()} | Axes: {joystick.get_numaxes()} | Buttons: {joystick.get_numbuttons()}")

# Create renderers for each camera
# camera_width = 84
# camera_height = 84
camera_width = 224
camera_height = 224
# Create environment
env = gym.make("gym_xarm/XarmLift-v0", obs_type="state", observation_width=camera_width, observation_height=camera_height)
observation, info = env.reset()
# Access MuJoCo model and data
model = env.unwrapped.model
data = env.unwrapped.data

# List all cameras defined in the XML
print("Available cameras in model:")
for i in range(model.ncam):
    cam_name = model.camera(i).name
    print(f"  Camera {i}: {cam_name}")

class Camera:
    def __init__(self, model, name, cam_id, depth=False) -> None:
        self.model = model
        self.name = name
        self.cam_id=cam_id
        self.depth = depth
        self.renderer = mujoco.Renderer(model, height=camera_height, width=camera_width)
        if depth:
            self.color_renderer = mujoco.Renderer(model, height=84, width=84)
            self.depth_renderer = mujoco.Renderer(model, height=84, width=84)
            self.depth_renderer.enable_depth_rendering()
            self.point_cloud_generator = MujocoPointcloudXArm(model, data, [cam_id], img_size=84, env_name='xarm', use_point_crop=True)

camera_frames = []
# camera_names = ["camera0", "camera_top", "camera_ef"]
# depth_camera_names = []
depth_camera_names = ["camera_top"]
camera_names = ["camera0", "camera1", "camera_top", "camera_ef"]
renderers = {}
cameras = []
point_cloud_generator = None

for cam_name in camera_names:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        print(f"Warning: Camera '{cam_name}' not found in model. Skipping.")
        continue
    camera = Camera(model, cam_name, cam_id, depth=cam_name in depth_camera_names)
    cameras.append(camera)

print(f"\nInitialized {len(renderers)} camera renderers")

# Dataset storage
dataset = []
episode = []
delta_scale = 0.8

print("\nTeleoperating xArm with camera recording enabled")

# Get object and target IDs for randomization
object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'object_joint0')
target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'target0')
object_qpos_adr = model.jnt_qposadr[object_joint_id]

def randomize_task():
    """Randomize object and target positions"""
    # Randomize object
    obj_x = np.random.uniform(1.3, 1.68)
    obj_y = np.random.uniform(0.06, 0.55)
    data.qpos[object_qpos_adr:object_qpos_adr+3] = [obj_x, obj_y, 0.68]
    
    # Randomize target
    tgt_x = np.random.uniform(1.45, 1.65)
    tgt_y = np.random.uniform(0.06, 0.55)
    model.site_pos[target_site_id] = [tgt_x, tgt_y, 0.545]
    
    mujoco.mj_forward(model, data)
    print(f"  Object: [{obj_x:.2f}, {obj_y:.2f}], Target: [{tgt_x:.2f}, {tgt_y:.2f}]")

def apply_deadzone(value, deadzone=0.15):
    """Clamp small values to 0, scale rest to full range (-1 to 1)."""
    if abs(value) < deadzone:
        return 0.0
    # Scale to preserve full range
    return (value - deadzone * (1 if value > 0 else -1)) / (1 - deadzone)

def get_joystick_state(joystick):
    pygame.event.pump()  # Update joystick state
    
    state = {}
    
    # Get all axes
    num_axes = joystick.get_numaxes()
    for i in range(num_axes):
        state[f'axis_{i}'] = apply_deadzone(joystick.get_axis(i))
    state['axis_4'] = max(0, (state['axis_4'] + 1) / 2) if state['axis_4'] < 0 else state['axis_4']  # Normalize trigger to 0-1
    state['axis_5'] = max(0, (state['axis_5'] + 1) / 2) if state['axis_5'] < 0 else state['axis_5']  # Normalize trigger to 0-1
    # Get all buttons
    num_buttons = joystick.get_numbuttons()
    for i in range(num_buttons):
        state[f'button_{i}'] = joystick.get_button(i)
    
    # Optional: Get hats (D-pads) if needed
    num_hats = joystick.get_numhats()
    for i in range(num_hats):
        state[f'hat_{i}'] = joystick.get_hat(i)
    
    return state

# Launch passive viewer
recording = False
episode_start_time = 0

def create_robot_action(state):
    action = np.zeros(4, dtype=np.float32)
    action[0]= state['axis_3'] * delta_scale  # Invert Y axis
    action[1]= (state['axis_2']) * delta_scale
    action[2]= (-state['axis_1']) * delta_scale  # Right - Left trigger
    if state['axis_4']>0.5:  # A button
        action[3] = 1.0
    else:
        action[3] = -1.0
    # if state['axis_5']>0.2:  # B button       
    #     action[3] = -1.0
    return action

features = {
    'action': {
            "dtype": "float32",
            "shape": [
                4
            ],
            "names": {
                'eef':['x','y','z','w'],
            }
        }
}
features['observation.state'] = {
    "dtype": "float32",
    "shape": observation.shape,
    "names": {"joints": [f'{i}' for i in range(observation.shape[0])],}
}

for camera in cameras:
    camera.renderer.update_scene(data, camera=camera.name)
    pixels = camera.renderer.render()
    pixels = pixels.transpose((2,0,1))  # Channels first
    # pixels
    features[f'observation.image.{camera.name}'] = {
        "dtype": "video",
        "shape": list(pixels.shape),
        "names": ["channel", "height", "width"],
    }
    if camera.depth:
        camera.depth_renderer.update_scene(data, camera=camera.name)
        depth_pixels = camera.depth_renderer.render()
        camera.color_renderer.update_scene(data, camera=camera.name)
        color_pixels = camera.color_renderer.render()
        point_cloud, depth=camera.point_cloud_generator.get_point_cloud(depth=depth_pixels, color_img=color_pixels)
        features[f'observation.environment_state'] = {
            "dtype": "float32",
            "shape": list(point_cloud.shape),
            "names": ["points", "dims"],
        }

resume=0
HF_REPO_ID = "rishabhrj11/gym-xarm-pointcloud-test"
ROOT='/Users/rishabh/development/robotics/gym-xarm-pointcloudx'
FPS=50

if resume:
    dataset = LeRobotDataset(
        HF_REPO_ID,
        root=ROOT,
    )
    dataset.start_image_writer(
        num_threads=4 * len(camera_names),
    )
else:
    # Create empty dataset or load existing saved episodes
    import shutil
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    dataset = LeRobotDataset.create(
        HF_REPO_ID,
        FPS,
        root=ROOT,
        robot_type='gym-xarm',
        features=features,
        use_videos=True,
        image_writer_threads=4 * len(camera_names),
        batch_encoding_size=1,
    )

def get_box_pose():
    return data.qpos[object_qpos_adr:object_qpos_adr+3]
def get_target_pose():
    return model.site_pos[target_site_id]

debug=True
episode_count=0
randomize_task()
with VideoEncodingManager(dataset):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            print('distance', np.linalg.norm(get_box_pose() - get_target_pose()))
            start_loop_t=time.perf_counter()
            state = get_joystick_state(joystick)
            # Exit on X
            if state['button_2']:
                # visualize_pointclouds_video(camera_frames, path=f'./evaluation_videos/test_{episode_count}_cloud.html')
                print('exit')
                viewer.sync()
                break
            if state['button_3']:
                try:
                    dataset.clear_episode_buffer()
                except:
                    pass
                print('exit')
                viewer.sync()
                recording = False
                observation, info = env.reset()
                randomize_task()  # Randomize positions on reset
                continue
            # Check if starting recording
            if state['button_0']:
                if not recording:
                    print('\n=== Now recording ===')
                    recording = True

            action = create_robot_action(state)
            next_observation, reward, terminated, truncated, info = env.step(action)

            if not recording:
                viewer.sync()
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / FPS - dt_s)
                continue

            current_state = np.float32(observation.copy())
            observations = {
                'observation.state': current_state,
            }

            for camera in cameras:
                # Capture images from all cameras at current state
                camera.renderer.update_scene(data, camera=camera.name)
                pixels = camera.renderer.render()
                observations[f'observation.image.{camera.name}'] = pixels.copy()
                if camera.depth:
                    camera.depth_renderer.update_scene(data, camera=camera.name)
                    depth_pixels = camera.depth_renderer.render()
                    camera.color_renderer.update_scene(data, camera=camera.name)
                    color_pixels = camera.color_renderer.render()
                    point_cloud, depth=camera.point_cloud_generator.get_point_cloud(depth=depth_pixels, color_img=color_pixels)
                    observations[f'observation.environment_state'] = point_cloud.copy()
                    # camera_frames.append(point_cloud.copy())


            frame = {**observations, 'action': action, "task": 'Pick and place the box on the red target.'}
            dataset.add_frame(frame)

            observation = next_observation
            
            # Save episode on R key
            if state['button_1']:
                if recording:
                    episode_count+=1
                    print('end episode', episode_count)
                    dataset.save_episode()
                    print(f'\n✓ Episode saved')
                    recording = False
                    observation, info = env.reset()
                    randomize_task()  # Randomize positions on reset
            
            viewer.sync()
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / FPS - dt_s)

# Cleanup
env.close()
dataset.finalize()
# dataset.push_to_hub()