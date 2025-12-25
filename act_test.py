import torch
from lerobot.policies.act.modeling_act import ACTPolicy, ACTConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import time
import gym_xarm
import numpy
import random
import cv2
import os
from pathlib import Path
from mujoco_point_cloud import *
from lerobot.utils.robot_utils import busy_wait
from point_cloud_wrapper import MujocoPointcloudXArm
from pointcloud import visualize_pointcloud, visualize_pointclouds_video

torch.manual_seed(100)
numpy.random.seed(100)
random.seed(100)

os.environ['MUJOCO_GL'] = 'glfw'    
device = torch.device("mps")
dataset_id = "rishabhrj11/gym-xarm-grab-6"
dataset_metadata = LeRobotDatasetMetadata(dataset_id)

model_id = ""
act_config = ACTConfig.from_pretrained(model_id)
act_config.temporal_ensemble_coeff = None
# act_config.n_action_steps = 10
act_config.temporal_ensemble_coeff = 0.01
act_config.n_action_steps = 1
act_config.use_vae = False
act_model = ACTPolicy.from_pretrained(pretrained_name_or_path=model_id, config=act_config).to(device)
preprocess, postprocess = make_pre_post_processors(policy_cfg=act_model.config,
                                                   pretrained_path=model_id,
                                                   dataset_stats=dataset_metadata.stats, 
                                                    preprocessor_overrides={"device_processor": {"device": str(device)}},
                                                   )

MAX_EPISODES = 25
MAX_STEPS_PER_EPISODE = 1000
camera_width = 224
camera_height = 224
# Create environment
env = gym.make("gym_xarm/XarmLift-v0", obs_type="state",
               observation_width=camera_width,
               observation_height=camera_height)
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
camera_names = ["camera_top", "camera_ef"]
depth_camera_names = []
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

# Get object and target IDs for randomization
object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'object_joint0')
target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'target0')
object_qpos_adr = model.jnt_qposadr[object_joint_id]

def randomize_task():
    """Randomize object and target positions"""
    # Randomize object
    obj_x = np.random.uniform(1.4, 1.7)
    obj_y = np.random.uniform(0.1, 0.5)
    data.qpos[object_qpos_adr:object_qpos_adr+2] = [obj_x, obj_y]
    
    # Randomize target
    tgt_x = np.random.uniform(1.4, 1.7)
    tgt_y = np.random.uniform(0.1, 0.5)
    model.site_pos[target_site_id] = [tgt_x, tgt_y, 0.545]
    
    mujoco.mj_forward(model, data)
    print(f"  Object: [{obj_x:.2f}, {obj_y:.2f}], Target: [{tgt_x:.2f}, {tgt_y:.2f}]")

def save_images_to_video(images, video_path, fps=50, codec='mp4v'):
    """
    Save numpy RGB images to a video file.
    
    Args:
        images: List or array of numpy RGB images. Each image should be shape (H, W, 3) 
                with values in range [0, 255] (uint8) or [0, 1] (float32/float64).
                If float, values will be converted to [0, 255].
        video_path: Path to save the video file (e.g., 'output.mp4')
        fps: Frames per second for the video (default: 30)
        codec: Video codec fourcc code (default: 'mp4v'). Other options: 'XVID', 'H264', etc.
    
    Returns:
        True if successful, False otherwise
    """
    if len(images) == 0:
        print("Warning: No images provided to save")
        return False
    
    # Convert to numpy array if list
    images = np.array(images)
    
    # Get first image to determine dimensions
    first_img = images[0]
    
    # Handle different input formats
    if first_img.dtype == np.float32 or first_img.dtype == np.float64:
        # Assume values are in [0, 1] range, convert to [0, 255]
        if first_img.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)
    else:
        images = images.astype(np.uint8)
    
    # Ensure images are in correct shape (H, W, 3)
    if len(first_img.shape) == 2:
        # Grayscale, convert to RGB
        images = np.stack([images, images, images], axis=-1)
    elif len(first_img.shape) == 3 and first_img.shape[-1] == 1:
        # Single channel, convert to RGB
        images = np.repeat(images, 3, axis=-1)
    
    # Get video dimensions
    height, width = images[0].shape[:2]
    
    # OpenCV uses BGR format, so convert RGB to BGR
    images_bgr = images[:, :, :, [2, 1, 0]]  # RGB to BGR
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {video_path}")
        return False
    
    # Write all frames
    for frame in images_bgr:
        out.write(frame)
    
    out.release()
    print(f"Saved video with {len(images)} frames to: {video_path}")
    return True

def save_multiple_camera_videos(camera_frames_dict, output_dir, fps=50, codec='mp4v'):
    """
    Save multiple camera streams to separate video files.
    
    Args:
        camera_frames_dict: Dictionary mapping camera names to lists of numpy RGB images
                           e.g., {'camera0': [img1, img2, ...], 'camera1': [img1, img2, ...]}
        output_dir: Directory to save video files
        fps: Frames per second for the videos (default: 30)
        codec: Video codec fourcc code (default: 'mp4v')
    
    Returns:
        Dictionary mapping camera names to saved video paths
    
    Example usage:
        # Collect frames during simulation
        camera_frames = {'camera0': [], 'camera1': []}
        for step in range(100):
            img0 = render_camera0()  # numpy array (H, W, 3) RGB, uint8 or float
            img1 = render_camera1()
            camera_frames['camera0'].append(img0)
            camera_frames['camera1'].append(img1)
        
        # Save all videos
        save_multiple_camera_videos(camera_frames, 'output_videos', fps=30)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    for cam_name, frames in camera_frames_dict.items():
        video_path = output_dir / f"{cam_name}.mp4"
        if save_images_to_video(frames, video_path, fps=fps, codec=codec):
            saved_paths[cam_name] = video_path
    
    return saved_paths

from pynput import keyboard as kb

# Key state tracking
key_states = {}

def on_press(key):
    try:
        k = key.char.upper() if hasattr(key, 'char') else str(key).upper()
        key_states[k] = True
    except AttributeError:
        pass

def on_release(key):
    try:
        k = key.char.upper() if hasattr(key, 'char') else str(key).upper()
        key_states[k] = False
    except AttributeError:
        pass

listener = kb.Listener(on_press=on_press, on_release=on_release)
listener.start()

act_model.reset()
act_model.eval()
randomize_task()
episode_count=0
step_count=0

save_snapshots= 0

with torch.inference_mode():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t=time.time()
        while time.time()-t<2:
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample()*0)
            viewer.sync()
        while viewer.is_running():
            start_loop_t=time.perf_counter()
            if episode_count>MAX_EPISODES:
                break
            step_count+=1
            if step_count>MAX_STEPS_PER_EPISODE or key_states.get('X', False):
                observation, info = env.reset()
                act_model.reset()
                step_count=0
                randomize_task()
                while time.time()-t<5:
                    observation, reward, terminated, truncated, info = env.step(env.action_space.sample()*0)
                    viewer.sync()
                episode_count+=1
                if save_snapshots:
                    # visualize_pointclouds_video(camera_frames, path=f'./evaluation_videos/episode_{episode_count}_cloud.html')
                    save_images_to_video(camera_frames, f'./evaluation_simple/episode_{episode_count}.mp4')
                camera_frames.clear()
            obs_processed = {
                'observation.state': torch.tensor(observation.copy(),dtype=torch.float32).to(device),
            }

            for camera in cameras:
                # Capture images from all cameras at current state
                camera.renderer.update_scene(data, camera=camera.name)
                pixels = camera.renderer.render()
                if camera.name == 'camera_top':
                    camera_frames.append(pixels.copy())
                pixels = pixels.astype(np.float32).transpose(2,0,1) / 255.0
                obs_processed[f'observation.image.{camera.name}'] = torch.from_numpy(pixels).unsqueeze(0).contiguous().to(device)
                if camera.depth:
                    camera.depth_renderer.update_scene(data, camera=camera.name)
                    depth_pixels = camera.depth_renderer.render()
                    camera.color_renderer.update_scene(data, camera=camera.name)
                    color_pixels = camera.color_renderer.render()
                    point_cloud, depth=camera.point_cloud_generator.get_point_cloud(depth=depth_pixels, color_img=color_pixels)
                    obs_processed['observation.environment_state'] = torch.tensor(point_cloud.copy(),dtype=torch.float32).unsqueeze(0).contiguous().to(device)

            frame=preprocess(obs_processed)
            action = act_model.select_action(frame)
            action=postprocess(action).cpu().numpy()
            action=np.clip(action, env.action_space.low, env.action_space.high).squeeze(0)
            next_observation, reward, terminated, truncated, info = env.step(action)
            observation = next_observation
            viewer.sync()
            end_loop_t=time.perf_counter()
            t=end_loop_t-start_loop_t
            # get fps
            fps = 1.0 / t if t > 0 else 0
            print(f"FPS: {fps:.2f}")

        print("Episode finished! Starting new episode...")