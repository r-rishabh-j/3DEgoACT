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
from pynput import keyboard as kb
import numpy
import random
import cv2
import os
import json
import argparse
from pathlib import Path
from mujoco_point_cloud import *
from lerobot.utils.robot_utils import busy_wait
from point_cloud_wrapper import MujocoPointcloudXArm
from pointcloud import visualize_pointcloud, visualize_pointclouds_video
from utils import save_images_to_video, save_multiple_camera_videos

# Parse CLI arguments for config file
os.environ['MUJOCO_GL'] = 'glfw'
parser = argparse.ArgumentParser(description="3D ACT Test Script")
parser.add_argument("--config", type=str, required=False,
                    help="Path to JSON config file containing dataset_id, root, and model_id")
args = parser.parse_args()

# Default values
DEFAULT_DATASET_ID = "rishabhrj11/gym-xarm-pointcloud-2"
DEFAULT_ROOT = "../gym-xarm-pointcloud-2"
DEFAULT_MODEL_ID = "../outputs/train3dnoef/"

default_config = {
    'experiment_name': 'camtop',
    "dataset_id": DEFAULT_DATASET_ID,
    "root": DEFAULT_ROOT,
    "model_id": DEFAULT_MODEL_ID,
    'seed': 100,
    'model_id': DEFAULT_MODEL_ID,
    'model_config': {
        'n_action_steps': 1,
        'use_vae': False,
        'temporal_ensemble_coeff': -0.005,
        'n_action_steps': 1,
    },
    'MAX_EPISODES': 25,
    'MAX_STEPS_PER_EPISODE': 1000,
    'save_snapshots': True,
    'save_dir': './evals/act3d/',
    'camera_names': ['camera_top', 'camera_ef'],
    'depth_camera_names': ['camera_top'],
}

config = default_config
if args.config:
    with open(args.config, "r") as f:
        config = json.load(f)
    config = {**default_config, **config}

save_dir = os.path.join(config['save_dir'], f'{config["experiment_name"]}_{config["seed"]}')
os.makedirs(save_dir, exist_ok=True)

torch.manual_seed(config['seed'])
numpy.random.seed(config['seed'])
random.seed(config['seed'])

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dataset_id = config['dataset_id']
dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=config['root'])

act_config = ACTConfig.from_pretrained(config['model_id'])
act_config.n_action_steps = config['model_config']['n_action_steps']
act_config.temporal_ensemble_coeff = config['model_config']['temporal_ensemble_coeff']
act_config.use_vae = config['model_config']['use_vae']

act_model = ACTPolicy.from_pretrained(
    pretrained_name_or_path=config['model_id'], config=act_config).to(device)
preprocess, postprocess = make_pre_post_processors(policy_cfg=act_model.config,
                                                   pretrained_path=config['model_id'],
                                                   dataset_stats=dataset_metadata.stats,
                                                   preprocessor_overrides={"device_processor": {"device": str(device)}})

MAX_EPISODES = config['MAX_EPISODES']
MAX_STEPS_PER_EPISODE = config['MAX_STEPS_PER_EPISODE']
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
        self.cam_id = cam_id
        self.depth = depth
        self.renderer = mujoco.Renderer(
            model, height=camera_height, width=camera_width)
        if depth:
            self.color_renderer = mujoco.Renderer(model, height=84, width=84)
            self.depth_renderer = mujoco.Renderer(model, height=84, width=84)
            self.depth_renderer.enable_depth_rendering()
            self.point_cloud_generator = MujocoPointcloudXArm(
                model, data, [cam_id], img_size=84, env_name='xarm', use_point_crop=True)


camera_names = config['camera_names']
depth_camera_names = config['depth_camera_names']
cameras = []
point_cloud_generator = None

for cam_name in camera_names:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        print(f"Warning: Camera '{cam_name}' not found in model. Skipping.")
        continue
    camera = Camera(model, cam_name, cam_id,
                    depth=cam_name in depth_camera_names)
    cameras.append(camera)

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
    print(
        f"  Object: [{obj_x:.2f}, {obj_y:.2f}], Target: [{tgt_x:.2f}, {tgt_y:.2f}]")

def get_box_pose():
    return data.qpos[object_qpos_adr:object_qpos_adr+3]
def get_target_pose():
    return model.site_pos[target_site_id]
def is_valid_start_state():
    return np.linalg.norm(get_box_pose() - get_target_pose()) > 0.11
class State:
    episode_number = 1
    def __init__(self):
        self.is_picked_up = False
        self.is_placed = False
        self.success = False
        self.step_count = 0
        self.box_pose = None
        self.target_pose = None

    def reset(self):
        self.is_picked_up = False
        self.is_placed = False
        self.success = False
        State.episode_number += 1
        self.step_count = 0
        self.box_pose = None
        self.target_pose = None
    
    def update(self):
        # logic to update the state
        self.box_pose = get_box_pose()
        self.target_pose = get_target_pose()
        if not self.is_picked_up:
            if self.box_pose[2] > 0.62:
                self.is_picked_up = True
                print('box picked up')
        if self.is_picked_up:
            if np.linalg.norm(self.box_pose - self.target_pose) < 0.05 and self.box_pose[2] < 0.569:
                self.is_placed = True
                print('box placed')
        if self.is_placed:
            if np.linalg.norm(self.box_pose - self.target_pose) < 0.05 and self.box_pose[2] < 0.569 and env.unwrapped.eef[2] > -0.035:
                self.success = True
                print('box success')
    
    def to_dict(self):
        return {
            'is_picked_up': self.is_picked_up,
            'is_placed': self.is_placed,
            'success': self.success,
            'box_pose': self.box_pose.tolist(),
            'target_pose': self.target_pose.tolist(),
        }

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

save_snapshots = config['save_snapshots']
save_point_clouds = config.get('save_point_clouds', False)
camera_frames = []
point_cloud_frames = []
total_fps = 0

episode_results = []
total_time = 0

state = State()

with torch.inference_mode():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = time.time()
        while time.time()-t < 2:
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()*0)
            viewer.sync()
        while not is_valid_start_state():
            observation, info = env.reset(seed=config['seed'])
            randomize_task()
        while viewer.is_running():
            start_loop_t = time.perf_counter()
            if state.episode_number > MAX_EPISODES:
                break
            state.step_count += 1
            if state.success or state.step_count > MAX_STEPS_PER_EPISODE or key_states.get('X', False):
                episode_results.append(state.to_dict())
                while time.time()-t < 5:
                    observation, reward, terminated, truncated, info = env.step(
                        env.action_space.sample()*0)
                    viewer.sync()
                if save_snapshots:
                    # visualize_pointclouds_video(camera_frames, path=f'./evaluation_videos/episode_{episode_count}_cloud.html')
                    save_images_to_video(
                        camera_frames, f'{save_dir}/episode_{state.episode_number}.mp4')
                    camera_frames.clear()
                if save_point_clouds:
                    visualize_pointclouds_video(
                        point_cloud_frames, path=f'{save_dir}/episode_{state.episode_number}_cloud.html', fps=150)
                    point_cloud_frames.clear()
                total_time = 0
                state.reset()
                act_model.reset()
                observation, info = env.reset(seed=config['seed'])
                randomize_task()
                while not is_valid_start_state():
                    print('invalid start state, resetting') 
                    observation, info = env.reset(seed=config['seed'])
                    randomize_task()
            obs_processed = {
                'observation.state': torch.tensor(observation.copy(), dtype=torch.float32).to(device),
            }

            for camera in cameras:
                # Capture images from all cameras at current state
                camera.renderer.update_scene(data, camera=camera.name)
                pixels = camera.renderer.render()
                if camera.name == 'camera_top' and save_snapshots:
                    camera_frames.append(pixels.copy())
                pixels = pixels.astype(np.float32).transpose(2, 0, 1) / 255.0
                obs_processed[f'observation.image.{camera.name}'] = torch.from_numpy(
                    pixels).unsqueeze(0).contiguous().to(device)
                if camera.depth:
                    camera.depth_renderer.update_scene(
                        data, camera=camera.name)
                    depth_pixels = camera.depth_renderer.render()
                    camera.color_renderer.update_scene(
                        data, camera=camera.name)
                    color_pixels = camera.color_renderer.render()
                    point_cloud, depth = camera.point_cloud_generator.get_point_cloud(
                        depth=depth_pixels, color_img=color_pixels)
                    obs_processed['observation.environment_state'] = torch.tensor(
                        point_cloud.copy(), dtype=torch.float32).unsqueeze(0).contiguous().to(device)
                    if save_point_clouds:
                        point_cloud_frames.append(point_cloud.copy())

            frame = preprocess(obs_processed)
            action = act_model.select_action(frame)
            action = postprocess(action).cpu().numpy()
            action = np.clip(action, env.action_space.low,
                             env.action_space.high).squeeze(0)
            end_loop_t = time.perf_counter()
            next_observation, reward, terminated, truncated, info = env.step(
                action)
            observation = next_observation
            viewer.sync()
            state.update()
            t = end_loop_t-start_loop_t
            total_time += t
        fps = state.step_count / total_time
        total_fps += fps
        print(f"Episode {state.episode_number} finished! Starting new episode...")


print(f"Average FPS: {np.mean(total_fps):.2f}")

with open(os.path.join(save_dir, 'results.json'), 'w') as f:
    json.dump({'average_fps': np.mean(total_fps)}, f)
with open(os.path.join(save_dir, 'config.json'), 'w') as f:
    json.dump(config, f)
with open(os.path.join(save_dir, 'episode_results.json'), 'w') as f:
    json.dump(episode_results, f)