import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import h5py
import time
import gym_xarm
import os

device = torch.device("mps")  # or "cuda" or "cpu"
model_id = "../../../../../outputs/train/checkpoints/last/pretrained_model"
act_model = ACTPolicy.from_pretrained(model_id).to(device)

dataset_id = "rishabhrj11/gym-xarm-grab-ds"
# This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
dataset_metadata = LeRobotDatasetMetadata(dataset_id, root="/Users/rishabh/development/robotics/gym-xarm-grab-ds")
preprocess, postprocess = make_act_pre_post_processors(act_model.config, dataset_stats=dataset_metadata.stats)

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 1000
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

# Define which cameras to record from
camera_names = ["camera0", "camera_top", "camera_ef"]

renderers = {}

for cam_name in camera_names:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        print(f"Warning: Camera '{cam_name}' not found in model. Skipping.")
        continue
    renderer = mujoco.Renderer(model, height=camera_height, width=camera_width)
    renderers[cam_name] = {'renderer': renderer, 'id': cam_id}

print(f"\nInitialized {len(renderers)} camera renderers")

# Dataset storage
dataset = []
episode = []
delta_scale = 0.5

# Get object and target IDs for randomization
object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'object_joint0')
target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'target0')
object_qpos_adr = model.jnt_qposadr[object_joint_id]

act_model.eval()
with torch.no_grad():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            current_state = torch.tensor(observation.copy(),dtype=torch.float32).unsqueeze(0).to(device)
            obs_processed = {
                'observation.state': current_state,
            }

            # Capture images from all cameras at current state
            for cam_name, cam_info in renderers.items():
                cam_info['renderer'].update_scene(data, camera=cam_name)
                pixels = cam_info['renderer'].render()
                obs_processed[f'observation.image.{cam_name}'] = torch.from_numpy(pixels).reshape(3,224,224).unsqueeze(0).float().to(device)
            obs_processed=preprocess(obs_processed)
            frame = {**obs_processed, "task": 'push'}
            action = act_model.select_action(frame)
            action = postprocess(action)
            next_observation, reward, terminated, truncated, info = env.step(-delta_scale*action.cpu().numpy().squeeze(0))
            observation = next_observation

            viewer.sync()
            time.sleep(0.01)

        print("Episode finished! Starting new episode...")