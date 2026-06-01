#!/usr/bin/env python3
"""Low-risk HIL-SERL FrankaEnv smoke test for the ROS2 HTTP bridge."""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import requests

from franka_env.envs.franka_env import DefaultEnvConfig, FrankaEnv


COMPLIANCE_PARAM = {
    "translational_stiffness": 800,
    "translational_damping": 50,
    "rotational_stiffness": 60,
    "rotational_damping": 5,
    "translational_Ki": 0,
    "translational_clip_x": 0.01,
    "translational_clip_y": 0.01,
    "translational_clip_z": 0.01,
    "translational_clip_neg_x": 0.01,
    "translational_clip_neg_y": 0.01,
    "translational_clip_neg_z": 0.01,
    "rotational_clip_x": 0.05,
    "rotational_clip_y": 0.05,
    "rotational_clip_z": 0.05,
    "rotational_clip_neg_x": 0.05,
    "rotational_clip_neg_y": 0.05,
    "rotational_clip_neg_z": 0.05,
    "rotational_Ki": 0,
}

PRECISION_PARAM = {
    **COMPLIANCE_PARAM,
    "translational_stiffness": 1000,
    "translational_damping": 60,
    "rotational_stiffness": 80,
    "rotational_damping": 6,
}


def get_current_pose(server_url):
    response = requests.post(f"{server_url.rstrip('/')}/getpos_euler", timeout=2)
    response.raise_for_status()
    pose = np.asarray(response.json()["pose"], dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected 6D euler pose, got {pose}")
    return pose


def make_config(args, current_pose):
    xyz_margin = np.asarray(args.xyz_margin, dtype=np.float64)
    rpy_margin = np.asarray(args.rpy_margin, dtype=np.float64)

    class SmokeEnvConfig(DefaultEnvConfig):
        SERVER_URL = args.server_url
        REALSENSE_CAMERAS = {
            args.camera_name: {
                "backend": "shared_memory",
                "shape": (args.image_height, args.image_width, 3),
                "shm_name": args.shm_name,
                "stale_after": args.stale_after,
            }
        }
        IMAGE_CROP = {}
        TARGET_POSE = current_pose.copy()
        RESET_POSE = current_pose.copy()
        ABS_POSE_LIMIT_LOW = current_pose - np.concatenate([xyz_margin, rpy_margin])
        ABS_POSE_LIMIT_HIGH = current_pose + np.concatenate([xyz_margin, rpy_margin])
        RANDOM_RESET = False
        ACTION_SCALE = (args.pos_scale, args.rot_scale, 1)
        DISPLAY_IMAGE = False
        MAX_EPISODE_LENGTH = args.steps + 2
        COMPLIANCE_PARAM = COMPLIANCE_PARAM
        PRECISION_PARAM = PRECISION_PARAM

    return SmokeEnvConfig()


def print_obs_summary(obs):
    print("observation keys:", sorted(obs.keys()))
    print("state keys:", sorted(obs["state"].keys()))
    for key, value in obs["state"].items():
        arr = np.asarray(value)
        print(f"state[{key}]: shape={arr.shape} dtype={arr.dtype}")
    for key, value in obs["images"].items():
        arr = np.asarray(value)
        print(
            f"image[{key}]: shape={arr.shape} dtype={arr.dtype} "
            f"min={int(arr.min())} max={int(arr.max())} mean={float(arr.mean()):.3f}"
        )


def save_debug_image(obs, camera_name, output_path):
    image = obs["images"][camera_name]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"saved image: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://192.168.1.104:5000/")
    parser.add_argument("--camera-name", default="wrist_1")
    parser.add_argument("--shm-name", default="hilserl_wrist_1")
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--stale-after", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--hz", type=int, default=5)
    parser.add_argument("--pos-scale", type=float, default=0.001)
    parser.add_argument("--rot-scale", type=float, default=0.005)
    parser.add_argument(
        "--xyz-margin", type=float, nargs=3, default=(0.01, 0.01, 0.01)
    )
    parser.add_argument(
        "--rpy-margin", type=float, nargs=3, default=(0.05, 0.05, 0.05)
    )
    parser.add_argument("--output-image", default="/tmp/hilserl_smoke_wrist_1.png")
    args = parser.parse_args()

    current_pose = get_current_pose(args.server_url)
    print("current pose euler:", current_pose.tolist())
    config = make_config(args, current_pose)

    env = FrankaEnv(hz=args.hz, fake_env=False, save_video=False, config=config)
    try:
        obs, info = env.reset()
        print("reset info:", info)
        print_obs_summary(obs)
        save_debug_image(obs, args.camera_name, args.output_image)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for step in range(args.steps):
            obs, reward, done, truncated, info = env.step(action)
            print(
                f"step={step} reward={reward} done={done} truncated={truncated} "
                f"succeed={info.get('succeed')}"
            )
            time.sleep(0.1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
