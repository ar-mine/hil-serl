"""Teleoperate a Franka arm through the HIL-SERL HTTP robot server."""

import argparse
import time

import numpy as np
import requests
from scipy.spatial.transform import Rotation

from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert


def _url(base_url: str, endpoint: str) -> str:
    return base_url.rstrip("/") + "/" + endpoint.lstrip("/")


def _post_json(base_url: str, endpoint: str, timeout: float, payload=None):
    response = requests.post(_url(base_url, endpoint), json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _post_command(base_url: str, endpoint: str, timeout: float, payload=None) -> bool:
    try:
        response = requests.post(_url(base_url, endpoint), json=payload, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        print(f"Warning: {endpoint} request failed or timed out: {exc}")
        return False


def _get_pose(base_url: str, timeout: float) -> np.ndarray:
    state = _post_json(base_url, "getstate", timeout)
    return np.asarray(state["pose"], dtype=np.float64)


def _parse_vec(text: str, expected_len: int, name: str) -> np.ndarray:
    parts = [float(x) for x in text.split(",")]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(
            f"{name} must contain {expected_len} comma-separated floats"
        )
    return np.asarray(parts, dtype=np.float64)


def _apply_deadband(action: np.ndarray, deadband: float) -> np.ndarray:
    action = action.copy()
    action[np.abs(action) < deadband] = 0.0
    return np.clip(action, -1.0, 1.0)


def _keep_dominant_axis(action: np.ndarray) -> np.ndarray:
    filtered = np.zeros_like(action)
    axis_idx = int(np.argmax(np.abs(action)))
    if np.abs(action[axis_idx]) > 0.0:
        filtered[axis_idx] = action[axis_idx]
    return filtered


def _clip_pose(
    pose: np.ndarray,
    xyz_low: np.ndarray,
    xyz_high: np.ndarray,
    rpy_low: np.ndarray,
    rpy_high: np.ndarray,
) -> np.ndarray:
    clipped = pose.copy()
    clipped[:3] = np.clip(clipped[:3], xyz_low, xyz_high)
    rpy = Rotation.from_quat(clipped[3:]).as_euler("xyz")
    rpy = np.clip(rpy, rpy_low, rpy_high)
    clipped[3:] = Rotation.from_euler("xyz", rpy).as_quat()
    return clipped


def main():
    parser = argparse.ArgumentParser(
        description="Control a Franka arm with a SpaceMouse via the HIL-SERL HTTP API."
    )
    parser.add_argument("--server-url", default="http://192.168.1.104:5000/")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--pos-scale", type=float, default=0.003)
    parser.add_argument("--rot-scale", type=float, default=0.03)
    parser.add_argument("--deadband", type=float, default=0.08)
    parser.add_argument("--request-timeout", type=float, default=0.5)
    parser.add_argument("--gripper-timeout", type=float, default=3.0)
    parser.add_argument("--gripper-cooldown", type=float, default=1.5)
    parser.add_argument(
        "--multi-axis",
        action="store_true",
        help="Allow simultaneous SpaceMouse axes instead of keeping only the dominant axis.",
    )
    parser.add_argument(
        "--xyz-margin",
        type=float,
        default=0.03,
        help="Default Cartesian safety half-width around the current pose in meters.",
    )
    parser.add_argument(
        "--rpy-margin",
        type=float,
        default=0.25,
        help="Default orientation safety half-width around the current pose in radians.",
    )
    parser.add_argument(
        "--xyz-low",
        type=lambda value: _parse_vec(value, 3, "xyz-low"),
        default=None,
        help="Absolute low xyz safety bound, e.g. 0.45,-0.20,0.20.",
    )
    parser.add_argument(
        "--xyz-high",
        type=lambda value: _parse_vec(value, 3, "xyz-high"),
        default=None,
        help="Absolute high xyz safety bound, e.g. 0.65,0.20,0.45.",
    )
    parser.add_argument(
        "--rpy-low",
        type=lambda value: _parse_vec(value, 3, "rpy-low"),
        default=None,
        help="Absolute low rpy safety bound in radians.",
    )
    parser.add_argument(
        "--rpy-high",
        type=lambda value: _parse_vec(value, 3, "rpy-high"),
        default=None,
        help="Absolute high rpy safety bound in radians.",
    )
    parser.add_argument(
        "--base-frame",
        action="store_true",
        help="Interpret SpaceMouse deltas in the robot base frame instead of tool frame.",
    )
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target poses without sending /pose commands.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the current processed tx/ty/tz/rx/ry/rz control values.",
    )
    args = parser.parse_args()

    pose = _get_pose(args.server_url, args.request_timeout)
    start_rpy = Rotation.from_quat(pose[3:]).as_euler("xyz")

    xyz_low = args.xyz_low if args.xyz_low is not None else pose[:3] - args.xyz_margin
    xyz_high = args.xyz_high if args.xyz_high is not None else pose[:3] + args.xyz_margin
    rpy_low = args.rpy_low if args.rpy_low is not None else start_rpy - args.rpy_margin
    rpy_high = args.rpy_high if args.rpy_high is not None else start_rpy + args.rpy_margin

    print("Starting SpaceMouse teleop")
    print(f"server_url: {args.server_url}")
    print(f"xyz_low:  {xyz_low}")
    print(f"xyz_high: {xyz_high}")
    print(f"rpy_low:  {rpy_low}")
    print(f"rpy_high: {rpy_high}")
    print("Left button closes gripper; right button opens gripper.")
    print("Press Ctrl-C to stop.")

    if not args.dry_run:
        confirmation = input('Type "yes" to enable live robot motion: ')
        if confirmation.strip().lower() != "yes":
            print("Aborted.")
            return

    expert = SpaceMouseExpert()
    last_gripper_time = 0.0
    period = 1.0 / args.hz

    try:
        while True:
            start = time.time()
            action, buttons = expert.get_action()
            action = _apply_deadband(np.asarray(action[:6], dtype=np.float64), args.deadband)
            if not args.multi_axis:
                action = _keep_dominant_axis(action)
            action[0] *= -1.0
            action[2] *= -1.0
            pose = _get_pose(args.server_url, args.request_timeout)
            current_rot = Rotation.from_quat(pose[3:])

            if args.base_frame:
                xyz_delta = action[:3] * args.pos_scale
                rot_delta = action[3:6] * args.rot_scale
            else:
                rot_matrix = current_rot.as_matrix()
                xyz_delta = rot_matrix @ (action[:3] * args.pos_scale)
                rot_delta = rot_matrix @ (action[3:6] * args.rot_scale)

            target = pose.copy()
            target[:3] += xyz_delta
            target[3:] = (Rotation.from_rotvec(rot_delta) * current_rot).as_quat()
            target = _clip_pose(target, xyz_low, xyz_high, rpy_low, rpy_high)
            if args.debug:
                current_rpy = current_rot.as_euler("xyz")
                target_rpy = Rotation.from_quat(target[3:]).as_euler("xyz")
                print(
                    "control "
                    f"tx={action[0]: .3f} ty={action[1]: .3f} tz={action[2]: .3f} "
                    f"rx={action[3]: .3f} ry={action[4]: .3f} rz={action[5]: .3f} | "
                    f"dxyz={target[:3] - pose[:3]} drpy={target_rpy - current_rpy}"
                )

            if not args.dry_run:
                requests.post(
                    _url(args.server_url, "pose"),
                    json={"arr": target.astype(np.float32).tolist()},
                    timeout=args.request_timeout,
                ).raise_for_status()
            else:
                print(np.array2string(target, precision=4, suppress_small=True))

            if (
                not args.disable_gripper
                and time.time() - last_gripper_time > args.gripper_cooldown
            ):
                left = len(buttons) > 0 and bool(buttons[0])
                right = len(buttons) > 1 and bool(buttons[1])
                if left and not args.dry_run:
                    last_gripper_time = time.time()
                    _post_command(
                        args.server_url,
                        "close_gripper",
                        timeout=args.gripper_timeout,
                    )
                elif right and not args.dry_run:
                    last_gripper_time = time.time()
                    _post_command(
                        args.server_url,
                        "open_gripper",
                        timeout=args.gripper_timeout,
                    )

            elapsed = time.time() - start
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\nStopping SpaceMouse teleop.")
    finally:
        expert.close()


if __name__ == "__main__":
    main()
