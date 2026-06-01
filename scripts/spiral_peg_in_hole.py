#!/usr/bin/env python3
"""Conservative HTTP-based spiral search policy for peg-in-hole.

The policy uses the existing HIL-SERL robot HTTP API:
approach above an approximate hole location, descend until force contact,
spiral with a small downward bias, then insert.
"""

import argparse
import math
import time

import numpy as np
import requests
from scipy.spatial.transform import Rotation


FORCE_AXES = {"x": 0, "y": 1, "z": 2}


def normalize_url(url):
    return url if url.endswith("/") else url + "/"


class FrankaHttpClient:
    def __init__(self, server_url, timeout=0.5, dry_run=True):
        self.url = normalize_url(server_url)
        self.timeout = timeout
        self.dry_run = dry_run

    def post(self, endpoint, json=None):
        if self.dry_run and endpoint == "pose":
            print(f"[dry-run] POST {self.url}{endpoint} {json}", flush=True)
            return None
        response = requests.post(self.url + endpoint, json=json, timeout=self.timeout)
        response.raise_for_status()
        return response

    def get_state(self):
        response = self.post("getstate")
        if response is None:
            raise RuntimeError("Cannot read robot state in dry-run without server access")
        return response.json()

    def pose(self):
        return np.asarray(self.get_state()["pose"], dtype=np.float64)

    def force(self):
        return np.asarray(self.get_state()["force"], dtype=np.float64)

    def send_pose(self, pose):
        self.post("pose", json={"arr": np.asarray(pose, dtype=np.float32).tolist()})

    def update_param(self, params):
        if params:
            self.post("update_param", json=params)


def force_value(force, axis, sign):
    value = float(force[FORCE_AXES[axis]])
    if sign == "positive":
        return value
    if sign == "negative":
        return -value
    return abs(value)


def print_force(prefix, force, extra=""):
    print(
        f"{prefix} Fx={force[0]: .3f} Fy={force[1]: .3f} Fz={force[2]: .3f}"
        + (f" {extra}" if extra else ""),
        flush=True,
    )


def interpolate_pose(start, goal, step_size):
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    distance = np.linalg.norm(goal[:3] - start[:3])
    steps = max(1, int(math.ceil(distance / step_size)))
    for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
        pose = start.copy()
        pose[:3] = (1.0 - alpha) * start[:3] + alpha * goal[:3]
        pose[3:] = goal[3:]
        yield pose


def move_linear(client, goal, args, label):
    start = client.pose()
    print(f"{label}: {start[:3]} -> {np.asarray(goal)[:3]}", flush=True)
    for pose in interpolate_pose(start, goal, args.cart_step):
        client.send_pose(pose)
        time.sleep(args.period)
        if not client.dry_run:
            force = client.force()
            f_contact = force_value(force, args.force_axis, args.force_sign)
            print_force(f"{label}", force, extra=f"metric={f_contact:.3f}")
            if f_contact > args.max_force:
                raise RuntimeError(
                    f"Force limit exceeded during {label}: {force.tolist()}"
                )


def wait_settle(client, args):
    time.sleep(args.settle_time)
    if client.dry_run:
        return np.zeros(3)
    return client.force()


def descend_until_contact(client, pose, args):
    print(
        "descending until sustained contact "
        f"({args.force_axis.upper()} {args.force_sign} >= {args.contact_force:.3f}N "
        f"for {args.contact_hold_time:.3f}s)",
        flush=True,
    )
    z0 = pose[2]
    steps = int(math.ceil(args.max_descent / args.descent_step))
    contact_started_at = None
    for i in range(1, steps + 1):
        target = pose.copy()
        target[2] = z0 - i * args.descent_step
        client.send_pose(target)
        time.sleep(args.period)
        force = client.force()
        f_contact = force_value(force, args.force_axis, args.force_sign)
        now = time.monotonic()
        if f_contact >= args.contact_force:
            if contact_started_at is None:
                contact_started_at = now
            contact_elapsed = now - contact_started_at
        else:
            contact_started_at = None
            contact_elapsed = 0.0

        print_force(
            f"  descend={i:03d} z={target[2]:.5f}",
            force,
            extra=(
                f"metric={f_contact:.3f} "
                f"hold={contact_elapsed:.2f}/{args.contact_hold_time:.2f}s"
            ),
        )
        if f_contact > args.max_force:
            raise RuntimeError(f"Force limit exceeded while descending: {force.tolist()}")
        if contact_elapsed >= args.contact_hold_time:
            print(f"sustained contact detected at z={target[2]:.5f}", flush=True)
            return target.copy(), force
    raise RuntimeError("No contact detected within max descent")


def spiral_search(client, contact_pose, args):
    print("spiral search", flush=True)
    theta = 0.0
    step_idx = 0
    best_pose = contact_pose.copy()
    prev_insert_metric = None
    insertion_jump_armed = False
    while True:
        radius = args.spiral_pitch * theta / (2.0 * math.pi)
        if radius > args.max_radius:
            break

        target = contact_pose.copy()
        target[0] = args.hole_x + radius * math.cos(theta)
        target[1] = args.hole_y + radius * math.sin(theta)
        if args.spiral_down:
            target[2] = contact_pose[2] - min(
                args.search_down_bias,
                args.search_down_per_rev * theta / (2.0 * math.pi),
            )

        client.send_pose(target)
        time.sleep(args.period)
        force = client.force()
        f_contact = force_value(force, args.force_axis, args.force_sign)
        f_insert_metric = force_value(
            force, args.insertion_force_axis, args.insertion_force_sign
        )
        if prev_insert_metric is None:
            force_jump = 0.0
        else:
            force_jump = abs(f_insert_metric - prev_insert_metric)
        prev_insert_metric = f_insert_metric
        if force_jump >= args.insertion_force_jump:
            insertion_jump_armed = True
        state_pose = client.pose()
        print(
            f"  r={radius:.5f}, theta={theta:.2f}, "
            f"cmd_z={target[2]:.5f}, z={state_pose[2]:.5f}, "
            f"Fx={force[0]: .3f} Fy={force[1]: .3f} Fz={force[2]: .3f}, "
            f"metric={f_contact:.3f}, "
            f"insert_metric={f_insert_metric:.3f}, "
            f"force_jump={force_jump:.3f}, "
            f"jump_armed={insertion_jump_armed}",
            flush=True,
        )

        if f_contact > args.max_force:
            raise RuntimeError(f"Force limit exceeded during spiral: {force.tolist()}")

        best_pose = state_pose.copy()
        z_drop = contact_pose[2] - state_pose[2]
        has_release_force = f_insert_metric < args.insertion_release_force
        has_required_z_drop = (
            not args.require_insertion_z_drop
            or z_drop >= args.insertion_detect_depth
        )
        if insertion_jump_armed and has_release_force and has_required_z_drop:
            print(
                "possible insertion detected, "
                f"z_drop={z_drop:.5f}, force_jump={force_jump:.3f}, "
                f"insert_metric={f_insert_metric:.3f}",
                flush=True,
            )
            return best_pose

        theta += args.spiral_theta_step
        step_idx += 1

    print(
        "spiral ended without clear drop; using last pose for cautious insertion",
        flush=True,
    )
    return best_pose


def insert_down(client, pose, args):
    print(
        "inserting until bottom contact "
        f"({args.insert_force_axis.upper()} {args.insert_force_sign} "
        f">= {args.insert_contact_force:.3f}N for "
        f"{args.insert_contact_hold_time:.3f}s, "
        f"deadband={args.insert_force_deadband:.4f}m)",
        flush=True,
    )
    start = pose.copy()
    steps = int(math.ceil(args.insert_depth / args.insert_step))
    contact_started_at = None
    last_cmd = start.copy()

    for i in range(1, steps + 1):
        cmd = start.copy()
        cmd[2] = start[2] - i * args.insert_step
        last_cmd = cmd.copy()
        client.send_pose(cmd)
        time.sleep(args.period)

        state_pose = client.pose()
        force = client.force()
        inserted_depth = start[2] - cmd[2]
        insert_metric = force_value(
            force, args.insert_force_axis, args.insert_force_sign
        )

        if inserted_depth < args.insert_force_deadband:
            contact_started_at = None
            contact_elapsed = 0.0
            force_active = False
        else:
            force_active = True
            now = time.monotonic()
            if insert_metric >= args.insert_contact_force:
                if contact_started_at is None:
                    contact_started_at = now
                contact_elapsed = now - contact_started_at
            else:
                contact_started_at = None
                contact_elapsed = 0.0

        print(
            f"  insert={i:03d} cmd_z={cmd[2]:.5f} z={state_pose[2]:.5f} "
            f"depth={inserted_depth:.5f} "
            f"Fx={force[0]: .3f} Fy={force[1]: .3f} Fz={force[2]: .3f}, "
            f"metric={insert_metric:.3f}, "
            f"force_active={force_active}, "
            f"hold={contact_elapsed:.2f}/{args.insert_contact_hold_time:.2f}s",
            flush=True,
        )

        if insert_metric > args.max_force:
            raise RuntimeError(
                f"Force limit exceeded during final insertion: {force.tolist()}"
            )
        if force_active and contact_elapsed >= args.insert_contact_hold_time:
            print(
                f"bottom contact detected at depth={inserted_depth:.5f}",
                flush=True,
            )
            return last_cmd

    print("final insertion depth reached without bottom contact", flush=True)
    return last_cmd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://192.168.1.104:5000/")
    parser.add_argument("--hole-x", type=float, required=True)
    parser.add_argument("--hole-y", type=float, required=True)
    parser.add_argument(
        "--approach-z",
        type=float,
        default=None,
        help=(
            "Optional Cartesian approach z. Defaults to current z so the policy "
            "only aligns x/y before descending."
        ),
    )
    parser.add_argument(
        "--rpy",
        type=float,
        nargs=3,
        default=None,
        help="Optional fixed end-effector rpy in radians. Defaults to current orientation.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--period", type=float, default=0.5)
    parser.add_argument("--settle-time", type=float, default=0.3)
    parser.add_argument("--cart-step", type=float, default=0.002)
    parser.add_argument("--descent-step", type=float, default=0.0004)
    parser.add_argument("--max-descent", type=float, default=0.04)
    parser.add_argument("--contact-force", type=float, default=2.0)
    parser.add_argument("--contact-hold-time", type=float, default=1.0)
    parser.add_argument("--max-force", type=float, default=25.0)
    parser.add_argument("--force-axis", choices=FORCE_AXES.keys(), default="z")
    parser.add_argument(
        "--force-sign",
        choices=("positive", "negative", "absolute"),
        default="positive",
    )
    parser.add_argument("--spiral-pitch", type=float, default=0.001)
    parser.add_argument("--spiral-theta-step", type=float, default=0.25)
    parser.add_argument("--max-radius", type=float, default=0.006)
    parser.add_argument("--search-down-bias", type=float, default=0.003)
    parser.add_argument("--search-down-per-rev", type=float, default=0.0005)
    parser.add_argument(
        "--spiral-down",
        action="store_true",
        help="Enable downward bias during spiral search. Defaults to pure XY spiral.",
    )
    parser.add_argument("--insertion-detect-depth", type=float, default=0.002)
    parser.add_argument(
        "--require-insertion-z-drop",
        action="store_true",
        help=(
            "Require z_drop >= insertion-detect-depth in addition to force jump. "
            "Defaults off because pure XY spiral keeps z fixed."
        ),
    )
    parser.add_argument("--insertion-force-jump", type=float, default=3.0)
    parser.add_argument("--insertion-release-force", type=float, default=2.0)
    parser.add_argument("--insertion-force-axis", choices=FORCE_AXES.keys(), default="z")
    parser.add_argument(
        "--insertion-force-sign",
        choices=("positive", "negative", "absolute"),
        default="positive",
    )
    parser.add_argument("--insert-depth", type=float, default=0.01)
    parser.add_argument("--insert-step", type=float, default=0.0004)
    parser.add_argument("--insert-force-deadband", type=float, default=0.002)
    parser.add_argument("--insert-contact-force", type=float, default=2.0)
    parser.add_argument("--insert-contact-hold-time", type=float, default=0.5)
    parser.add_argument("--insert-force-axis", choices=FORCE_AXES.keys(), default="z")
    parser.add_argument(
        "--insert-force-sign",
        choices=("positive", "negative", "absolute"),
        default="positive",
    )
    parser.add_argument(
        "--continue-insert",
        action="store_true",
        help="Continue with the final insertion motion after insertion is detected.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = FrankaHttpClient(
        args.server_url, timeout=args.timeout, dry_run=not args.execute
    )
    current_pose = client.pose()

    if args.rpy is None:
        quat = current_pose[3:]
    else:
        quat = Rotation.from_euler("xyz", args.rpy).as_quat()

    approach_pose = current_pose.copy()
    approach_pose[0] = args.hole_x
    approach_pose[1] = args.hole_y
    approach_pose[2] = current_pose[2] if args.approach_z is None else args.approach_z
    approach_pose[3:] = quat

    print(f"current pose: {current_pose.tolist()}", flush=True)
    print(f"approach pose: {approach_pose.tolist()}", flush=True)
    print(f"execute: {args.execute}", flush=True)

    if args.execute and not args.yes:
        answer = input("Type 'yes' to run spiral peg-in-hole on the real robot: ")
        if answer != "yes":
            print("aborted", flush=True)
            return

    move_linear(client, approach_pose, args, "approach")
    wait_settle(client, args)
    contact_pose, _ = descend_until_contact(client, approach_pose, args)
    search_pose = spiral_search(client, contact_pose, args)
    if args.continue_insert:
        insert_down(client, search_pose, args)
    else:
        print(
            "insertion/search point reached; stopping before final insert motion",
            flush=True,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
