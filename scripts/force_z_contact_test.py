#!/usr/bin/env python3
"""Slow Z-axis force contact test through the HIL-SERL HTTP robot API."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import requests


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def normalize_url(url):
    return url if url.endswith("/") else url + "/"


class FrankaHttpClient:
    def __init__(self, server_url, timeout=0.5, execute=False):
        self.url = normalize_url(server_url)
        self.timeout = timeout
        self.execute = execute

    def post(self, endpoint, json=None):
        if endpoint == "pose" and not self.execute:
            print(f"[dry-run] POST {self.url}{endpoint} {json}", flush=True)
            return None
        response = requests.post(self.url + endpoint, json=json, timeout=self.timeout)
        response.raise_for_status()
        return response

    def get_state(self):
        response = requests.post(self.url + "getstate", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def pose(self):
        return np.asarray(self.get_state()["pose"], dtype=np.float64)

    def force(self):
        return np.asarray(self.get_state()["force"], dtype=np.float64)

    def send_pose(self, pose):
        self.post("pose", json={"arr": np.asarray(pose, dtype=np.float32).tolist()})


def print_force_line(prefix, pose, force, baseline_abs_force=None, stop_axis="z"):
    abs_force = np.abs(force)
    if baseline_abs_force is None:
        delta_abs_force = np.zeros(3)
        stop_metric = 0.0
    else:
        delta_abs_force = np.abs(abs_force - baseline_abs_force)
        if stop_axis == "any":
            stop_metric = float(np.max(delta_abs_force))
        else:
            stop_metric = float(delta_abs_force[AXIS_INDEX[stop_axis]])

    print(
        f"{prefix} actual_z={pose[2]:.5f} "
        f"Fx={force[0]: .3f} Fy={force[1]: .3f} Fz={force[2]: .3f} | "
        f"|Fx|={abs_force[0]:.3f} |Fy|={abs_force[1]:.3f} |Fz|={abs_force[2]:.3f} | "
        f"d|Fx|={delta_abs_force[0]:.3f} "
        f"d|Fy|={delta_abs_force[1]:.3f} "
        f"d|Fz|={delta_abs_force[2]:.3f} "
        f"metric({stop_axis})={stop_metric:.3f}",
        flush=True,
    )
    return abs_force, delta_abs_force, stop_metric


def move_linear(client, target, step_size, period, label="approach"):
    start = client.pose()
    target = np.asarray(target, dtype=np.float64)
    distance = np.linalg.norm(target[:3] - start[:3])
    steps = max(1, int(np.ceil(distance / step_size)))
    for i, alpha in enumerate(np.linspace(0.0, 1.0, steps + 1)[1:], start=1):
        pose = start.copy()
        pose[:3] = (1.0 - alpha) * start[:3] + alpha * target[:3]
        pose[3:] = target[3:]
        client.send_pose(pose)
        time.sleep(period)
        state_pose = client.pose()
        force = client.force()
        print_force_line(f"{label}={i:03d}/{steps:03d}", state_pose, force)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://192.168.1.104:5000/")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--period", type=float, default=0.25)
    parser.add_argument("--approach-step", type=float, default=0.001)
    parser.add_argument("--descent-step", type=float, default=0.0002)
    parser.add_argument("--max-descent", type=float, default=0.02)
    parser.add_argument("--force-delta", type=float, default=2.0)
    parser.add_argument(
        "--force-delta-axis",
        choices=("x", "y", "z", "any"),
        default="z",
        help="Stop when the absolute force change on this axis exceeds force-delta.",
    )
    parser.add_argument("--max-abs-fz", type=float, default=15.0)
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--log-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    client = FrankaHttpClient(args.server_url, timeout=args.timeout, execute=args.execute)

    current = client.pose()
    target = current.copy()
    target[:3] = [args.x, args.y, args.z]

    log_path = args.log_path
    if log_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"/tmp/hilserl_force_z_contact_{stamp}.csv"
    log_path = Path(log_path)

    print(f"current pose: {current.tolist()}", flush=True)
    print(f"target pose: {target.tolist()}", flush=True)
    print(f"execute: {args.execute}", flush=True)
    print(f"log path: {log_path}", flush=True)

    if args.execute and not args.yes:
        answer = input("Type 'yes' to run slow Z force test on the real robot: ")
        if answer != "yes":
            print("aborted", flush=True)
            return

    print("approaching target xy/z slowly", flush=True)
    move_linear(client, target, args.approach_step, args.period, label="approach")
    time.sleep(args.settle_time)

    baseline_force = client.force()
    baseline_abs_force = np.abs(baseline_force)
    baseline_abs_fz = float(baseline_abs_force[2])
    print(
        "baseline force: "
        f"Fx={baseline_force[0]:.3f}N Fy={baseline_force[1]:.3f}N "
        f"Fz={baseline_force[2]:.3f}N | "
        f"|Fx|={baseline_abs_force[0]:.3f}N "
        f"|Fy|={baseline_abs_force[1]:.3f}N "
        f"|Fz|={baseline_abs_force[2]:.3f}N",
        flush=True,
    )

    rows = []
    rows.append(
        {
            "t": time.time(),
            "cmd_z": target[2],
            "actual_z": client.pose()[2],
            "fx": baseline_force[0],
            "fy": baseline_force[1],
            "fz": baseline_force[2],
            "abs_fx": baseline_abs_force[0],
            "abs_fy": baseline_abs_force[1],
            "abs_fz": baseline_abs_fz,
            "delta_abs_fx": 0.0,
            "delta_abs_fy": 0.0,
            "delta_abs_fz": 0.0,
            "stop_metric": 0.0,
        }
    )

    steps = int(np.ceil(args.max_descent / args.descent_step))
    stop_reason = "max_descent"
    last_cmd = target.copy()
    for i in range(1, steps + 1):
        cmd = target.copy()
        cmd[2] = target[2] - i * args.descent_step
        client.send_pose(cmd)
        time.sleep(args.period)

        state_pose = client.pose()
        force = client.force()
        abs_force = np.abs(force)
        delta_abs_force = np.abs(abs_force - baseline_abs_force)
        abs_fz = float(abs_force[2])
        delta_fz = float(delta_abs_force[2])
        if args.force_delta_axis == "any":
            stop_metric = float(np.max(delta_abs_force))
        else:
            stop_metric = float(delta_abs_force[AXIS_INDEX[args.force_delta_axis]])
        last_cmd = cmd.copy()

        row = {
            "t": time.time(),
            "cmd_z": cmd[2],
            "actual_z": state_pose[2],
            "fx": force[0],
            "fy": force[1],
            "fz": force[2],
            "abs_fx": abs_force[0],
            "abs_fy": abs_force[1],
            "abs_fz": abs_fz,
            "delta_abs_fx": delta_abs_force[0],
            "delta_abs_fy": delta_abs_force[1],
            "delta_abs_fz": delta_fz,
            "stop_metric": stop_metric,
        }
        rows.append(row)
        print(
            f"step={i:03d} cmd_z={cmd[2]:.5f} actual_z={state_pose[2]:.5f} "
            f"Fx={force[0]: .3f} Fy={force[1]: .3f} Fz={force[2]: .3f} | "
            f"|Fx|={abs_force[0]:.3f} |Fy|={abs_force[1]:.3f} |Fz|={abs_fz:.3f} | "
            f"d|Fx|={delta_abs_force[0]:.3f} "
            f"d|Fy|={delta_abs_force[1]:.3f} "
            f"d|Fz|={delta_fz:.3f} "
            f"metric({args.force_delta_axis})={stop_metric:.3f}",
            flush=True,
        )

        if abs_fz >= args.max_abs_fz:
            stop_reason = "max_abs_fz"
            break
        if stop_metric >= args.force_delta:
            stop_reason = "force_delta"
            break

    client.send_pose(last_cmd)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"stopped: {stop_reason}", flush=True)
    print(f"saved log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
