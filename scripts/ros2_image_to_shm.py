#!/usr/bin/env python3
"""Mirror a ROS2 image topic into a local shared-memory frame buffer.

Run this from a ROS2-sourced shell on the machine that receives camera topics.
HIL-SERL itself can then read the frame from shared memory without importing ROS2.
"""

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for package_dir in (REPO_ROOT / "serl_robot_infra",):
    package_dir = str(package_dir)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

from franka_env.camera.shared_memory_capture import SharedMemoryFrameWriter


def parse_crop(value):
    if value is None:
        return None
    try:
        y_part, x_part = value.split(",", 1)
        y1, y2 = [int(v) for v in y_part.split(":", 1)]
        x1, x2 = [int(v) for v in x_part.split(":", 1)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "crop must be formatted as y1:y2,x1:x2"
        ) from exc
    return y1, y2, x1, x2


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def image_msg_to_bgr(msg):
    if msg.encoding not in ("bgr8", "rgb8", "bgra8", "rgba8", "mono8"):
        raise ValueError(f"Unsupported raw image encoding: {msg.encoding}")

    channels = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }[msg.encoding]
    frame = np.frombuffer(msg.data, dtype=np.uint8)
    frame = frame.reshape((msg.height, msg.width, channels))

    if msg.encoding == "rgb8":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "rgba8":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    elif msg.encoding == "bgra8":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif msg.encoding == "mono8":
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def compressed_msg_to_bgr(msg):
    encoded = np.frombuffer(msg.data, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode compressed image")
    return frame


class ImageToSharedMemory(Node):
    def __init__(self, args):
        super().__init__("hilserl_image_to_shm")
        self.args = args
        self.crop = args.crop
        self.output_shape = (args.height, args.width, 3)
        self.writer = SharedMemoryFrameWriter(
            name=args.name,
            shape=self.output_shape,
            shm_name=args.shm_name,
        )
        self.received_once = False
        self.frame_count = 0
        self.last_log_time = time.time()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        msg_type = CompressedImage if args.compressed else Image
        self.create_subscription(msg_type, args.topic, self.callback, qos)
        self.get_logger().info(
            f"Mirroring {args.topic} to shared memory {self.writer.shm_name} "
            f"with shape {self.output_shape}"
        )

    def callback(self, msg):
        try:
            frame = compressed_msg_to_bgr(msg) if self.args.compressed else image_msg_to_bgr(msg)
            if self.crop is not None:
                y1, y2, x1, x2 = self.crop
                frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, (self.args.width, self.args.height))
            timestamp_ns = (
                stamp_to_ns(msg.header.stamp)
                if self.args.use_message_stamp
                else time.time_ns()
            )
            self.writer.write(frame, timestamp_ns=timestamp_ns)
            self.frame_count += 1
            self.received_once = True
            now = time.time()
            if now - self.last_log_time > 5:
                self.get_logger().info(f"Wrote {self.frame_count} frames")
                self.last_log_time = now
        except Exception as exc:
            self.get_logger().warning(f"Failed to mirror frame: {exc}")

    def destroy_node(self):
        self.writer.close()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--shm-name", default=None)
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--crop", type=parse_crop, default=None)
    parser.add_argument(
        "--use-message-stamp",
        action="store_true",
        help="Use the ROS message header stamp instead of the local receive time.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after receiving and writing the first frame.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = ImageToSharedMemory(args)
    try:
        if args.once:
            while rclpy.ok() and not node.received_once:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            try:
                rclpy.spin(node)
            except ExternalShutdownException:
                pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
