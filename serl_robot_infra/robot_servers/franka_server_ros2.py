"""
ROS 2 HTTP bridge for HIL-SERL Franka environments.

Run this on the NUC that is already running franka_ros2 and the custom
JointImpedanceExampleController.  It keeps the ROS 1 franka_server.py HTTP
surface mostly intact so the existing FrankaEnv can talk to this process.
"""

import argparse
import ast
import threading
import time
from typing import Any, Dict, Optional

# Werkzeug 0.15.x still constructs ast.Module(body) without the Python 3.8+
# type_ignores field.  This NUC currently has Flask 1.1.4 / Werkzeug 0.15.4,
# so keep the server runnable without requiring a system package update.
if "type_ignores" in getattr(ast.Module, "_fields", ()):
    _ast_module = ast.Module

    def _module_with_type_ignores(*args, **kwargs):
        kwargs.setdefault("type_ignores", [])
        return _ast_module(*args, **kwargs)

    ast.Module = _module_with_type_ignores

from flask import Flask, jsonify, request
import numpy as np
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from franka_msgs.action import Grasp, Homing, Move, MoveToStart, SetTrajectory
from franka_msgs.msg import FrankaRobotState, GraspEpsilon
from franka_msgs.srv import GetPose, SetControlMode
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class FrankaRos2Server(Node):
    def __init__(
        self,
        controller_name: str,
        state_topic: str,
        jacobian_topic: str,
        gripper_joint_topic: str,
        pose_service: str,
        control_mode_service: str,
        trajectory_action: str,
        equilibrium_pose_topic: str,
        pose_command_mode: str,
        move_to_start_action: str,
        gripper_move_action: str,
        gripper_grasp_action: str,
        gripper_homing_action: str,
        pose_steps: int,
        action_timeout: float,
        default_gripper_open_width: float,
        default_gripper_speed: float,
        default_gripper_force: float,
    ):
        super().__init__("franka_ros2_http_server")
        self.controller_name = controller_name
        self.pose_steps = max(1, pose_steps)
        self.pose_command_mode = pose_command_mode
        self.action_timeout = action_timeout
        self.default_gripper_open_width = float(default_gripper_open_width)
        self.default_gripper_speed = default_gripper_speed
        self.default_gripper_force = default_gripper_force

        self._lock = threading.Lock()
        self.pos = np.zeros(7)
        self.vel = np.zeros(6)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.jacobian = np.zeros((6, 7))
        self.gripper_pos = 0.0
        self.gripper_load_mass = 0.0
        self.gripper_load_center_of_mass = [0.0, 0.0, 0.0]
        self.last_command_pose: Optional[np.ndarray] = None
        self.last_command_pose_time = 0.0
        self._last_state_time = 0.0
        self._last_gripper_time = 0.0

        self.create_subscription(
            FrankaRobotState,
            state_topic,
            self._state_callback,
            10,
        )
        self.create_subscription(
            JointState,
            gripper_joint_topic,
            self._gripper_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            jacobian_topic,
            self._jacobian_callback,
            10,
        )

        self._pose_client = self.create_client(GetPose, pose_service)
        self._control_mode_client = self.create_client(SetControlMode, control_mode_service)
        self._equilibrium_pose_publisher = self.create_publisher(
            PoseStamped,
            equilibrium_pose_topic,
            10,
        )
        self._set_parameters_client = self.create_client(
            SetParameters,
            f"/{controller_name}/set_parameters",
        )

        self._trajectory_client = ActionClient(self, SetTrajectory, trajectory_action)
        self._move_to_start_client = ActionClient(self, MoveToStart, move_to_start_action)
        self._gripper_move_client = ActionClient(self, Move, gripper_move_action)
        self._gripper_grasp_client = ActionClient(self, Grasp, gripper_grasp_action)
        self._gripper_homing_client = ActionClient(self, Homing, gripper_homing_action)

    def _state_callback(self, msg: FrankaRobotState) -> None:
        tmatrix = np.array(list(msg.o_t_ee), dtype=np.float64).reshape(4, 4).T
        quat = R.from_matrix(tmatrix[:3, :3]).as_quat()
        pose = np.concatenate([tmatrix[:3, 3], quat])

        dq = np.array(list(msg.dq), dtype=np.float64)
        wrench = np.array(list(msg.k_f_ext_hat_k), dtype=np.float64)

        with self._lock:
            self.pos = pose
            self.q = np.array(list(msg.q), dtype=np.float64)
            self.dq = dq
            self.force = wrench[:3]
            self.torque = wrench[3:]
            self.vel = self.jacobian @ dq
            self._last_state_time = time.time()

    def _gripper_callback(self, msg: JointState) -> None:
        if not msg.position:
            return
        with self._lock:
            self.gripper_pos = float(np.clip(np.sum(msg.position) / 0.08, 0.0, 1.0))
            self._last_gripper_time = time.time()

    def _jacobian_callback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 42:
            self.get_logger().warn(f"Ignoring Jacobian with {len(msg.data)} values; expected 42")
            return
        jacobian = np.array(msg.data, dtype=np.float64).reshape((6, 7), order="F")
        with self._lock:
            self.jacobian = jacobian
            self.vel = jacobian @ self.dq

    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pose": self.pos.tolist(),
                "vel": self.vel.tolist(),
                "force": self.force.tolist(),
                "torque": self.torque.tolist(),
                "q": self.q.tolist(),
                "dq": self.dq.tolist(),
                "jacobian": self.jacobian.tolist(),
                "gripper_pos": self.gripper_pos,
                "gripper_load_mass": self.gripper_load_mass,
                "gripper_load_center_of_mass": list(self.gripper_load_center_of_mass),
                "pose_command_mode": self.pose_command_mode,
                "state_age": time.time() - self._last_state_time if self._last_state_time else None,
                "gripper_age": time.time() - self._last_gripper_time if self._last_gripper_time else None,
            }

    def pose_debug_dict(self) -> Dict[str, Any]:
        with self._lock:
            current_pose = self.pos.copy()
            command_pose = None if self.last_command_pose is None else self.last_command_pose.copy()
            command_age = (
                time.time() - self.last_command_pose_time
                if self.last_command_pose_time
                else None
            )

        if command_pose is None:
            return {
                "has_command_pose": False,
                "current_pose": current_pose.tolist(),
                "command_pose": None,
                "command_age": command_age,
            }

        current_rot = R.from_quat(current_pose[3:])
        command_rot = R.from_quat(command_pose[3:])
        orientation_error = current_rot.inv() * command_rot
        return {
            "has_command_pose": True,
            "current_pose": current_pose.tolist(),
            "command_pose": command_pose.tolist(),
            "command_age": command_age,
            "position_error_current_minus_command": (current_pose[:3] - command_pose[:3]).tolist(),
            "orientation_error_rotvec_current_to_command": orientation_error.as_rotvec().tolist(),
            "orientation_error_angle_rad": float(orientation_error.magnitude()),
            "current_rpy": current_rot.as_euler("xyz").tolist(),
            "command_rpy": command_rot.as_euler("xyz").tolist(),
            "rpy_error_command_minus_current": (
                command_rot.as_euler("xyz") - current_rot.as_euler("xyz")
            ).tolist(),
            "pose_command_mode": self.pose_command_mode,
        }

    def pose_from_service(self) -> Optional[np.ndarray]:
        if not self._pose_client.wait_for_service(timeout_sec=0.2):
            return None
        future = self._pose_client.call_async(GetPose.Request())
        if not self._wait_for_future(future, self.action_timeout):
            return None
        pose = future.result().pose.pose
        return np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        )

    def send_pose(self, pose: np.ndarray) -> bool:
        if pose.shape != (7,):
            raise ValueError("pose must be xyz+quaternion with shape (7,)")
        with self._lock:
            self.last_command_pose = pose.copy()
            self.last_command_pose_time = time.time()
        pose_msg = self._pose_to_msg(pose)

        if self.pose_command_mode == "equilibrium":
            self._equilibrium_pose_publisher.publish(pose_msg)
            return True

        if not self._trajectory_client.wait_for_server(timeout_sec=self.action_timeout):
            self.get_logger().error("SetTrajectory action server is not available")
            return False

        goal = SetTrajectory.Goal()
        goal.trajectory = [pose_msg for _ in range(self.pose_steps)]
        return self._send_action_goal_and_wait(self._trajectory_client, goal)

    def _pose_to_msg(self, pose: np.ndarray) -> PoseStamped:
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "panda_link0"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position = Point(x=float(pose[0]), y=float(pose[1]), z=float(pose[2]))
        pose_msg.pose.orientation = Quaternion(
            x=float(pose[3]),
            y=float(pose[4]),
            z=float(pose[5]),
            w=float(pose[6]),
        )
        return pose_msg

    def joint_reset(self) -> bool:
        if not self._move_to_start_client.wait_for_server(timeout_sec=self.action_timeout):
            self.get_logger().error("MoveToStart action server is not available")
            return False
        return self._send_action_goal_and_wait(self._move_to_start_client, MoveToStart.Goal())

    def set_control_mode(self, mode: str) -> bool:
        if not self._control_mode_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("SetControlMode service is not available")
            return False
        req = SetControlMode.Request()
        req.mode = mode
        future = self._control_mode_client.call_async(req)
        if not self._wait_for_future(future, self.action_timeout):
            return False
        result = future.result()
        if not result.success:
            self.get_logger().warn(result.message)
        return bool(result.success)

    def open_gripper(self) -> bool:
        with self._lock:
            if self.gripper_pos > 0.95:
                return True
        return self.move_gripper_width(
            width=self.default_gripper_open_width,
            speed=self.default_gripper_speed,
        )

    def close_gripper(self, slow: bool = False) -> bool:
        if not self._gripper_grasp_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn("Franka gripper grasp action is not available")
            return False
        goal = Grasp.Goal()
        goal.width = 0.01
        goal.speed = 0.1 if slow else self.default_gripper_speed
        goal.force = self.default_gripper_force
        goal.epsilon = GraspEpsilon(inner=1.0, outer=1.0)
        return self._send_action_goal_and_wait(self._gripper_grasp_client, goal)

    def move_gripper_position(self, position: int) -> bool:
        position = int(np.clip(position, 0, 255))
        return self.move_gripper_width(width=float(position) / (255.0 * 10.0), speed=self.default_gripper_speed)

    def move_gripper_width(self, width: float, speed: float) -> bool:
        if not self._gripper_move_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn("Franka gripper move action is not available")
            return False
        goal = Move.Goal()
        goal.width = float(width)
        goal.speed = float(speed)
        return self._send_action_goal_and_wait(self._gripper_move_client, goal)

    def homing_gripper(self) -> bool:
        if not self._gripper_homing_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn("Franka gripper homing action is not available")
            return False
        return self._send_action_goal_and_wait(self._gripper_homing_client, Homing.Goal())

    def update_compliance_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mapped = self._map_serl_params(params)
        if not mapped:
            return {
                "success": True,
                "message": "No ROS 2 controller parameters mapped from request",
                "mapped": {},
            }
        if not self._set_parameters_client.wait_for_service(timeout_sec=0.5):
            return {
                "success": False,
                "message": "Controller set_parameters service is not available",
                "mapped": mapped,
            }

        req = SetParameters.Request()
        req.parameters = [self._make_parameter(name, value) for name, value in mapped.items()]
        future = self._set_parameters_client.call_async(req)
        if not self._wait_for_future(future, self.action_timeout):
            return {
                "success": False,
                "message": "Timed out while setting controller parameters",
                "mapped": mapped,
            }

        results = future.result().results
        success = all(result.successful for result in results)
        reason = "; ".join(result.reason for result in results if result.reason)
        return {
            "success": bool(success),
            "message": reason or "Updated ROS 2 controller parameters",
            "mapped": mapped,
        }

    def adjust_gripper_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            current_mass = float(self.gripper_load_mass)
            current_com = list(self.gripper_load_center_of_mass)

        if "mass" in params:
            target_mass = float(params["mass"])
        elif "m_load" in params:
            target_mass = float(params["m_load"])
        elif "gripper_load_mass" in params:
            target_mass = float(params["gripper_load_mass"])
        else:
            target_mass = current_mass

        if "mass_delta" in params:
            target_mass += float(params["mass_delta"])

        if not np.isfinite(target_mass) or abs(target_mass) > 3.0:
            return {
                "success": False,
                "message": "mass must be finite and within +/-3 kg",
                "current_mass": current_mass,
            }

        com = params.get(
            "center_of_mass",
            params.get(
                "com",
                params.get("F_x_Cload", params.get("gripper_load_center_of_mass", current_com)),
            ),
        )
        if not isinstance(com, (list, tuple)) or len(com) != 3:
            return {
                "success": False,
                "message": "center_of_mass/com must contain exactly 3 values",
                "current_center_of_mass": current_com,
            }
        target_com = [float(v) for v in com]

        result = self.update_compliance_params(
            {
                "gripper_load_mass": target_mass,
                "gripper_load_center_of_mass": target_com,
            }
        )
        if result["success"]:
            with self._lock:
                self.gripper_load_mass = target_mass
                self.gripper_load_center_of_mass = target_com
            result["gripper_load"] = {
                "mass": target_mass,
                "center_of_mass": target_com,
            }
        return result

    def _map_serl_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mapped: Dict[str, Any] = {}

        if "translational_stiffness" in params or "rotational_stiffness" in params:
            translational = float(params.get("translational_stiffness", 1000.0))
            rotational = float(params.get("rotational_stiffness", 200.0))
            mapped["stiffness"] = [translational] * 3 + [rotational] * 3

        if "translational_damping" in params or "rotational_damping" in params:
            translational = float(params.get("translational_damping", 45.0))
            rotational = float(params.get("rotational_damping", 10.0))
            mapped["damping"] = [translational] * 3 + [rotational] * 3

        if "translational_Ki" in params or "rotational_Ki" in params:
            translational = float(params.get("translational_Ki", 0.0))
            rotational = float(params.get("rotational_Ki", 0.0))
            mapped["integral"] = [translational] * 3 + [rotational] * 3

        for direct_name in (
            "stiffness",
            "damping",
            "integral",
            "k_gains",
            "d_gains",
            "gripper_load_mass",
            "gripper_load_center_of_mass",
        ):
            if direct_name in params:
                value = params[direct_name]
                mapped[direct_name] = list(value) if isinstance(value, (list, tuple)) else value

        return mapped

    def _make_parameter(self, name: str, value: Any) -> Parameter:
        parameter = Parameter()
        parameter.name = name
        parameter.value = ParameterValue()
        if isinstance(value, bool):
            parameter.value.type = ParameterType.PARAMETER_BOOL
            parameter.value.bool_value = value
        elif isinstance(value, int):
            parameter.value.type = ParameterType.PARAMETER_INTEGER
            parameter.value.integer_value = value
        elif isinstance(value, float):
            parameter.value.type = ParameterType.PARAMETER_DOUBLE
            parameter.value.double_value = value
        elif isinstance(value, str):
            parameter.value.type = ParameterType.PARAMETER_STRING
            parameter.value.string_value = value
        elif isinstance(value, (list, tuple, np.ndarray)):
            parameter.value.type = ParameterType.PARAMETER_DOUBLE_ARRAY
            parameter.value.double_array_value = [float(v) for v in value]
        else:
            raise ValueError(f"Unsupported parameter type for {name}: {type(value)!r}")
        return parameter

    def _send_action_goal_and_wait(self, client: ActionClient, goal: Any) -> bool:
        future = client.send_goal_async(goal)
        if not self._wait_for_future(future, self.action_timeout):
            return False
        goal_handle = future.result()
        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        if not self._wait_for_future(result_future, self.action_timeout):
            self.get_logger().warn("Action result timed out")
            return False
        result = result_future.result().result
        success = bool(getattr(result, "success", True))
        if not success:
            message = getattr(result, "message", "") or getattr(result, "error", "")
            self.get_logger().warn(f"Action failed: {message}" if message else "Action failed")
        return success

    def _wait_for_future(self, future: Any, timeout: float) -> bool:
        deadline = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < deadline:
            time.sleep(0.001)
        return bool(future.done())


def make_app(server: FrankaRos2Server) -> Flask:
    webapp = Flask(__name__)

    @webapp.route("/health", methods=["GET", "POST"])
    def health():
        return jsonify({"ok": True, **server.state_dict()})

    @webapp.route("/pose_debug", methods=["GET", "POST"])
    def pose_debug():
        return jsonify(server.pose_debug_dict())

    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        server.set_control_mode("idle")
        return "Started impedance"

    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        server.set_control_mode("zero_gravity")
        return "Stopped impedance"

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        pose = server.pose_from_service()
        if pose is None:
            pose = np.array(server.state_dict()["pose"], dtype=np.float64)
        xyz = pose[:3]
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([xyz, euler]).tolist()})

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        pose = server.pose_from_service()
        if pose is None:
            pose = np.array(server.state_dict()["pose"], dtype=np.float64)
        return jsonify({"pose": pose.tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": server.state_dict()["vel"]})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": server.state_dict()["force"]})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": server.state_dict()["torque"]})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": server.state_dict()["q"]})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": server.state_dict()["dq"]})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": server.state_dict()["jacobian"]})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": server.state_dict()["gripper_pos"]})

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        ok = server.joint_reset()
        return ("Reset Joint" if ok else "Joint reset failed", 200 if ok else 503)

    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        ok = server.homing_gripper()
        return ("Activated" if ok else "Activate failed", 200 if ok else 503)

    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        ok = server.homing_gripper()
        return ("Reset" if ok else "Reset failed", 200 if ok else 503)

    @webapp.route("/open_gripper", methods=["POST"])
    def open_gripper():
        ok = server.open_gripper()
        return ("Opened" if ok else "Open failed", 200 if ok else 503)

    @webapp.route("/close_gripper", methods=["POST"])
    def close_gripper():
        ok = server.close_gripper(slow=False)
        return ("Closed" if ok else "Close failed", 200 if ok else 503)

    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_gripper_slow():
        ok = server.close_gripper(slow=True)
        return ("Closed" if ok else "Close failed", 200 if ok else 503)

    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        pos = int(request.json["gripper_pos"])
        ok = server.move_gripper_position(pos)
        return ("Moved Gripper" if ok else "Move gripper failed", 200 if ok else 503)

    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        # franka_ros2 in this workspace does not expose an error recovery service.
        # Keeping this endpoint as a no-op preserves FrankaEnv compatibility.
        return "Clear"

    @webapp.route("/set_load", methods=["POST"])
    def set_load():
        result = server.adjust_gripper_load(request.json or {})
        status = 200 if result["success"] else 503
        return jsonify(result), status

    @webapp.route("/adjust_gripper_load", methods=["POST"])
    def adjust_gripper_load():
        result = server.adjust_gripper_load(request.json or {})
        status = 200 if result["success"] else 503
        return jsonify(result), status

    @webapp.route("/pose", methods=["POST"])
    def pose():
        arr = np.array(request.json["arr"], dtype=np.float64)
        ok = server.send_pose(arr)
        return ("Moved" if ok else "Move failed", 200 if ok else 503)

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify(server.state_dict())

    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        result = server.update_compliance_params(request.json or {})
        status = 200 if result["success"] else 503
        return jsonify(result), status

    return webapp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flask_url", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--controller_name", default="joint_impedance_example_controller")
    parser.add_argument("--state_topic", default="/franka_robot_state_broadcaster/robot_state")
    parser.add_argument("--jacobian_topic", default="/franka_jacobian")
    parser.add_argument("--gripper_joint_topic", default="/panda_gripper/joint_states")
    parser.add_argument("--pose_service", default="/get_pose")
    parser.add_argument("--control_mode_service", default="/set_control_mode")
    parser.add_argument("--trajectory_action", default="/set_trajectory")
    parser.add_argument("--equilibrium_pose_topic", default="/equilibrium_pose")
    parser.add_argument("--pose_command_mode", default="equilibrium", choices=["trajectory", "equilibrium"])
    parser.add_argument("--move_to_start_action", default="/move_to_start")
    parser.add_argument("--gripper_move_action", default="/panda_gripper/move")
    parser.add_argument("--gripper_grasp_action", default="/panda_gripper/grasp")
    parser.add_argument("--gripper_homing_action", default="/panda_gripper/homing")
    parser.add_argument("--pose_steps", type=int, default=4)
    parser.add_argument("--action_timeout", type=float, default=5.0)
    parser.add_argument("--default_gripper_open_width", type=float, default=0.075)
    parser.add_argument("--default_gripper_speed", type=float, default=0.3)
    parser.add_argument("--default_gripper_force", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pose_command_mode not in {"trajectory", "equilibrium"}:
        raise ValueError("--pose_command_mode must be either 'trajectory' or 'equilibrium'")
    rclpy.init()
    server = FrankaRos2Server(
        controller_name=args.controller_name,
        state_topic=args.state_topic,
        jacobian_topic=args.jacobian_topic,
        gripper_joint_topic=args.gripper_joint_topic,
        pose_service=args.pose_service,
        control_mode_service=args.control_mode_service,
        trajectory_action=args.trajectory_action,
        equilibrium_pose_topic=args.equilibrium_pose_topic,
        pose_command_mode=args.pose_command_mode,
        move_to_start_action=args.move_to_start_action,
        gripper_move_action=args.gripper_move_action,
        gripper_grasp_action=args.gripper_grasp_action,
        gripper_homing_action=args.gripper_homing_action,
        pose_steps=args.pose_steps,
        action_timeout=args.action_timeout,
        default_gripper_open_width=args.default_gripper_open_width,
        default_gripper_speed=args.default_gripper_speed,
        default_gripper_force=args.default_gripper_force,
    )

    spin_thread = threading.Thread(target=rclpy.spin, args=(server,), daemon=True)
    spin_thread.start()

    app = make_app(server)
    try:
        app.run(host=args.flask_url, port=args.port, threaded=True)
    finally:
        server.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
