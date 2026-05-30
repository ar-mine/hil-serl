# HIL-SERL ROS2 Franka 部署记录

## 当前架构

```
Franka 控制箱
  |
  | FCI，独立千兆网口
  v
NUC，从机，Ubuntu 22.04，CPU-only
  - ROS2 Humble
  - PREEMPT_RT 实时内核
  - libfranka / franka_ros2
  - 自定义 JointImpedanceExampleController
  - franka_server_ros2.py，监听 :5000

实验室 LAN
  |
  v
主 PC，Ubuntu 22.04，RTX5090
  - HIL-SERL actor / learner
  - JAX GPU 环境
  - 相机 / SpaceMouse，优先接在主 PC
  - SERVER_URL = "http://192.168.1.104:5000/"
```

核心原则：ROS2 只在 NUC 本地和 Franka 通信；HIL-SERL 上层训练代码运行在主 PC，并通过 HTTP 调用 NUC。这样保留原始 HIL-SERL 的 `FrankaEnv -> franka_server.py` 交互方式，只把 ROS1 server 层替换成 ROS2 server 层。

## NUC 当前状态

NUC 侧已经完成：

- Franka FCI 网络可达。
  - Franka 控制箱 IP：`192.168.2.103`
  - NUC Franka 独立网口：`192.168.2.1/24`
  - NUC 实验室 LAN IP：`192.168.1.104`
- ROS2 Humble workspace 可以构建，并能加载自定义 Franka 包。
- `joint_impedance_with_ik_controller.launch.py` 可以正常启动。
- 当前 active controllers：
  - `joint_state_broadcaster`
  - `franka_robot_state_broadcaster`
  - `joint_impedance_example_controller`
- 当前 ROS2 runtime 接口：
  - `/franka_robot_state_broadcaster/robot_state`
  - `/get_pose`
  - `/set_control_mode`
  - `/set_trajectory`
  - `/move_to_start`
  - `/franka_jacobian`
  - `/panda_gripper/move`
  - `/panda_gripper/grasp`
  - `/panda_gripper/homing`
- 已新增 ROS2 HTTP bridge：
  - `serl_robot_infra/robot_servers/franka_server_ros2.py`
- NUC 本机和主 PC 到 NUC 的 HTTP smoke test 已经完成。

## NUC 启动命令

启动 Franka ROS2 controller：

```bash
cd /home/armine/ros2_ws
source install/setup.bash
ros2 launch franka_bringup joint_impedance_with_ik_controller.launch.py robot_ip:=192.168.2.103 use_rviz:=false
```

另开一个终端，启动 HIL-SERL 兼容 HTTP server：

```bash
cd /home/armine/ros2_ws
source install/setup.bash
python3 src/hil-serl/serl_robot_infra/robot_servers/franka_server_ros2.py --flask_url 0.0.0.0 --port 5000
```

NUC 本机 smoke test：

```bash
curl -X POST http://127.0.0.1:5000/getstate
curl -X POST http://127.0.0.1:5000/getpos
curl -X POST http://127.0.0.1:5000/getpos_euler
curl -X POST http://127.0.0.1:5000/getjacobian
```

主 PC 到 NUC smoke test：

```bash
curl -X POST http://192.168.1.104:5000/getstate
```

## HTTP API 状态

`franka_server_ros2.py` 已实现：

- `/getstate`
- `/getpos`
- `/getpos_euler`
- `/getvel`
- `/getforce`
- `/gettorque`
- `/getq`
- `/getdq`
- `/getjacobian`
- `/get_gripper`
- `/pose`
- `/jointreset`
- `/open_gripper`
- `/close_gripper`
- `/close_gripper_slow`
- `/move_gripper`
- `/activate_gripper`
- `/reset_gripper`
- `/clearerr`
- `/set_load`
- `/update_param`
- `/health`

当前注意事项：

- `/getjacobian` 已由 `/franka_jacobian` 提供真实 Jacobian 数据。controller 重新编译并重启后生效。
- `/update_param` 会把 SERL 风格 compliance 参数映射成 ROS2 controller 参数，并在运行时更新 `stiffness`、`damping`、`integral`、`k_gains`、`d_gains` 和 `time_step`。
- `/clearerr` 目前是兼容用 no-op，因为当前 `franka_ros2` graph 没有暴露 error recovery service。
- `/set_load` 目前不支持，因为当前 active ROS2 graph 没有暴露对应服务。

## `/update_param` 实现说明

目标：让 HIL-SERL 里已有的调用保持不变：

```python
requests.post(SERVER_URL + "update_param", json=COMPLIANCE_PARAM)
```

并且实际更新正在运行的 Cartesian impedance 行为。

实现路径：

- HIL-SERL config 仍然发送 ROS1 dynamic-reconfigure 风格字段：
  - `translational_stiffness`
  - `translational_damping`
  - `rotational_stiffness`
  - `rotational_damping`
  - `translational_Ki`
  - `rotational_Ki`
  - translational / rotational clipping 相关字段
- `franka_server_ros2.py` 会把主要 stiffness、damping、Ki 字段映射为：
  - `stiffness: [tx, ty, tz, rx, ry, rz]`
  - `damping: [tx, ty, tz, rx, ry, rz]`
  - `integral: [tx, ty, tz, rx, ry, rz]`
- `JointImpedanceExampleController` 已新增 `on_set_parameters_callback`。
- controller 运行时接受这些参数更新：
  - `stiffness`
  - `damping`
  - `integral`
  - `k_gains`
  - `d_gains`
  - `time_step`
- 参数长度会在 controller 内校验。
- controller 内部变量更新由 `control_mutex_` 保护。

后续可选增强：

- 增加 clipping 支持。HIL-SERL 原 ROS1 controller 使用 clipping 限制 Cartesian target delta。这个比 stiffness/damping/Ki 更新更侵入，建议作为第二阶段处理。

## `/getjacobian` 实现说明

目标：通过 HTTP 返回真实 `6x7` zero Jacobian：

```bash
curl -X POST http://192.168.1.104:5000/getjacobian
```

实现路径：

- controller 内部已经调用：
  - `franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector)`
- controller 现在发布：
  - topic：`/franka_jacobian`
  - 类型：`std_msgs/msg/Float64MultiArray`
  - 数据长度：`42`
  - layout：libfranka / Eigen column-major layout
- `franka_server_ros2.py` 订阅 `/franka_jacobian`，按 `order="F"` reshape 为 `(6, 7)`。
- HTTP server 通过这些接口返回缓存的 Jacobian：
  - `/getjacobian`
  - `/getstate`

验证命令：

```bash
ros2 topic echo /franka_jacobian --once
curl -X POST http://127.0.0.1:5000/getjacobian
```

## 下一步工作分工

NUC 侧：

- 维持 Franka ROS2 launch 和 `franka_server_ros2.py` 稳定运行。
- 必要时继续补真实 `/clearerr` 和 `/set_load`，前提是 ROS2 hardware 层暴露合适接口。
- 不建议在 NUC 上部署 learner、JAX GPU 环境或完整 HIL-SERL 训练流程。

主 PC 侧：

- 安装 HIL-SERL 训练环境。
- 安装或配置 JAX GPU 环境。
- 把实验配置里的 `SERVER_URL` 改成：

```python
SERVER_URL = "http://192.168.1.104:5000/"
```

- 配置 RealSense 相机和 SpaceMouse。
- 先做 actor / env smoke test，不急着直接训练。
- 然后按 HIL-SERL 流程继续：
  - 调整 task config 和安全盒；
  - 采集 reward classifier 数据；
  - 训练 reward classifier；
  - 采集 demonstrations；
  - 启动 actor / learner；
  - 逐步恢复完整 HIL intervention 流程。
