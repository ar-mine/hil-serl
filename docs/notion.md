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

## 主 PC SpaceMouse 遥操作状态

目标：先不进入 actor / learner 训练流程，只验证主 PC 上的 SpaceMouse 可以通过 NUC 的 ROS2 HTTP bridge 低速、小范围控制 Franka。

已新增主 PC 侧遥操作脚本：

- `serl_robot_infra/franka_env/spacemouse/spacemouse_teleop.py`

脚本行为：

- 读取主 PC 上的 SpaceMouse。
- 通过 HTTP 调用 NUC：
  - `/getstate`
  - `/pose`
  - `/open_gripper`
  - `/close_gripper`
- 默认 `SERVER_URL`：

```bash
http://192.168.1.104:5000/
```

- 默认控制频率：`10Hz`
- 默认普通 HTTP 请求超时：`0.5s`
- 默认夹爪 HTTP 请求超时：`3.0s`
- 默认夹爪命令冷却时间：`1.5s`
- 默认当前位姿附近安全盒：
  - xyz：`+-0.03m`
  - rpy：`+-0.25rad`
- 默认 tool-frame 控制，和 HIL-SERL `RelativeFrame` 思路一致。
- 当前 SpaceMouse 平移 X 轴和 Z 轴在遥操作脚本中已反向，以匹配真机直觉控制方向。
- 默认只保留 SpaceMouse 原始输入 `tx/ty/tz/rx/ry/rz` 中绝对值最大的单个轴，便于逐轴测试和降低耦合。需要恢复多轴混合控制时使用 `--multi-axis`。
- 使用 `--debug` 可以实时打印 deadband、单轴过滤、X/Z 方向反向之后真正用于控制的 `tx/ty/tz/rx/ry/rz` 数值。
- 左键关闭夹爪，右键打开夹爪。
- 真机运动前需要输入 `yes` 二次确认。

已完成：

- 主 PC 到 NUC 的 HTTP bridge 可达。
- SpaceMouse 低速、小范围真机遥操作已经可以运行。
- 测试中发现 `/close_gripper` 可能超过 `0.5s` 才返回。遥操作脚本已将夹爪请求超时与普通 `/getstate`、`/pose` 请求超时分离，并把夹爪请求异常改成 warning 后继续运行，避免遥操作主循环直接退出。

### uv 环境注意事项

当前使用 `uv` 管理主 PC Python 环境。

曾遇到问题：

```bash
uv run python -m franka_env.spacemouse.spacemouse_test
```

报错：

```text
ModuleNotFoundError: No module named 'franka_env'
```

原因：

- `serl_robot_infra/franka_env/` 原先缺少顶层 `__init__.py`。
- `setup.py` 里的 `find_packages()` 因此没有把 `franka_env` 注册进 editable install。
- 当前 `.venv` 的 editable finder 只映射了 `robot_servers`，没有映射 `franka_env`。

已修复：

- 新增：

```text
serl_robot_infra/franka_env/__init__.py
```

- `serl_robot_infra/setup.py` 增加 `easyhid` 依赖，因为本地 `pyspacemouse.py` 实际 import 了 `easyhid`。

修复后需要重新安装 editable 包：

```bash
uv pip install -e ./serl_robot_infra --reinstall
```

验证：

```bash
uv run python -c "import franka_env; print(franka_env.__file__)"
uv run python -m franka_env.spacemouse.spacemouse_test
```

如果只想临时绕过安装，也可以：

```bash
PYTHONPATH=serl_robot_infra uv run python -m franka_env.spacemouse.spacemouse_test
```

### 遥操作运行命令

先 dry-run，只打印目标位姿，不发送 `/pose`：

```bash
uv run python -m franka_env.spacemouse.spacemouse_teleop \
  --server-url http://192.168.1.104:5000/ \
  --dry-run
```

低速小范围真机运行：

```bash
uv run python -m franka_env.spacemouse.spacemouse_teleop \
  --server-url http://192.168.1.104:5000/ \
  --xyz-margin 0.02 \
  --rpy-margin 0.15 \
  --pos-scale 0.0015 \
  --rot-scale 0.015 \
  --gripper-timeout 3.0 \
  --gripper-cooldown 1.5
```

### 提速建议

遥操作速度主要由这些参数控制：

- `--pos-scale`：每个控制周期最大平移增量，单位约为 m。
- `--rot-scale`：每个控制周期最大旋转增量，单位 rad。
- `--hz`：控制频率。
- `--deadband`：SpaceMouse 小输入死区。

最大平移速度可粗略估算为：

```text
max_speed ~= hz * pos_scale
```

例如：

- `10Hz * 0.0015m = 0.015m/s`
- `10Hz * 0.003m = 0.03m/s`
- `20Hz * 0.003m = 0.06m/s`

推荐提速顺序：

1. 先保持 `--hz 10`，只增大 `--pos-scale`。
2. 确认跟随稳定后，再增大 `--rot-scale`。
3. 最后再尝试把 `--hz` 提到 `20`。
4. 如果感觉慢是因为小动作不灵敏，先降低 `--deadband`，例如 `0.06` 或 `0.04`。

下一档推荐命令：

```bash
uv run python -m franka_env.spacemouse.spacemouse_teleop \
  --server-url http://192.168.1.104:5000/ \
  --hz 10 \
  --xyz-margin 0.03 \
  --rpy-margin 0.20 \
  --pos-scale 0.003 \
  --rot-scale 0.02 \
  --deadband 0.06 \
  --gripper-timeout 3.0 \
  --gripper-cooldown 1.5
```

如果稳定，再尝试：

```bash
uv run python -m franka_env.spacemouse.spacemouse_teleop \
  --server-url http://192.168.1.104:5000/ \
  --hz 20 \
  --xyz-margin 0.03 \
  --rpy-margin 0.20 \
  --pos-scale 0.003 \
  --rot-scale 0.02 \
  --deadband 0.06 \
  --gripper-timeout 3.0 \
  --gripper-cooldown 1.5
```

不建议一开始同时使用：

```bash
--hz 20 --pos-scale 0.005
```

这接近 `0.10m/s` 最大手控速度，对当前跨主 PC 到 NUC 的 HTTP 控制链路偏激进。

提速时先不要同时扩大：

- 速度；
- xyz 安全盒；
- rpy 姿态范围。

每次只放大一个维度，确认稳定后再继续。

### 待解决：平移时末端姿态旋转漂移

现象：

- SpaceMouse 设置为单轴控制后，控制 `X+` 平移时，真机末端仍观察到旋转。

PC 侧排查结果：

- 使用 `--debug` 打印遥操作脚本实际控制量后，确认 PC 侧没有发送旋转命令。
- 默认 tool-frame 控制下，`tx` 会因为末端姿态映射到 base frame，带出少量 `y/z` 位移分量，这是预期行为。
- 使用 `--base-frame` 后，PC 侧发送的目标已经是纯 base X 平移，姿态增量仍接近 0：

```text
control tx= 0.360 ty= 0.000 tz=-0.000 rx= 0.000 ry= 0.000 rz= 0.000 | dxyz=[0.0108 0.     0.    ] drpy=[0.00000000e+00 2.22044605e-16 0.00000000e+00]
```

结论：

- PC 侧 `spacemouse_teleop.py` 没有主动生成旋转目标。
- 如果真机仍有明显旋转，问题大概率在 NUC 侧 ROS2 bridge 或底层 controller。

NUC 侧建议检查：

- `franka_server_ros2.py` 收到 `/pose` 后是否原样转发目标 quaternion。
- quaternion 顺序是否始终为 `xyzw`，没有和 `wxyz` 混用。
- `/set_trajectory` 或 controller 内部是否真的使用了目标 orientation，而不是只使用 position。
- `JointImpedanceExampleController` 的姿态误差项、rotational stiffness、rotational damping 是否足够强。
- IK 是否允许位置运动时牺牲末端姿态，导致 orientation drift。
- `/update_param` 映射后的 rotational 参数是否过软。
- 末端负载、夹爪或线缆是否在较软姿态阻抗下造成被动旋转。

建议 NUC 侧增加日志：

```text
received target xyz
received target quat
current quat
target-current orientation error
actual quat after execution
```

判断方式：

- 如果 NUC 收到的 target quaternion 已经变化，检查 ROS2 bridge 的 pose 转换。
- 如果 NUC 收到的 target quaternion 不变，但实际 quaternion 漂移，检查 controller 姿态约束和 rotational impedance。

## 下一步工作分工

NUC 侧：

- 维持 Franka ROS2 launch 和 `franka_server_ros2.py` 稳定运行。
- 解决 SpaceMouse 单轴平移时末端姿态旋转漂移问题。当前 PC 侧已确认 `--base-frame` 下发送的是纯 base X 平移，`drpy≈0`。
- 必要时继续补真实 `/clearerr` 和 `/set_load`，前提是 ROS2 hardware 层暴露合适接口。
- 不建议在 NUC 上部署 learner、JAX GPU 环境或完整 HIL-SERL 训练流程。

主 PC 侧：

- SpaceMouse 遥操作已低速、小范围跑通。
- 继续完善 `uv` 管理的 HIL-SERL 训练环境。
- 安装或配置 JAX GPU 环境。
- 把实验配置里的 `SERVER_URL` 改成：

```python
SERVER_URL = "http://192.168.1.104:5000/"
```

- 配置 RealSense 相机。
- 新建真实任务配置目录，不直接修改示例 RAM/USB 任务。
- 先做 env smoke test，不急着直接训练：
  - reset；
  - get obs；
  - 小 action step；
  - 验证安全盒；
  - 验证相机图像；
  - 验证 SpaceMouse intervention。
- 然后按 HIL-SERL 流程继续：
  - 调整 task config 和安全盒；
  - 采集 reward classifier 数据；
  - 训练 reward classifier；
  - 采集 demonstrations；
  - 启动 actor / learner；
  - 逐步恢复完整 HIL intervention 流程。

## 当前下一步：主 PC env smoke test

当前理解确认：

```text
NUC:
  1. 启动 Franka ROS2 controller
  2. 启动 franka_server_ros2.py
  3. 发布 RealSense ROS2 image topic

主 PC:
  1. 启动 ROS2 image -> shared memory bridge
  2. 在 uv / HIL-SERL 环境中运行 env smoke test
```

这一步的目标不是训练，而是验证 HIL-SERL 的真实环境闭环：

```text
FrankaEnv
  -> HTTP getstate / pose
  -> NUC franka_server_ros2.py
  -> Franka ROS2 controller

FrankaEnv
  -> shared_memory camera backend
  -> 主 PC ros2_image_to_shm.py
  -> NUC RealSense topic
```

### NUC 端启动

终端 1，启动 Franka ROS2 controller：

```bash
cd /home/armine/ros2_ws
source install/setup.bash
ros2 launch franka_bringup joint_impedance_with_ik_controller.launch.py robot_ip:=192.168.2.103 use_rviz:=false
```

终端 2，启动 HIL-SERL 兼容 HTTP server：

```bash
cd /home/armine/ros2_ws
source install/setup.bash
python3 src/hil-serl/serl_robot_infra/robot_servers/franka_server_ros2.py --flask_url 0.0.0.0 --port 5000
```

NUC 端或主 PC 端都可以先验证 HTTP：

```bash
curl -X POST http://192.168.1.104:5000/health
curl -X POST http://192.168.1.104:5000/getstate
curl -X POST http://192.168.1.104:5000/getpos_euler
```

### 主 PC 端启动图像 bridge

终端 1，ROS2 环境中启动图像 bridge：

```bash
cd /mnt/FAST/hil-serl
source /opt/ros/humble/setup.bash
export ROS_LOG_DIR=/tmp/ros_logs

python3 scripts/ros2_image_to_shm.py \
  --topic /camera/camera/color/image_raw/compressed \
  --name wrist_1 \
  --compressed \
  --width 128 \
  --height 128
```

可选单帧测试：

```bash
python3 scripts/ros2_image_to_shm.py \
  --topic /camera/camera/color/image_raw/compressed \
  --name wrist_1 \
  --compressed \
  --once
```

### 主 PC 端运行 smoke test

终端 2，uv / HIL-SERL 环境中运行。此终端不要 source ROS2：

```bash
cd /mnt/FAST/hil-serl
uv run python <smoke_test_script>.py
```

smoke test 应该只做低风险动作：

- 构造 task config；
- `SERVER_URL = "http://192.168.1.104:5000/"`；
- camera backend 使用 `shared_memory`；
- 安全盒围绕当前 pose 设置得很小；
- `ACTION_SCALE` 设置得很小；
- `env.reset()`；
- 打印 observation keys / state shape / image shape；
- 保存或显示一张 observation image；
- 执行 3-5 次 zero action 或极小 action；
- 确认机械臂没有异常运动。

推荐初始参数：

```python
ACTION_SCALE = (0.001, 0.005, 1)
MAX_EPISODE_LENGTH = 10
REALSENSE_CAMERAS = {
    "wrist_1": {
        "backend": "shared_memory",
        "shape": (128, 128, 3),
        "shm_name": "hilserl_wrist_1",
        "stale_after": 1.0,
    },
}
```

### smoke test 通过标准

- `env.reset()` 不报错；
- `/getstate`、`/pose` 调用稳定；
- observation 中包含 `state` 和 `wrist_1` 图像；
- `wrist_1` shape 为 `(1, 128, 128, 3)` 或 wrapper 后对应的 batch/chunk shape；
- 图像不是全黑或冻结；
- zero action / 小 action 不触发异常运动；
- ESC / 终止逻辑可用。

通过后，再进入真实任务配置、reward classifier 数据采集和 demo 采集阶段。

### smoke test 结果记录

主 PC 侧已新增低风险 smoke 脚本：

```text
scripts/franka_env_smoke.py
```

运行命令：

```bash
uv run python scripts/franka_env_smoke.py \
  --server-url http://192.168.1.104:5000/ \
  --steps 3 \
  --hz 5 \
  --xyz-margin 0.01 0.01 0.01 \
  --rpy-margin 0.05 0.05 0.05 \
  --output-image /tmp/hilserl_smoke_wrist_1.png
```

已验证通过：

- 主 PC 能访问 NUC 的 `/health`、`/getstate`、`/getpos_euler`。
- `/health` 返回 `ok: true`，包含 pose、q、dq、force、torque、jacobian、gripper_pos。
- ROS2 image bridge 从 `/camera/camera/color/image_raw/compressed` 写入 shared memory。
- HIL-SERL shared memory camera backend 能读取 `wrist_1` 图像：

```text
shape=(128, 128, 3), dtype=uint8, min=0, max=187, mean≈122
```

- `FrankaEnv` 成功初始化并执行 `env.reset()`。
- observation 包含：
  - `state/tcp_pose`
  - `state/tcp_vel`
  - `state/gripper_pose`
  - `state/tcp_force`
  - `state/tcp_torque`
  - `images/wrist_1`
- `images/wrist_1` 保存到：

```text
/tmp/hilserl_smoke_wrist_1.png
```

- 机械臂完成 3 次 zero action step：

```text
step=0 reward=0 done=False truncated=False succeed=False
step=1 reward=0 done=False truncated=False succeed=False
step=2 reward=0 done=False truncated=False succeed=False
```

结论：`FrankaEnv -> NUC HTTP bridge -> Franka ROS2 controller` 和 `FrankaEnv -> shared_memory -> ROS2 RealSense topic` 两条链路已经闭环。下一步可以进入真实任务配置目录创建，而不是继续只测底层接口。

## 主 PC 接收 NUC RealSense 图像

当前 RealSense 接在 NUC 上，因此主 PC 不再使用本机 `pyrealsense2` 直接取图。采用两进程结构：

```text
主 PC ROS2 shell:
  scripts/ros2_image_to_shm.py
    订阅 NUC 发布的 ROS2 image topic
    crop / resize 到 HIL-SERL 需要的大小
    写入主 PC 本机 shared memory

主 PC uv / HIL-SERL shell:
  FrankaEnv
    使用 shared_memory camera backend
    从 shared memory 读取最新图像
```

这样 HIL-SERL 的 uv/JAX 环境不需要 import `rclpy` / `cv_bridge`，ROS2 依赖只存在于独立图像接收进程里。

### 启动图像接收进程

在主 PC 的 ROS2 环境中运行。示例：

```bash
cd /mnt/FAST/hil-serl
source /opt/ros/humble/setup.bash

python3 scripts/ros2_image_to_shm.py \
  --topic /wrist_1/color/image_raw/compressed \
  --name wrist_1 \
  --compressed \
  --width 128 \
  --height 128
```

如果订阅的是 raw `sensor_msgs/msg/Image`，去掉 `--compressed`：

```bash
python3 scripts/ros2_image_to_shm.py \
  --topic /wrist_1/color/image_raw \
  --name wrist_1 \
  --width 128 \
  --height 128
```

可选 crop 格式为 `y1:y2,x1:x2`：

```bash
python3 scripts/ros2_image_to_shm.py \
  --topic /wrist_1/color/image_raw/compressed \
  --name wrist_1 \
  --compressed \
  --crop 120:620,300:900 \
  --width 128 \
  --height 128
```

多个相机时，每个 topic 启一个进程：

```bash
python3 scripts/ros2_image_to_shm.py --topic /wrist_1/color/image_raw/compressed --name wrist_1 --compressed
python3 scripts/ros2_image_to_shm.py --topic /wrist_2/color/image_raw/compressed --name wrist_2 --compressed
```

共享内存默认名为：

```text
hilserl_<camera_name>
```

例如 `wrist_1` 对应 `hilserl_wrist_1`。

默认写入共享内存的 timestamp 使用主 PC 本机接收时间，避免 NUC 和主 PC 时钟未同步时触发 stale frame 误判。如果已经配置了 PTP / chrony 并希望保留 ROS header stamp，可追加：

```bash
--use-message-stamp
```

### HIL-SERL task config 示例

在主 PC 的 HIL-SERL 实验配置里，把相机 backend 改成 shared memory：

```python
class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://192.168.1.104:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "backend": "shared_memory",
            "shape": (128, 128, 3),
            "shm_name": "hilserl_wrist_1",
            "stale_after": 1.0,
        },
    }
```

如果 `ros2_image_to_shm.py` 已经在主 PC 上完成 crop / resize，那么 `IMAGE_CROP` 可以不再为该 camera 配置 crop。若需要保留 HIL-SERL 原有 crop 逻辑，也可以让 shared memory 写入更大尺寸图像，再由 `FrankaEnv.get_im()` 继续 crop / resize。

### 验证步骤

1. 主 PC 确认能看到 NUC 的 ROS2 topic：

```bash
source /opt/ros/humble/setup.bash
ros2 topic list | grep image
```

2. 启动 shared memory bridge：

```bash
python3 scripts/ros2_image_to_shm.py --topic <image_topic> --name wrist_1 --compressed
```

单帧连通性测试可以加 `--once`：

```bash
python3 scripts/ros2_image_to_shm.py --topic /camera/camera/color/image_raw/compressed --name wrist_1 --compressed --once
```

3. 在 HIL-SERL uv 环境中启动 env smoke test，确认 `env.reset()` 能读到图像。

4. 若出现 stale frame，优先检查：

- NUC 是否仍在发布 topic；
- 主 PC ROS2 domain / network 是否能发现 NUC；
- bridge 进程是否仍在写帧；
- `shm_name` 是否和 config 一致。

## 传统 peg-in-hole spiral search

当前 HTTP bridge 已经具备实现传统 peg-in-hole 搜索策略的核心接口：

- `/pose`：发送末端目标位姿；
- `/getstate`：读取当前 pose、force、torque、joint、Jacobian；
- `/update_param`：切换/调整 compliance 参数。

已新增主 PC 侧脚本：

```text
scripts/spiral_peg_in_hole.py
```

流程：

```text
接近 approximate hole xy，默认保持当前 z
  -> 竖直向下
  -> 通过持续 force 阈值检测接触平面
  -> 纯 xy spiral search，默认不继续向下
  -> force jump 触发 armed 状态
  -> armed 后检测 Fz 小于 release 阈值，判断插入
  -> 默认停止；只有显式 --continue-insert 时继续小步下插
```

当前接触平面判据：

```text
Fz >= 2N 持续 1s
```

当前 spiral 阶段插入判据：

```text
1. insert_metric 相邻两次变化量 >= 3N 后，进入 jump_armed=True
2. jump_armed 状态会保留
3. jump_armed 后，insert_metric < 2N 时认为 peg 已进入 hole
```

默认使用：

```text
insert_metric = positive Fz
--insertion-force-jump 3.0
--insertion-release-force 2.0
```

默认 spiral search 是纯 XY 螺旋，Z 固定在接触检测到的位置。只有显式添加 `--spiral-down` 时，才会启用 spiral 阶段的向下 bias。

默认检测到 insertion 后会停止，不执行最终下插。只有显式添加 `--continue-insert` 时，才进入 final insertion 阶段。

`--continue-insert` 的 final insertion 触底判据：

```text
前 2mm 为 force deadband，不计算触底
超过 2mm 后，Fz >= 2N 持续 0.5s 判定触底并停止
```

默认是 dry-run：会读取 `/getstate`，但不会发送 `/pose`。示例：

```bash
python3 scripts/spiral_peg_in_hole.py \
  --server-url http://192.168.1.104:5000/ \
  --hole-x 0.50 \
  --hole-y 0.00 \
  --approach-z 0.25
```

真机执行需要显式加 `--execute`，且默认需要二次输入 `yes`：

```bash
python3 scripts/spiral_peg_in_hole.py \
  --server-url http://192.168.1.104:5000/ \
  --hole-x 0.43 \
  --hole-y -0.02 \
  --contact-force 2.0 \
  --contact-hold-time 1.0 \
  --force-axis z \
  --force-sign positive \
  --max-force 15.0 \
  --descent-step 0.0004 \
  --period 0.5 \
  --spiral-pitch 0.001 \
  --spiral-theta-step 0.25 \
  --max-radius 0.006 \
  --insertion-force-jump 3.0 \
  --insertion-release-force 2.0 \
  --insertion-force-axis z \
  --insertion-force-sign positive \
  --execute
```

如果需要检测到 insertion 后继续下插，添加：

```bash
--continue-insert \
--insert-depth 0.01 \
--insert-step 0.0004 \
--insert-force-deadband 0.002 \
--insert-contact-force 2.0 \
--insert-contact-hold-time 0.5
```

初次真机建议：

- 使用很小的 `--max-radius`，例如 `0.003`；
- 默认不要使用 `--continue-insert`；需要 final insertion 时，先使用很小的 `--insert-depth`，例如 `0.003`；
- 降低 `--max-force`；
- 当前测试确认接触平面可用 `Fz >= 2N 持续 1s`；
- 先不改变姿态，默认沿用当前末端 quaternion；如需固定竖直姿态，用 `--rpy roll pitch yaw`。
- peg 已经夹紧时，绝对 z 高度参考性较弱；建议省略 `--approach-z`，让脚本只对齐 x/y 后再按力反馈下探。
- 已验证的慢速下探默认值为 `--descent-step 0.0004 --period 0.5`。

当前参数组合：
```bash
python3 -u scripts/spiral_peg_in_hole.py   --server-url http://192.168.1.104:5000/   --hole-x 0.45   --hole-y -0.02   --cart-step 0.001   --max-descent 0.32   --contact-force 2.0   --contact-hold-time 0.5   --force-axis z   --force-sign positive   --max-force 15.0   --spiral-pitch 0.01   --spiral-theta-step 0.2   --max-radius 6   --search-down-bias 0.002   --search-down-per-rev 0.0003   --insert-depth 0.1   --execute  --descent-step 0.0010 --period 0.1 --insertion-force-jump 1.5 --continue-insert --insert-force-deadband 0.005 --insertion-release-force 3.0
```
