# 统一重定向脚本使用说明

## 概述

本文档包含两个统一重定向脚本的使用说明：

1. **`bvh_to_robot_unified.py`**: BVH 文件到机器人重定向脚本
   - 输入：BVH 运动捕捉文件
   - 整合了三个项目（more、bydmmc、TK_AMP）的功能

2. **`gvhmr_to_robot_unified.py`**: GVHMR 视频到机器人重定向脚本
   - 输入：GVHMR 项目提取的 SMPLX 骨架数据（通过视频重建）
   - 整合了相同的三个项目（more、bydmmc、TK_AMP）的功能

## 项目配置对照表

| 项目 | 默认机器人 | 保存格式 | DoF | 关节顺序 |
|------|-----------|---------|-----|---------|
| **more** | pi_plus | CSV (57维) | 22 | 左腿→左臂→右腿→右臂 |
| **bydmmc** | pi_plus_waist | CSV (19/29/30/31维) | 12/22/23/24 | 见下方详细说明 |
| **TK_AMP** | pi_plus | PKL | 22/12 (去除wrist后20/12) | 见下方详细说明 |

### more 项目支持的机器人

| 机器人 | DoF | 输出维度 | 关节顺序 |
|--------|-----|---------|---------|
| **pi_plus** | 22 | 57维 | 左腿(6)→左臂(5)→右腿(6)→右臂(5) |

### bydmmc 项目支持的机器人

| 机器人 | DoF | 输出维度 | 关节顺序 |
|--------|-----|---------|---------|
| **pi** | 12 | 19维 | 左腿(6)→右腿(6) |
| **pi_plus** | 22 | 29维 | 左腿(6)→右腿(6)→左臂(5)→右臂(5) |
| **pi_plus_waist** | 23 | 30维 | 左腿(6)→右腿(6)→腰(1)→左臂(5)→右臂(5) |
| **pi_plus_head** | 24 | 31维 | 左腿(6)→右腿(6)→左臂(5)→右臂(5)→头(2) |

### TK_AMP 项目支持的机器人

| 机器人 | DoF | PKL输出 | 关节顺序 |
|--------|-----|---------|---------|
| **pi_plus** | 22 | 20 DoF (去除wrist) | 左腿(6)→左臂(4)→右腿(6)→右臂(4) |
| **pi** | 12 | 12 DoF | 左腿(6)→右腿(6) |

---

# 第一部分：BVH 重定向脚本 (bvh_to_robot_unified.py)

## 基本用法

### 1. more 项目
```bash
python bvh_to_robot_unified.py \
    --bvh_file /path/to/motion.bvh \
    --repo more \
    --save_path output/motion.csv
```

**输出格式**: CSV 57维
- 0:3 - root_pos (x, y, z)
- 3:7 - root_rot (x, y, z, w)
- 7:10 - root_lin_vel
- 10:13 - root_ang_vel
- 13:35 - 22个关节角度
- 35:57 - 22个关节角速度

### 2. bydmmc 项目

**使用 pi_plus_waist (默认)**:
```bash
python bvh_to_robot_unified.py \
    --bvh_file /path/to/motion.bvh \
    --repo bydmmc \
    --save_path output/motion.csv \
    --root waist_yaw_link  # 可选: 使用waist作为根四元数
```

**使用 pi**:
```bash
python bvh_to_robot_unified.py \
    --bvh_file /path/to/motion.bvh \
    --repo bydmmc \
    --robot pi \
    --save_path output/motion.csv
```

**使用 pi_plus**:
```bash
python bvh_to_robot_unified.py \
    --bvh_file /path/to/motion.bvh \
    --repo bydmmc \
    --robot pi_plus \
    --save_path output/motion.csv
```

**使用 pi_plus_head**:
```bash
python bvh_to_robot_unified.py \
    --bvh_file /path/to/motion.bvh \
    --repo bydmmc \
    --robot pi_plus_head \
    --save_path output/motion.csv
```

**输出格式**: CSV (维度根据机器人类型而定)
- pi: 19维 (3 + 4 + 12)
- pi_plus: 29维 (3 + 4 + 22)
- pi_plus_waist: 30维 (3 + 4 + 23)
- pi_plus_head: 31维 (3 + 4 + 24)

格式结构:
- 0:3 - root_pos (x, y, z)
- 3:7 - root_rot (x, y, z, w)
- 7:end - 关节角度（数量取决于机器人类型）

**特殊参数**:
- `--root base_link` (默认): 使用骨盆作为根四元数
- `--root waist_yaw_link`: 使用腰部作为根四元数（仅适用于 pi_plus_waist）

### 3. TK_AMP 项目

**使用 pi_plus (默认)**:
```bash
python bvh_to_robot_unified.py \
    --bvh_file /path/to/motion.bvh \
    --repo TK_AMP \
    --robot pi_plus \
    --save_path output/motion.pkl
```

**使用 pi**:
```bash
python bvh_to_robot_unified.py \
    --bvh_file /path/to/motion.bvh \
    --repo TK_AMP \
    --robot pi \
    --save_path output/motion.pkl
```

**输出格式**: PKL 字典
- pi_plus: 20 DoF (去除 l_wrist 和 r_wrist)
- pi: 12 DoF (只有腿部)

```python
{
    'fps': 30,
    'root_pos': ndarray (N, 3),
    'root_rot': ndarray (N, 4),  # xyzw格式
    'dof_pos': ndarray (N, 20 或 12),  # 根据机器人类型
    'local_body_pos': ndarray (N, num_bodies, 3),
    'link_body_list': list of body names
}
```

## 通用参数

### 必需参数
- `--bvh_file`: BVH 文件路径
- `--repo`: 项目类型 (more / bydmmc / TK_AMP)

### 可选参数
- `--robot`: 机器人类型（默认使用项目默认机器人）
- `--save_path`: 输出文件路径
- `--offset_to_ground`: 是否将机器人偏移到地面（默认 False）
- `--record_video`: 是否录制视频
- `--video_path`: 视频保存路径（默认 videos/example.mp4）
- `--rate_limit`: 是否限制渲染帧率
- `--show_world_frame`: 在可视化窗口中显示世界坐标系（原点位置）
- `--show_body_lin_vel_frame`: 在可视化窗口中显示body线速度参考坐标系（跟随机器人yaw角度）
- `--debug`: [仅more] 输出DoF调试信息
- `--root`: [仅bydmmc] 选择根四元数来源 (base_link / waist_yaw_link)

## 高级示例

### 使用不同机器人（TK_AMP项目支持）
```bash
python bvh_to_robot_unified.py \
    --bvh_file motion.bvh \
    --repo TK_AMP \
    --robot unitree_g1 \
    --save_path output/motion.pkl
```

### 启用地面偏移
```bash
python bvh_to_robot_unified.py \
    --bvh_file motion.bvh \
    --repo more \
    --offset_to_ground \
    --save_path output/motion.csv
```

### 录制视频 + 保存数据
```bash
python bvh_to_robot_unified.py \
    --bvh_file motion.bvh \
    --repo bydmmc \
    --record_video \
    --video_path videos/bydmmc_motion.mp4 \
    --save_path output/motion.csv
```

### 调试模式（more项目）
```bash
python bvh_to_robot_unified.py \
    --bvh_file motion.bvh \
    --repo more \
    --debug \
    --save_path output/motion.csv
```
会生成 `output/motion.csv.debug.txt` 文件，包含DoF顺序和统计信息。

### 显示坐标系参考框架
```bash
# 显示世界坐标系（位于原点）
python bvh_to_robot_unified.py \
    --bvh_file motion.bvh \
    --repo TK_AMP \
    --robot pi \
    --show_world_frame \
    --save_path output/motion.pkl

# 显示body_lin_vel_frame（跟随机器人yaw角度的水平坐标系）
python bvh_to_robot_unified.py \
    --bvh_file motion.bvh \
    --repo TK_AMP \
    --robot pi \
    --show_body_lin_vel_frame \
    --save_path output/motion.pkl

# 同时显示两个坐标系
python bvh_to_robot_unified.py \
    --bvh_file motion.bvh \
    --repo more \
    --show_world_frame \
    --show_body_lin_vel_frame \
    --save_path output/motion.csv
```

**坐标系说明**:
- **World坐标系**: 位于世界原点(0,0,0)，固定不动。红色=X轴，绿色=Y轴，蓝色=Z轴
- **BodyLinVel坐标系**: 位于机器人root位置，Z轴与世界Z轴平行，XY平面跟随机器人yaw角度旋转。用于表示机器人在水平面上的朝向

## 项目与机器人兼容性

### more 项目
✅ 支持:
- `pi_plus` (22 DoF) - 57维输出

### bydmmc 项目
✅ 支持:
- `pi` (12 DoF) - 19维输出
- `pi_plus` (22 DoF) - 29维输出
- `pi_plus_waist` (23 DoF) - 30维输出
- `pi_plus_head` (24 DoF) - 31维输出

### TK_AMP 项目
✅ 支持:
- `pi_plus` (22 DoF) - PKL 20 DoF输出
- `pi` (12 DoF) - PKL 12 DoF输出

## 原始脚本对照

| 原始脚本 | 新统一脚本参数 |
|---------|---------------|
| `bvh_to_robot_more.py` | `--repo more` |
| `bvh_to_robot_waist_bydmmc.py` | `--repo bydmmc` |
| `bvh_to_robot_TK_AMP.py` | `--repo TK_AMP` |

## 注意事项

1. **offset_to_ground**: 默认为 `False`，可通过参数启用
2. **数据加载器**: 所有项目统一使用 `load_lafan1_file` 加载 BVH 数据
3. **关节重排序**: 根据项目需求自动处理，more项目已修正关节重排序逻辑（原始脚本存在bug）
4. **TK_AMP CSV**: 已移除，仅保留 PKL 格式
5. **机器人验证**: 如果指定的机器人不在项目支持列表中，会显示警告但仍会继续执行
6. **速度坐标系转换**: more项目保存的 `root_lin_vel` 和 `root_ang_vel` 是相对于 `body_lin_vel_frame` 的速度，不是世界坐标系速度。这样机器人向前走时速度主要在body frame的X轴上，更符合控制直觉

## 常见问题

**Q: pi 机器人和 pi_plus 有什么区别？**
A:
- **pi** (12 DoF): 只有腿部关节（左腿6个 + 右腿6个），没有手臂
- **pi_plus** (22 DoF): 有腿部和手臂（左腿6个 + 左臂5个 + 右腿6个 + 右臂5个）

**Q: 为什么 more 项目的 pi_plus 输出是 57 维？**
A: 57 = 3 (root_pos) + 4 (root_rot) + 3 (root_lin_vel) + 3 (root_ang_vel) + 22 (joint angles) + 22 (joint velocities)

**Q: 为什么 bydmmc 项目的 pi 输出是 19 维？**
A: 19 = 3 (root_pos) + 4 (root_rot) + 12 (joint angles)。bydmmc 项目不包含速度信息。

**Q: bydmmc 项目为什么没有速度数据？**
A: 根据原始脚本，bydmmc 项目只保存位置和角度，不包含速度信息。

**Q: bydmmc 项目的四种机器人有什么区别？**
A:
- **pi** (12 DoF): 只有腿部，关节顺序：左腿→右腿
- **pi_plus** (22 DoF): 无腰部、无头部，关节顺序：左腿→右腿→左臂→右臂
- **pi_plus_waist** (23 DoF): 有腰部、无头部，关节顺序：左腿→右腿→腰→左臂→右臂
- **pi_plus_head** (24 DoF): 无腰部、有头部，关节顺序：左腿→右腿→左臂→右臂→头

**Q: pi_plus_head 的头部关节为什么在最后？**
A: 按照机器人设计，头部关节（head_pitch 和 head_yaw）位于关节序列的最后两位，这样可以更好地模拟人类头部的独立运动。

**Q: TK_AMP 的 PKL 文件中 pi_plus 为什么 dof_pos 是 20 维而不是 22 维？**
A: TK_AMP 项目移除了左右手腕关节（l_wrist 和 r_wrist），因此从 22 DoF 减少到 20 DoF。

**Q: TK_AMP 的 PKL 文件中 pi 机器人的 dof_pos 是多少维？**
A: pi 机器人只有 12 DoF（腿部），PKL 文件中 dof_pos 保持 12 维，不需要移除任何关节。

**Q: 什么是 body_lin_vel_frame？为什么要使用它？**
A: `body_lin_vel_frame` 是一个跟随机器人yaw角度的水平参考坐标系。它的特点是：
- Z轴与世界坐标系Z轴平行（始终竖直向上）
- XY平面跟随机器人绕Z轴的旋转（yaw角度）
- Roll和Pitch角度始终为0

使用它的好处是：当机器人向前走时，线速度主要集中在这个坐标系的某一个水平轴上，而不是分散在世界坐标系的各个轴上。这样速度数据更符合机器人控制的直觉，便于训练和控制。

**Q: more项目的速度数据是在哪个坐标系下的？**
A: more项目保存的 `root_lin_vel` 和 `root_ang_vel` 是相对于 `body_lin_vel_frame` 的速度，不是世界坐标系速度。转换公式为：
```
v_body = R_body_to_world.T @ v_world
```
其中 `R_body_to_world` 是只包含yaw旋转的旋转矩阵。

**Q: 如何在可视化中查看坐标系？**
A: 使用以下参数：
- `--show_world_frame`: 显示世界坐标系（原点位置）
- `--show_body_lin_vel_frame`: 显示body_lin_vel_frame（机器人root位置）

两个参数可以同时使用，方便对比观察。

**Q: more项目的关节重排序逻辑修正了什么？**
A: 原始的 `bvh_to_robot_more.py` 脚本中虽然注释说明要将GMR返回的 `左腿-右腿-左臂-右臂` 重排为 `左腿-左臂-右腿-右臂`，但实际的 `reorder_indices` 数组是 `[0,1,2,...,21]`，并没有真正执行重排序。统一脚本已修正为正确的重排序：从输入的 `[0-5(左腿), 6-11(右腿), 12-16(左臂), 17-21(右臂)]` 重排为 `[0-5(左腿), 12-16(左臂), 6-11(右腿), 17-21(右臂)]`。

---

# 第二部分：GVHMR 重定向脚本 (gvhmr_to_robot_unified.py)

## 概述

`gvhmr_to_robot_unified.py` 是用于 GVHMR 项目的统一重定向脚本。GVHMR 通过视频提取人体运动骨架数据（SMPLX格式），然后重定向到机器人。

**与 BVH 脚本的主要区别**:
- 输入格式：SMPLX 骨架数据（从视频重建）vs BVH 运动捕捉文件
- 数据加载：使用 `load_smplx_file` vs `load_lafan1_file`
- IK配置：使用 `smplx_to_*.json` vs `bvh_to_*.json`

**相同之处**:
- 支持相同的三个项目（more、bydmmc、TK_AMP）
- 支持相同的机器人类型（pi、pi_plus、pi_plus_waist、pi_plus_head）
- 输出格式完全相同
- 参数名称和用法几乎完全一致

## 基本用法

### 前置条件

在使用 `gvhmr_to_robot_unified.py` 之前，需要先使用 GVHMR 项目从视频中提取 SMPLX 骨架数据。GVHMR 会生成包含骨架序列的数据文件。

### 1. more 项目
```bash
python gvhmr_to_robot_unified.py \
    --motion_file /path/to/gvhmr_output.pkl \
    --repo more \
    --robot pi_plus \
    --save_path output/motion.csv
```

**输出格式**: CSV 57维
- 0:3 - root_pos (x, y, z)
- 3:7 - root_rot (x, y, z, w)
- 7:10 - root_lin_vel（body_lin_vel_frame坐标系）
- 10:13 - root_ang_vel（body_lin_vel_frame坐标系）
- 13:35 - 22个关节角度
- 35:57 - 22个关节角速度

### 2. bydmmc 项目

**使用 pi_plus_waist (默认)**:
```bash
python gvhmr_to_robot_unified.py \
    --motion_file /path/to/gvhmr_output.pkl \
    --repo bydmmc \
    --save_path output/motion.csv \
    --root waist_yaw_link  # 可选: 使用waist作为根四元数
```

**使用 pi**:
```bash
python gvhmr_to_robot_unified.py \
    --motion_file /path/to/gvhmr_output.pkl \
    --repo bydmmc \
    --robot pi \
    --save_path output/motion.csv
```

**使用 pi_plus**:
```bash
python gvhmr_to_robot_unified.py \
    --motion_file /path/to/gvhmr_output.pkl \
    --repo bydmmc \
    --robot pi_plus \
    --save_path output/motion.csv
```

**使用 pi_plus_head**:
```bash
python gvhmr_to_robot_unified.py \
    --motion_file /path/to/gvhmr_output.pkl \
    --repo bydmmc \
    --robot pi_plus_head \
    --save_path output/motion.csv
```

**输出格式**: CSV (维度根据机器人类型而定)
- pi: 19维 (3 + 4 + 12)
- pi_plus: 29维 (3 + 4 + 22)
- pi_plus_waist: 30维 (3 + 4 + 23)
- pi_plus_head: 31维 (3 + 4 + 24)

格式结构:
- 0:3 - root_pos (x, y, z)
- 3:7 - root_rot (x, y, z, w)
- 7:end - 关节角度（数量取决于机器人类型）

### 3. TK_AMP 项目

**使用 pi_plus (默认)**:
```bash
python gvhmr_to_robot_unified.py \
    --motion_file /path/to/gvhmr_output.pkl \
    --repo TK_AMP \
    --robot pi_plus \
    --save_path output/motion.pkl
```

**使用 pi**:
```bash
python gvhmr_to_robot_unified.py \
    --motion_file /path/to/gvhmr_output.pkl \
    --repo TK_AMP \
    --robot pi \
    --save_path output/motion.pkl
```

**输出格式**: PKL 字典
- pi_plus: 20 DoF (去除 l_wrist 和 r_wrist)
- pi: 12 DoF (只有腿部)

```python
{
    'fps': 30,
    'root_pos': ndarray (N, 3),
    'root_rot': ndarray (N, 4),  # xyzw格式
    'dof_pos': ndarray (N, 20 或 12),  # 根据机器人类型
    'local_body_pos': ndarray (N, num_bodies, 3),
    'link_body_list': list of body names
}
```

## 通用参数

### 必需参数
- `--motion_file`: GVHMR 生成的运动数据文件路径（SMPLX格式）
- `--repo`: 项目类型 (more / bydmmc / TK_AMP)

### 可选参数
- `--robot`: 机器人类型（默认使用项目默认机器人）
- `--save_path`: 输出文件路径
- `--offset_to_ground`: 是否将机器人偏移到地面（默认 False）
- `--record_video`: 是否录制视频
- `--video_path`: 视频保存路径（默认 videos/example.mp4）
- `--rate_limit`: 是否限制渲染帧率
- `--show_world_frame`: 在可视化窗口中显示世界坐标系（原点位置）
- `--show_body_lin_vel_frame`: 在可视化窗口中显示body线速度参考坐标系（跟随机器人yaw角度）
- `--debug`: [仅more] 输出DoF调试信息
- `--root`: [仅bydmmc] 选择根四元数来源 (base_link / waist_yaw_link)

## 高级示例

### 录制视频 + 保存数据
```bash
python gvhmr_to_robot_unified.py \
    --motion_file gvhmr_output.pkl \
    --repo more \
    --robot pi_plus \
    --record_video \
    --video_path videos/gvhmr_motion.mp4 \
    --save_path output/motion.csv
```

### 启用地面偏移
```bash
python gvhmr_to_robot_unified.py \
    --motion_file gvhmr_output.pkl \
    --repo bydmmc \
    --robot pi \
    --offset_to_ground \
    --save_path output/motion.csv
```

### 显示坐标系参考框架
```bash
# 显示世界坐标系和body_lin_vel_frame
python gvhmr_to_robot_unified.py \
    --motion_file gvhmr_output.pkl \
    --repo TK_AMP \
    --robot pi \
    --show_world_frame \
    --show_body_lin_vel_frame \
    --save_path output/motion.pkl
```

### 调试模式（more项目）
```bash
python gvhmr_to_robot_unified.py \
    --motion_file gvhmr_output.pkl \
    --repo more \
    --robot pi_plus \
    --debug \
    --save_path output/motion.csv
```
会生成 `output/motion.csv.debug.txt` 文件，包含DoF顺序和统计信息。

## 与 BVH 脚本的参数对照

| 参数 | bvh_to_robot_unified.py | gvhmr_to_robot_unified.py |
|------|------------------------|---------------------------|
| 输入文件参数 | `--bvh_file` | `--motion_file` |
| 项目类型 | `--repo` | `--repo` |
| 机器人类型 | `--robot` | `--robot` |
| 输出路径 | `--save_path` | `--save_path` |
| 地面偏移 | `--offset_to_ground` | `--offset_to_ground` |
| 视频录制 | `--record_video` | `--record_video` |
| 视频路径 | `--video_path` | `--video_path` |
| 帧率限制 | `--rate_limit` | `--rate_limit` |
| 显示世界坐标系 | `--show_world_frame` | `--show_world_frame` |
| 显示body坐标系 | `--show_body_lin_vel_frame` | `--show_body_lin_vel_frame` |
| 调试模式 | `--debug` | `--debug` |
| 根四元数选择 | `--root` | `--root` |

## 注意事项

1. **SMPLX 数据**: 确保输入的 motion_file 是 GVHMR 生成的有效 SMPLX 骨架数据
2. **IK 配置**: 脚本会自动使用 `smplx_to_*.json` 配置文件而非 `bvh_to_*.json`
3. **输出格式**: 与 BVH 脚本保持完全一致，方便混合使用两种数据源
4. **坐标系转换**: more项目同样使用 body_lin_vel_frame 进行速度转换
5. **关节重排序**: 与 BVH 脚本使用相同的重排序逻辑

## IK 配置文件

GVHMR 脚本使用以下 IK 配置文件：

| 机器人类型 | IK 配置文件 |
|-----------|------------|
| pi | `smplx_to_pi.json` |
| pi_plus | `smplx_to_pi_plus.json` |
| pi_plus_waist | `smplx_to_pi_plus_waist.json` |
| pi_plus_head | `smplx_to_pi_plus_head.json` |

这些配置文件与 BVH 对应的配置文件 (`bvh_to_*.json`) 功能相同，只是骨架映射关系不同（SMPLX骨架 vs BVH骨架）。

## 常见问题

**Q: GVHMR 脚本和 BVH 脚本的输出能混合使用吗？**
A: 可以。两个脚本在相同的 --repo 和 --robot 参数下输出格式完全一致，可以互换使用。

**Q: 如果我没有 GVHMR 数据，只有视频怎么办？**
A: 需要先运行 GVHMR 项目从视频中提取 SMPLX 骨架数据。GVHMR 是一个基于视频的人体运动重建项目，能够从单目视频中恢复3D人体运动。

**Q: GVHMR 脚本支持哪些机器人？**
A: 与 BVH 脚本支持相同的机器人：pi、pi_plus、pi_plus_waist、pi_plus_head。

**Q: GVHMR 脚本的速度数据也在 body_lin_vel_frame 坐标系下吗？**
A: 是的。more项目使用相同的速度坐标系转换逻辑。

**Q: 为什么需要两个不同的脚本？**
A: 因为输入数据格式不同：
- BVH 文件：传统的运动捕捉格式，精确但需要专业设备
- GVHMR/SMPLX：从视频重建的格式，更容易获取但可能有噪声

两个脚本的重定向逻辑相同，只是数据加载和骨架映射不同。
