import argparse
import pathlib
import os
import time
import csv
import pickle
import torch

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print


# 项目配置映射
def compute_waist_quaternion(model, qpos, waist_joint_idx=18):
    """计算waist_yaw_link的世界姿态四元数"""
    data = mj.MjData(model)
    data.qpos[:] = qpos
    mj.mj_forward(model, data)
    waist_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'waist_yaw_link')
    waist_quat_wxyz = data.xquat[waist_body_id].copy()
    waist_quat_xyzw = waist_quat_wxyz[[1, 2, 3, 0]]
    return waist_quat_xyzw


REPO_CONFIGS = {
    "more": {
        "default_robot": "pi_plus",
        "supported_robots": ["pi_plus"],
        "save_format": "csv_57d",
        "joint_order": "left_leg_left_arm_right_leg_right_arm",
        "has_velocity": True,
    },
    "bydmmc": {
        "default_robot": "pi_plus_waist",
        "supported_robots": ["pi_plus", "pi_plus_waist", "pi_plus_head", "pi"],
        "save_format": "csv_multi",  # 19/29/30/31维
        "joint_order": "left_leg_right_leg_(waist@middle/left_arm_right_arm)_(head@end)",
        "has_velocity": False,
    },
    "TK_AMP": {
        "default_robot": "pi_plus",
        "supported_robots": ["unitree_g1", "unitree_g1_with_hands", "booster_t1",
                            "stanford_toddy", "fourier_n1", "engineai_pm01",
                            "pi_plus", "hightorque_hi", "pi"],
        "save_format": "pkl",
        "joint_order": "left_leg_(left_arm)_right_leg_(right_arm)",
        "has_velocity": False,
    },
}


def save_more_format(args, qpos_list, motion_fps):
    """保存 more 项目的 CSV 格式 (57维)"""
    from tqdm import tqdm

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])  # wxyz -> xyzw
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    # 重排序为 左腿→左臂→右腿→右臂
    reorder_indices = [
        0, 1, 2, 3, 4, 5,           # 左腿
        12, 13, 14, 15, 16,         # 左臂
        6, 7, 8, 9, 10, 11,         # 右腿
        17, 18, 19, 20, 21          # 右臂
    ]
    dof_pos = dof_pos[:, reorder_indices]

    joint_names = [
        'l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll',
        'l_shoulder_pitch', 'l_shoulder_roll', 'l_upper_arm', 'l_elbow', 'l_wrist',
        'r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll',
        'r_shoulder_pitch', 'r_shoulder_roll', 'r_upper_arm', 'r_elbow', 'r_wrist'
    ]

    # 计算速度
    dt = 1.0 / motion_fps
    n_frames = len(qpos_list)

    # 基座线速度
    root_lin_vel = np.zeros_like(root_pos)
    for i in range(n_frames):
        if i == 0:
            root_lin_vel[i] = (root_pos[i+1] - root_pos[i]) / dt
        elif i == n_frames - 1:
            root_lin_vel[i] = (root_pos[i] - root_pos[i-1]) / dt
        else:
            root_lin_vel[i] = (root_pos[i+1] - root_pos[i-1]) / (2 * dt)

    # 基座角速度
    root_ang_vel = np.zeros((n_frames, 3))
    for i in range(n_frames):
        if i == 0:
            q0 = R.from_quat(root_rot[i])
            q1 = R.from_quat(root_rot[i+1])
            dq = q1 * q0.inv()
            root_ang_vel[i] = dq.as_rotvec() / dt
        elif i == n_frames - 1:
            q0 = R.from_quat(root_rot[i-1])
            q1 = R.from_quat(root_rot[i])
            dq = q1 * q0.inv()
            root_ang_vel[i] = dq.as_rotvec() / dt
        else:
            q0 = R.from_quat(root_rot[i-1])
            q1 = R.from_quat(root_rot[i+1])
            dq = q1 * q0.inv()
            root_ang_vel[i] = dq.as_rotvec() / (2 * dt)

    # 将线速度和角速度从世界坐标系转换到body_lin_vel_frame
    root_lin_vel_body = np.zeros_like(root_lin_vel)
    root_ang_vel_body = np.zeros_like(root_ang_vel)
    for i in range(n_frames):
        rot_quat = R.from_quat(root_rot[i])
        euler_xyz = rot_quat.as_euler('xyz', degrees=False)
        yaw = euler_xyz[2]
        R_body_to_world = R.from_euler('z', yaw).as_matrix()
        root_lin_vel_body[i] = R_body_to_world.T @ root_lin_vel[i]
        root_ang_vel_body[i] = R_body_to_world.T @ root_ang_vel[i]

    root_lin_vel = root_lin_vel_body
    root_ang_vel = root_ang_vel_body

    # 关节速度
    dof_vel = np.zeros_like(dof_pos)
    for i in range(n_frames):
        if i == 0:
            dof_vel[i] = (dof_pos[i+1] - dof_pos[i]) / dt
        elif i == n_frames - 1:
            dof_vel[i] = (dof_pos[i] - dof_pos[i-1]) / dt
        else:
            dof_vel[i] = (dof_pos[i+1] - dof_pos[i-1]) / (2 * dt)

    # 保存CSV
    with open(args.save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['frame']
        header.extend([f'root_pos_{i}' for i in ['x', 'y', 'z']])
        header.extend([f'root_rot_{i}' for i in ['x', 'y', 'z', 'w']])
        header.extend([f'root_lin_vel_{i}' for i in ['x', 'y', 'z']])
        header.extend([f'root_ang_vel_{i}' for i in ['x', 'y', 'z']])
        header.extend([f'{name}_angle' for name in joint_names])
        header.extend([f'{name}_angvel' for name in joint_names])
        writer.writerow(header)

        for frame_idx in tqdm(range(n_frames), desc="Saving CSV"):
            row = [frame_idx]
            row.extend(root_pos[frame_idx].tolist())
            row.extend(root_rot[frame_idx].tolist())
            row.extend(root_lin_vel[frame_idx].tolist())
            row.extend(root_ang_vel[frame_idx].tolist())
            row.extend(dof_pos[frame_idx].tolist())
            row.extend(dof_vel[frame_idx].tolist())
            writer.writerow(row)

    print(f"Saved to {args.save_path} (57 dimensions per frame)")


def save_bydmmc_format(args, qpos_list, retargeter):
    """保存 bydmmc 项目的 CSV 格式"""
    from tqdm import tqdm

    root_pos = np.array([qpos[:3] for qpos in qpos_list])

    # 根据--root参数决定使用哪个link的四元数
    if hasattr(args, 'root') and args.root == "waist_yaw_link":
        print("计算waist_yaw_link的世界姿态四元数...")
        root_rot = np.array([compute_waist_quaternion(retargeter.model, qpos)
                            for qpos in tqdm(qpos_list, desc="Computing waist quaternions")])
    else:  # base_link
        print("使用base_link的四元数...")
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])

    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    robot_type = args.robot

    if robot_type == "pi":
        reorder_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        joint_names = [
            'l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll',
            'r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll'
        ]
        output_dims = 19
    elif robot_type == "pi_plus":
        reorder_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        joint_names = [
            'l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll',
            'r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll',
            'l_shoulder_pitch', 'l_shoulder_roll', 'l_upper_arm', 'l_elbow', 'l_wrist',
            'r_shoulder_pitch', 'r_shoulder_roll', 'r_upper_arm', 'r_elbow', 'r_wrist'
        ]
        output_dims = 29
    elif robot_type == "pi_plus_waist":
        reorder_indices = list(range(23))
        joint_names = [
            'l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll',
            'r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll',
            'waist_yaw',
            'l_shoulder_pitch', 'l_shoulder_roll', 'l_upper_arm', 'l_elbow', 'l_wrist',
            'r_shoulder_pitch', 'r_shoulder_roll', 'r_upper_arm', 'r_elbow', 'r_wrist'
        ]
        output_dims = 30
    elif robot_type == "pi_plus_head":
        reorder_indices = list(range(24))
        joint_names = [
            'l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll',
            'r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll',
            'l_shoulder_pitch', 'l_shoulder_roll', 'l_upper_arm', 'l_elbow', 'l_wrist',
            'r_shoulder_pitch', 'r_shoulder_roll', 'r_upper_arm', 'r_elbow', 'r_wrist',
            'head_pitch', 'head_yaw'
        ]
        output_dims = 31
    else:
        raise ValueError(f"Unsupported robot type for bydmmc format: {robot_type}")

    dof_pos = dof_pos[:, reorder_indices]

    # 保存CSV
    with open(args.save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['frame']
        header.extend([f'root pos {i}' for i in ['x', 'y', 'z']])
        header.extend([f'root rot {i}' for i in ['x', 'y', 'z', 'w']])
        header.extend(joint_names)
        writer.writerow(header)

        for frame_idx in tqdm(range(len(qpos_list)), desc="Saving CSV"):
            row = [frame_idx]
            row.extend(root_pos[frame_idx].tolist())
            row.extend(root_rot[frame_idx].tolist())
            row.extend(dof_pos[frame_idx].tolist())
            writer.writerow(row)

    print(f"Saved to {args.save_path} ({output_dims} dimensions per frame, robot: {robot_type})")


def save_tk_amp_format(args, qpos_list, retargeter, motion_fps):
    """保存 TK_AMP 项目的 PKL 格式"""
    print("Saving as pickle format...")

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    robot_type = args.robot

    if robot_type == "pi":
        dof_pos_final = dof_pos
    elif robot_type == "pi_plus":
        pkl_reorder_indices = [
            0, 1, 2, 3, 4, 5,           # 左腿
            11, 12, 13, 14, 15, 16,     # 左臂 (从原12-16移到6-11)
            6, 7, 8, 9, 10,             # 右腿 (从原6-11移到11-15)
            17, 18, 19, 20, 21          # 右臂
        ]
        dof_pos_reordered = dof_pos[:, pkl_reorder_indices]
        # 移除手腕: l_wrist (index 10) 和 r_wrist (index 21)
        wrist_indices_to_remove = [10, 21]
        all_indices = list(range(dof_pos_reordered.shape[1]))
        keep_indices = [i for i in all_indices if i not in wrist_indices_to_remove]
        dof_pos_final = dof_pos_reordered[:, keep_indices]
    else:
        # 其他机器人类型不需要特殊处理
        dof_pos_final = dof_pos

    # Compute forward kinematics
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kinematics_model = KinematicsModel(retargeter.xml_file, device=device)

    num_frames = root_pos.shape[0]
    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos, fk_root_rot,
        torch.from_numpy(dof_pos_final).to(device=device, dtype=torch.float)
    )
    body_names = kinematics_model.body_names

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(args.save_path, "wb") as f:
        pickle.dump(
            dict(
                fps=motion_fps,
                root_pos=root_pos,
                root_rot=root_rot,
                dof_pos=dof_pos_final,
                local_body_pos=local_body_pos.detach().cpu().numpy(),
                link_body_list=body_names,
            ),
            f,
        )

    print(f"Saved to {args.save_path} (PKL format, robot: {robot_type})")

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gvhmr_pred_file",
        help="GVHMR prediction file to load.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--repo",
        choices=["more", "bydmmc", "TK_AMP"],
        required=True,
        help="Target repository/project type",
    )

    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="Robot type (if not specified, will use repo default)",
    )

    parser.add_argument(
        "--offset_to_ground",
        action="store_true",
        default=False,
        help="Offset the robot to ground level",
    )

    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--show_world_frame",
        action="store_true",
        default=False,
        help="Display world coordinate frame at origin",
    )

    parser.add_argument(
        "--show_body_lin_vel_frame",
        action="store_true",
        default=False,
        help="Display body linear velocity frame (yaw-aligned, Z-axis parallel to world Z)",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Loop the motion.",
    )

    # bydmmc specific
    parser.add_argument(
        "--root",
        choices=["base_link", "waist_yaw_link"],
        default="base_link",
        help="[bydmmc only] Which link to use as the root quaternion.",
    )

    args = parser.parse_args()

    # 获取项目配置
    repo_config = REPO_CONFIGS[args.repo]

    # 确定机器人类型
    if args.robot is None:
        args.robot = repo_config["default_robot"]
        print(f"Using default robot for {args.repo}: {args.robot}")
    else:
        # 验证机器人是否被该项目支持
        if args.robot not in repo_config["supported_robots"]:
            print(f"[WARNING] Robot '{args.robot}' may not be fully supported by repo '{args.repo}'")
            print(f"Supported robots: {repo_config['supported_robots']}")


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # 创建保存目录
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    # Load SMPLX trajectory
    print(f"Loading GVHMR file...")
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, SMPLX_FOLDER
    )

    # Align fps
    tgt_fps = 50
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)

    # Initialize the retargeting system
    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    motion_fps = aligned_fps

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=0,
        show_world_frame=args.show_world_frame,
        show_body_lin_vel_frame=args.show_body_lin_vel_frame,
        record_video=args.record_video,
        video_path=args.video_path,
    )
    

    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    print(f"mocap_frame_rate: {motion_fps}")
    print(f"offset_to_ground: {args.offset_to_ground}")

    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets
        smplx_data = smplx_data_frames[i]

        # Retarget
        print(f"Processing frame {i+1}/{len(smplx_data_frames)}")
        qpos = retargeter.retarget(smplx_data, offset_to_ground=args.offset_to_ground)

        # Visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=True,
            rate_limit=args.rate_limit,
        )

        if args.save_path is not None:
            qpos_list.append(qpos)
            
    # 保存数据
    if args.save_path is not None:
        print(f"\nProcessing {len(qpos_list)} frames for saving...")

        # 根据项目类型调用相应的保存函数
        if args.repo == "more":
            save_more_format(args, qpos_list, motion_fps)
        elif args.repo == "bydmmc":
            save_bydmmc_format(args, qpos_list, retargeter)
        elif args.repo == "TK_AMP":
            save_tk_amp_format(args, qpos_list, retargeter, motion_fps)

    robot_motion_viewer.close()