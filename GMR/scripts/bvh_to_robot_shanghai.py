import argparse
import pathlib
import time
import csv
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.shanghai_bvh import load_shanghai_bvh_file
from rich import print
from tqdm import tqdm
import os
import numpy as np

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01","pi_plus","hightorque_hi","pi_plus_waist","pi_plus_head","pi"],
        default="unitree_g1",
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
        "--show_frames",
        action="store_true",
        default=False,
        help="Show robot joint coordinate frames",
    )

    parser.add_argument(
        "--frame_size",
        type=float,
        default=0.08,
        help="Size of coordinate frame arrows",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Dump DoF name ordering and sample values to a sidecar debug file to verify mapping.",
    )
    
    
    args = parser.parse_args()
    

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load Shanghai BVH trajectory
    lafan1_data_frames, actual_human_height = load_shanghai_bvh_file(args.bvh_file)
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human="bvh",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    motion_fps = 120
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0

    while i < len(lafan1_data_frames):
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data, offset_to_ground=True)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        i += 1

        if args.save_path is not None:
            qpos_list.append(qpos)
    
    if args.save_path is not None:
        from scipy.spatial.transform import Rotation as R

        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])

        # 可选调试输出：记录原始DoF名称顺序以及数值统计，帮助定位某列恒为0的问题
        if args.debug:
            try:
                import mujoco as mj
                # 原始DoF名称（不重排）
                dof_names_in_order = []
                for vi in range(retargeter.model.nv):
                    jnid = retargeter.model.dof_jntid[vi]
                    nm = mj.mj_id2name(retargeter.model, mj.mjtObj.mjOBJ_JOINT, jnid)
                    dof_names_in_order.append(nm)
                debug_lines = []
                debug_lines.append("[DEBUG] Robot DoF order (nv={}):".format(retargeter.model.nv))
                for idx, nm in enumerate(dof_names_in_order):
                    debug_lines.append(f"  {idx:02d}: {nm}")
                # 数值统计（未重排）
                col_stats = []
                for k in range(dof_pos.shape[1]):
                    col = dof_pos[:, k]
                    col_stats.append((k, float(np.min(col)), float(np.max(col)), float(np.std(col))))
                debug_lines.append("[DEBUG] DoF column stats before reorder (index, min, max, std):")
                for k, mn, mx, sd in col_stats:
                    flag = " <- ZERO" if sd == 0.0 and mn == 0.0 and mx == 0.0 else ""
                    debug_lines.append(f"  {k:02d}: min={mn:.6f}, max={mx:.6f}, std={sd:.6f}{flag}")
                # 保存调试文件
                sidecar = args.save_path + ".debug.txt"
                with open(sidecar, "w") as df:
                    df.write("\n".join(debug_lines))
                print(f"[DEBUG] Wrote DoF order and stats to {sidecar}")
            except Exception as e:
                print(f"[DEBUG] Failed to dump debug info: {e}")

        # 重排序为 左腿→左臂→右腿→右臂 (和MuJoCo Actuator顺序一致)
        # retargeter返回的实际顺序: 0-5(左腿), 6-11(右腿), 12-16(左臂), 17-21(右臂)
        # 期望导出顺序: 0-5(左腿), 6-10(左臂), 11-16(右腿), 17-21(右臂)
        reorder_indices = [
            # 左腿 (保持原位)
            0, 1, 2, 3, 4, 5,           # l_hip_pitch → l_ankle_roll
            # 右腿 (从原12-16移到6-10)
            6, 7, 8, 9, 10, 11,         
            # 左臂 (从原6-11移到11-16)
            12, 13, 14, 15, 16, 
            # 右臂 (保持原位)
            17, 18, 19, 20, 21          # r_shoulder_pitch → r_wrist
        ]
        dof_pos = dof_pos[:, reorder_indices]

        # joint_names按导出顺序（左腿→左臂→右腿→右臂）
        joint_names = [
            'l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll',
            'r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll',
            'l_shoulder_pitch', 'l_shoulder_roll', 'l_upper_arm', 'l_elbow', 'l_wrist',
            'r_shoulder_pitch', 'r_shoulder_roll', 'r_upper_arm', 'r_elbow', 'r_wrist'
        ]

        # 若开启调试，输出重排后的前几帧按名称配对的值，便于快速核对列名与数据是否错位
        if args.debug:
            try:
                preview_n = min(3, dof_pos.shape[0])
                sidecar2 = args.save_path + ".reorder_preview.txt"
                with open(sidecar2, "w") as pf:
                    for fi in range(preview_n):
                        pf.write(f"Frame {fi}:\n")
                        for nm, val in zip(joint_names, dof_pos[fi].tolist()):
                            pf.write(f"  {nm}: {val:.9f}\n")
                print(f"[DEBUG] Wrote reordered preview to {sidecar2}")
            except Exception as e:
                print(f"[DEBUG] Failed to write reorder preview: {e}")

        # 计算速度 (通过数值微分)
        dt = 1.0 / motion_fps
        n_frames = len(qpos_list)

        # 基座线速度 (中心差分)
        root_lin_vel = np.zeros_like(root_pos)
        for i in range(n_frames):
            if i == 0:
                root_lin_vel[i] = (root_pos[i+1] - root_pos[i]) / dt
            elif i == n_frames - 1:
                root_lin_vel[i] = (root_pos[i] - root_pos[i-1]) / dt
            else:
                root_lin_vel[i] = (root_pos[i+1] - root_pos[i-1]) / (2 * dt)

        # 基座角速度 (通过四元数差分计算)
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

        # 关节速度 (中心差分)
        dof_vel = np.zeros_like(dof_pos)
        for i in range(n_frames):
            if i == 0:
                dof_vel[i] = (dof_pos[i+1] - dof_pos[i]) / dt
            elif i == n_frames - 1:
                dof_vel[i] = (dof_pos[i] - dof_pos[i-1]) / dt
            else:
                dof_vel[i] = (dof_pos[i+1] - dof_pos[i-1]) / (2 * dt)

        # 保存为CSV格式 (57维)
        with open(args.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            header = ['frame']
            # 0:3 基座位置
            header.extend([f'root_pos_{i}' for i in ['x', 'y', 'z']])
            # 3:7 基座四元数
            header.extend([f'root_rot_{i}' for i in ['x', 'y', 'z', 'w']])
            # 7:10 基座线速度
            # header.extend([f'root_lin_vel_{i}' for i in ['x', 'y', 'z']])
            # 10:13 基座角速度
            # header.extend([f'root_ang_vel_{i}' for i in ['x', 'y', 'z']])

            # 关节名称按重排序后的顺序
            joint_names = [
                # 左腿
                'l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll',
                # 右腿
                'r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll',
                # 左臂
                'l_shoulder_pitch', 'l_shoulder_roll', 'l_upper_arm', 'l_elbow', 'l_wrist',
                # 右臂
                'r_shoulder_pitch', 'r_shoulder_roll', 'r_upper_arm', 'r_elbow', 'r_wrist'
            ]
            # 13:35 关节角度位置
            header.extend([f'{name}_angle' for name in joint_names])
            # 35:57 关节角速度
            # header.extend([f'{name}_angvel' for name in joint_names])
            writer.writerow(header)

            # 按帧写入数据 (57维)
            for frame_idx in range(n_frames):
                row = [frame_idx]
                row.extend(root_pos[frame_idx].tolist())       # 0:3
                row.extend(root_rot[frame_idx].tolist())       # 3:7
                # row.extend(root_lin_vel[frame_idx].tolist())   # 7:10
                # row.extend(root_ang_vel[frame_idx].tolist())   # 10:13
                row.extend(dof_pos[frame_idx].tolist())        # 13:35
                # row.extend(dof_vel[frame_idx].tolist())        # 35:57
                writer.writerow(row)
        print(f"Saved to {args.save_path} (57 dimensions per frame)")

    # Close progress bar
    pbar.close()

    robot_motion_viewer.close()

    # Force exit to ensure MuJoCo viewer is properly closed
    import sys
    sys.exit(0)
