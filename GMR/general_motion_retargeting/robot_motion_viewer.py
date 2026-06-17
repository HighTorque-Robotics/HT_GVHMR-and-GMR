import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # visualization
                show_world_frame=False,
                show_body_lin_vel_frame=False,
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.show_world_frame = show_world_frame
        self.show_body_lin_vel_frame = show_body_lin_vel_frame
        self.record_video = record_video


        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False)      

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
        
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos
        
        mj.mj_forward(self.model, self.data)
        
        if follow_camera:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # 正面视角，轻微向下看
            # self.viewer.cam.azimuth = 180    # 正面朝向机器人
        
        if self.show_world_frame or self.show_body_lin_vel_frame or human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0

            # Draw world coordinate frame at origin
            if self.show_world_frame:
                draw_frame(
                    pos=np.array([0.0, 0.0, 0.0]),
                    mat=np.eye(3),
                    v=self.viewer,
                    size=1.0,  # 坐标轴长度
                    joint_name="World"
                )

            # Draw body linear velocity frame (yaw-aligned horizontal frame)
            if self.show_body_lin_vel_frame:
                # 从root四元数中提取yaw角度（绕Z轴的旋转）
                # root_rot 是 wxyz 格式的四元数
                root_quat_wxyz = root_rot
                rot = R.from_quat(root_quat_wxyz, scalar_first=True)

                # 转换为欧拉角（XYZ顺序）
                euler_xyz = rot.as_euler('xyz', degrees=False)
                yaw = euler_xyz[2]  # Z轴旋转角度

                # 创建只包含yaw旋转的旋转矩阵（roll=0, pitch=0）
                # 这样Z轴保持与世界坐标系Z轴平行，但XY平面跟随机器人yaw角度旋转
                body_lin_vel_rot = R.from_euler('z', yaw).as_matrix()

                # 在机器人root位置绘制这个坐标系
                draw_frame(
                    pos=root_pos,
                    mat=body_lin_vel_rot,
                    v=self.viewer,
                    size=0.8,  # 坐标轴长度
                    joint_name="BodyLinVel"
                )

            # Draw the task targets for reference
            if human_motion_data is not None:
                for human_body_name, (pos, rot) in human_motion_data.items():
                    draw_frame(
                        pos,
                        R.from_quat(rot, scalar_first=True).as_matrix(),
                        self.viewer,
                        human_point_scale,
                        pos_offset=human_pos_offset,
                        joint_name=human_body_name if show_human_body_name else None
                        )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
    
    def close(self):
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")

        # Close viewer
        try:
            self.viewer.close()
        except:
            pass

        # Force cleanup
        del self.viewer
        del self.data
        del self.model

        time.sleep(0.5)
