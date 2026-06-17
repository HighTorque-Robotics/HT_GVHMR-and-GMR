import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def load_shanghai_bvh_file(bvh_file):
    """
    Load Shanghai BVH data with skeleton structure:
    - ROOT: Hips
    - Legs: RightHip, RightKnee, RightAnkle, RightToe / LeftHip, LeftKnee, LeftAnkle, LeftToe
    - Spine: Chest, Chest2, Chest3, Chest4, Neck, Head
    - Arms: RightCollar, RightShoulder, RightElbow, RightWrist / LeftCollar, LeftShoulder, LeftElbow, LeftWrist

    Returns a dictionary with the following structure compatible with LAFAN1 format:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = (position, orientation)

        # Map Shanghai skeleton to LAFAN1-compatible naming
        # This allows reuse of existing IK configs

        # Hips (root)
        if "Hips" in result:
            result["Hips"] = result["Hips"]

        # Spine mapping
        if "Chest" in result:
            result["Spine"] = result["Chest"]
        if "Chest2" in result:
            result["Spine1"] = result["Chest2"]
        if "Chest3" in result:
            result["Spine2"] = result["Chest3"]
        if "Chest4" in result:
            result["Spine3"] = result["Chest4"]

        # Head and Neck
        if "Neck" in result:
            result["Neck"] = result["Neck"]
        if "Head" in result:
            result["Head"] = result["Head"]

        # Left Arm
        if "LeftCollar" in result:
            result["LeftShoulder"] = result["LeftCollar"]
        if "LeftShoulder" in result:
            result["LeftArm"] = result["LeftShoulder"]
        if "LeftElbow" in result:
            result["LeftForeArm"] = result["LeftElbow"]
        if "LeftWrist" in result:
            result["LeftHand"] = result["LeftWrist"]

        # Right Arm
        if "RightCollar" in result:
            result["RightShoulder"] = result["RightCollar"]
        if "RightShoulder" in result:
            result["RightArm"] = result["RightShoulder"]
        if "RightElbow" in result:
            result["RightForeArm"] = result["RightElbow"]
        if "RightWrist" in result:
            result["RightHand"] = result["RightWrist"]

        # Left Leg
        if "LeftHip" in result:
            result["LeftUpLeg"] = result["LeftHip"]
        if "LeftKnee" in result:
            result["LeftLeg"] = result["LeftKnee"]
        if "LeftAnkle" in result:
            result["LeftFoot"] = result["LeftAnkle"]
        if "LeftToe" in result:
            result["LeftToe"] = result["LeftToe"]

        # Right Leg
        if "RightHip" in result:
            result["RightUpLeg"] = result["RightHip"]
        if "RightKnee" in result:
            result["RightLeg"] = result["RightKnee"]
        if "RightAnkle" in result:
            result["RightFoot"] = result["RightAnkle"]
        if "RightToe" in result:
            result["RightToe"] = result["RightToe"]

        # Add modified foot pose (required by IK system)
        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftToe"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightToe"][1])

        frames.append(result)

    # Calculate human height from last frame
    human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # Override with typical height for better retargeting
    human_height = 1.75  # meters

    return frames, human_height
