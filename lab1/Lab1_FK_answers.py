import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    end_symbol = "End Site"
    bone_prefixes = ["ROOT", "JOINT", end_symbol]

    offset_symbol = "OFFSET"
    joint_dict = {}  # dict(indent, index)
    index = -1
    parent_name = ""
    with open(bvh_file_path) as f:
        for line in f.readlines():
            for prefix in bone_prefixes:
                if prefix in line:
                    if end_symbol in line:
                        name = parent_name + "_end"
                    else:
                        name = line[line.find(prefix) + len(prefix) + 1: line.find("\n")]
                    parent_name = name
                    joint_name.append(name)
                    if line.find(prefix)/4 == 0:
                        joint_parent.append(index)
                    else:
                        parent_index = joint_dict[int(line.find(prefix) / 4) - 1]
                        joint_parent.append(parent_index)
                    index += 1
                    joint_dict[int(line.find(prefix) / 4)] = index
            # if end_symbol in line:
            #     valid = False
            if offset_symbol in line:
                array_str = line[line.find(offset_symbol)+len(offset_symbol): line.find("\n")].split()
                offset = np.array([float(a) for a in array_str])
                joint_offset.append(offset)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    length = len(joint_name)
    joint_positions = np.zeros((length, 3), dtype=np.float64)
    joint_orientations = np.zeros((length, 4), dtype=np.float64)
    motion = motion_data[frame_id]
    motion = motion.reshape(21, -1)
    motion_index = 0
    for i in range(length):
        name = joint_name[i]
        is_end = name.endswith("_end")
        if joint_parent[i] == -1:
            joint_positions[i] = motion[motion_index]
            joint_orientations[i] = R.from_euler('XYZ', motion[motion_index + 1], degrees=True).as_quat()
            motion_index += 1
        else:
            parent_orientation = R(joint_orientations[joint_parent[i]])
            joint_positions[i] = joint_positions[joint_parent[i]] + parent_orientation.apply(joint_offset[i])
            if is_end:
                joint_orientations[i, 3] = 1.0
            else:
                joint_orientations[i] = (parent_orientation * R.from_euler('XYZ', motion[motion_index + 1], degrees=True)).as_quat()
                motion_index += 1
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    return motion_data
