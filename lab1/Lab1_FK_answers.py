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


def truncate_joint_array(joint_name, joint_parent, joint_offset):
    new_names = []
    new_offsets = []
    new_parent = []
    id_map = {}
    for i, name in enumerate(joint_name):
        id_map[i] = len(new_names)
        if name.endswith("_end"):
            continue
        new_names.append(name)
        new_offsets.append(joint_offset[i])
        if joint_parent[i] == -1:
            new_parent.append(joint_parent[i])
        else:
            new_parent.append(id_map[joint_parent[i]])
    return new_names, new_parent, new_offsets


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
    joint_name_t, joint_parent_t, joint_offset_t = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_a, joint_parent_a, joint_offset_a = part1_calculate_T_pose(A_pose_bvh_path)
    joint_name_t, joint_parent_t, joint_offset_t = truncate_joint_array(joint_name_t, joint_parent_t, joint_offset_t)
    joint_name_a, joint_parent_a, joint_offset_a = truncate_joint_array(joint_name_a, joint_parent_a, joint_offset_a)

    zero_r = R([0, 0, 0, 1])
    r_offsets = [zero_r for _ in range(len(joint_name_t))]
    index_map = {}
    for i, t_offset in enumerate(joint_offset_t):
        joint_name = joint_name_t[i]
        a_index = joint_name_a.index(joint_name)
        index_map[i] = a_index
        a_offset = joint_offset_a[a_index]

        # 计算两个向量的点积
        dot_product = np.dot(t_offset, a_offset)

        # 计算两个向量的模长
        norm_t = np.linalg.norm(t_offset)
        norm_a = np.linalg.norm(a_offset)

        cos_angle = min(dot_product / (norm_t * norm_a), 1.0)

        # 计算夹角（弧度）
        angle_rad = np.arccos(cos_angle)
        if np.isnan(angle_rad):
            continue

        # 计算轴角旋转向量，即向量a和向量b的叉积，然后除以2得到旋转轴，再乘以夹角
        axis = np.cross(t_offset / norm_t, a_offset / norm_a)
        axis = axis / np.linalg.norm(axis)
        if any(np.isnan(axis)):
            continue
        rotation_axis = axis * angle_rad
        parent_r = R.from_rotvec(rotation_axis)
        parent_id = joint_parent_t[i]
        if parent_id != -1:
            grand_parent_id = joint_parent_t[parent_id]
            if grand_parent_id != -1:
                parent_r = r_offsets[grand_parent_id].inv() * parent_r
            r_offsets[parent_id] = parent_r

    motion_data = load_motion_data(A_pose_bvh_path)
    temp = []

    for motion in motion_data:
        motion_channel = motion.reshape(21, -1)
        new_channel = motion_channel.copy()
        for i in range(len(joint_name_t)):
            t_index = i + 1
            a_index = index_map[i] + 1
            channel = motion_channel[a_index]
            r = R.from_euler("XYZ", channel, degrees=True)
            new_channel[t_index] = (r * r_offsets[t_index - 1]).as_euler("XYZ", degrees=True)
            # print("t name", joint_name_t[t_index-1], "a name:", joint_name_a[a_index - 1], "before:", channel, "after:", (r * r_offsets[t_index - 1]).as_euler("XYZ", degrees=True))
        temp.append(new_channel.reshape(1, -1))
    motion_data = np.concatenate(temp, axis=0)
    return motion_data
