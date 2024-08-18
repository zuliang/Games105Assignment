import numpy as np
from scipy.spatial.transform import Rotation as R
from task2_inverse_kinematics import MetaData

ZERO_R = R([0, 0, 0, 1])


def cal_2_vectors_rotation(v1, v2):
    # 计算两个向量的点积
    dot_product = np.dot(v1, v2)

    # 计算两个向量的模长
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    cos_angle = min(dot_product / (norm1 * norm2), 1.0)

    # 计算夹角（弧度）
    angle_rad = np.arccos(cos_angle)
    deg = np.degrees(angle_rad)
    # 计算轴角旋转向量，即向量a和向量b的叉积，然后除以2得到旋转轴，再乘以夹角
    axis = np.cross(v1 / norm1, v2 / norm2)
    axis = axis / np.linalg.norm(axis)
    if any(np.isnan(axis)):
        return ZERO_R
    rotation_axis = axis * angle_rad
    return R.from_rotvec(rotation_axis)


def cal_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
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
    motion = motion.reshape(length + 1, -1)
    for i in range(length):
        if joint_parent[i] == -1:
            joint_positions[i] = motion[i]
            joint_orientations[i] = R.from_euler('XYZ', motion[i + 1], degrees=True).as_quat()
        else:
            parent_orientation = R(joint_orientations[joint_parent[i]])
            joint_positions[i] = joint_positions[joint_parent[i]] + parent_orientation.apply(joint_offset[i])
            joint_orientations[i] = (parent_orientation * R.from_euler('XYZ', motion[i + 1], degrees=True)).as_quat()
    return joint_positions, joint_orientations


def cal_joint_offset(joint_parent, init_poses):
    joint_offset = np.zeros((len(init_poses), 3), dtype=np.float64)
    for i in range(len(init_poses)):
        p_id = joint_parent[i]
        if p_id == -1:
            joint_offset[i] = init_poses[i]
        else:
            joint_offset[i] = init_poses[i] - init_poses[p_id]
    return joint_offset


def part1_inverse_kinematics(meta_data: MetaData, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_offset = cal_joint_offset(joint_parent, meta_data.joint_initial_position)
    root_id = path2[-1]
    motion_data = [R.from_quat(q).as_euler("XYZ", degrees=True) for q in joint_orientations]
    motion_data.insert(0, joint_positions[root_id])
    motion_data = [np.concatenate(motion_data, axis=0)]
    motion = motion_data[0].reshape(26, -1)
    # using CCD from root to end
    end_pos = joint_positions[path[-1]]
    count = 0
    while not np.allclose(end_pos, target_pose, atol=0.01) and count <= 100:
        count += 1
        for i, j_id in enumerate(path[::-1]):
            parent_id = joint_parent[j_id]
            if parent_id == -1:
                continue
            if j_id in path1:
                # cal joint rotation
                p_joint_pos = joint_positions[parent_id]
                curr_vector = end_pos - p_joint_pos
                target_vector = target_pose - p_joint_pos
                joint_r = cal_2_vectors_rotation(curr_vector, target_vector)
                motion[parent_id + 1] = joint_r.as_euler("XYZ", degrees=True)
            else:
                if i <= 0:
                    continue
                child_pos = joint_positions[path[i-1]]
                curr_vector = end_pos - child_pos
                target_vector = target_pose - child_pos
                joint_r = cal_2_vectors_rotation(curr_vector, target_vector)
                motion[j_id + 1] = joint_r.inv().as_euler("XYZ", degrees=True)

            # cal end position
            joint_positions, joint_orientations = cal_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, 0)
            end_pos = joint_positions[path[-1]]
    print("try times:", count)
    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations