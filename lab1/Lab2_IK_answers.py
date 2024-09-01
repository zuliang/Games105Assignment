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
    # 计算轴角旋转向量，即向量a和向量b的叉积，然后除以2得到旋转轴，再乘以夹角
    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)
    if any(np.isnan(axis)):
        return ZERO_R
    rotation_axis = axis * angle_rad
    return R.from_rotvec(rotation_axis)


# def cal_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data):
#     """请填写以下内容
#     输入: part1 获得的关节名字，父节点列表，偏移量列表
#         motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
#         frame_id: int，需要返回的帧的索引
#     输出:
#         joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
#         joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
#     Tips:
#         1. joint_orientations的四元数顺序为(x, y, z, w)
#         2. from_euler时注意使用大写的XYZ
#     """
#     length = len(joint_name)
#     joint_positions = np.zeros((length, 3), dtype=np.float64)
#     joint_orientations = np.zeros((length, 4), dtype=np.float64)
#     for i in range(length):
#         if joint_parent[i] == -1:
#             joint_positions[i] = motion_data[i]
#             joint_orientations[i] = motion_data[i + 1].as_quat()
#         else:
#             parent_orientation = R(joint_orientations[joint_parent[i]])
#             joint_positions[i] = joint_positions[joint_parent[i]] + parent_orientation.apply(joint_offset[i])
#             joint_orientations[i] = (parent_orientation * motion_data[i + 1]).as_quat()
#     return joint_positions, joint_orientations


def cal_joint_offset(joint_parent, init_poses):
    joint_offset = np.zeros((len(init_poses), 3), dtype=np.float64)
    for i in range(len(init_poses)):
        p_id = joint_parent[i]
        if p_id == -1:
            joint_offset[i] = init_poses[i]
        else:
            joint_offset[i] = init_poses[i] - init_poses[p_id]
    return joint_offset


def update_chain_ori_pos(chain_orientations, chain_poses, chain_rotations, chain_offset, index):
    # 计算并更新所有子骨骼的朝向和位置
    for i in range(index, len(chain_poses)):
        chain_orientations[i] = chain_orientations[i - 1] * chain_rotations[i]
        chain_poses[i] = chain_poses[i - 1] + chain_orientations[i - 1].apply(chain_offset[i])


t_pose = np.array(np.zeros(3))


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
    def get_joint_rotations():
        joint_rotations = [ZERO_R for _ in joint_name]
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.])
            else:
                joint_rotations[i] = (R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i]))
        return joint_rotations

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    if len(path2) == 1 and path2[0] != 0:
        path2 = []
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_offset = cal_joint_offset(joint_parent, meta_data.joint_initial_position)
    # 每个bone的局部旋转
    joint_rotations = get_joint_rotations()
    root_id = joint_parent.index(-1)

    # init chain
    chain_poses = [joint_positions[j_id] for j_id in path]
    chain_rotations = [ZERO_R for _ in path]
    chain_orientations = [R.from_quat(joint_orientations[j_id]) for j_id in path]
    chain_offset = np.empty((len(path), 3))

    if len(path2) > 1:
        chain_orientations[0] = R.from_quat(joint_orientations[path2[1]]).inv()
    else:
        chain_orientations[0] = R.from_quat(joint_orientations[path[0]])

    for i, j_id in enumerate(path):
        if i == 0:
            continue
        if j_id in path2:
            chain_orientations[i] = R.from_quat(joint_orientations[path[i + 1]])
            chain_offset[i] = -joint_offset[path[i - 1]]
            # 骨骼链的局部旋转
            chain_rotations[i] = joint_rotations[j_id].inv()
        else:
            chain_offset[i] = joint_offset[j_id]
            # 骨骼链的局部旋转
            chain_rotations[i] = joint_rotations[j_id]

    # using CCD from root to end
    end_pos = joint_positions[path[-1]]
    path_len = len(path)
    count = 0
    while not np.allclose(end_pos, target_pose, atol=0.01) and count <= 100:
        for i in range(path_len - 2, -1, -1):
            if joint_parent[path[i]] == -1:
                continue
            # cal joint rotation
            p_joint_pos = chain_poses[i]
            curr_vector = end_pos - p_joint_pos
            target_vector = target_pose - p_joint_pos
            delta_r = cal_2_vectors_rotation(curr_vector, target_vector)
            chain_orientations[i] = delta_r * chain_orientations[i]
            chain_rotations[i] = chain_orientations[i - 1].inv() * chain_orientations[i]
            update_chain_ori_pos(chain_orientations, chain_poses, chain_rotations, chain_offset, i + 1)
            end_pos = chain_poses[-1]
        count += 1

    # 把计算之后的IK写回joint_rotation
    for i, j_id in enumerate(path):
        joint_positions[j_id] = chain_poses[i]
        if j_id in path1:
            joint_rotations[j_id] = chain_rotations[i]
        elif len(path2) > 1:
            joint_rotations[j_id] = chain_rotations[i].inv()

    if len(path2) == 0:
        joint_rotations[path[0]] = R.from_quat(joint_orientations[joint_parent[path[0]]]).inv() * chain_orientations[0]

    if root_id in path:
        path_root_id = path.index(root_id)
        if path_root_id != 0:
            joint_positions[root_id] = chain_poses[path_root_id]
            joint_orientations[root_id] = chain_orientations[path_root_id].as_quat()

    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * joint_rotations[i]).as_quat()
        joint_positions[i] = joint_positions[p] + R.from_quat(joint_orientations[p]).apply(joint_offset[i])
    # joint_positions, joint_orientations = cal_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data)

    print("try times:", count)
    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    target_pose = np.array([relative_x + joint_positions[0][0], target_height, relative_z + joint_positions[0][2]])
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose)
    # lcl_joint_rots = [ZERO_R for _ in path]
    # joint_offset = cal_joint_offset(joint_parents, meta_data.joint_initial_position)
    #
    # path_len = len(path)
    # end_pose = joint_positions[path[-1]]
    # count = 0
    # while not np.allclose(end_pose, tar_pose, atol=0.01) and count <= 10:
    #     for i in range(path_len - 1, 0, -1):
    #         if joint_parents[path[i]] == -1:
    #             continue
    #         # cal joint rotation
    #         curr_index = path[i]
    #         c_joint_pos = joint_positions[curr_index]
    #         curr_vector = end_pose - c_joint_pos
    #         target_vector = tar_pose - c_joint_pos
    #         delta_r = cal_2_vectors_rotation(curr_vector, target_vector)
    #         joint_orientations[curr_index] = (delta_r * R.from_quat(joint_orientations[curr_index])).as_quat()
    #         lcl_joint_rots[i] = R.from_quat(joint_orientations[joint_parents[curr_index]]).inv() * R.from_quat(joint_orientations[curr_index])
    #         # 计算并更新所有子骨骼的朝向和位置
    #         b_id = curr_index
    #         while b_id in joint_parents:
    #             # get child bone index
    #             b_id = joint_parents.index(b_id)
    #             lcl_rot = ZERO_R if b_id not in path else lcl_joint_rots[path.index(b_id)]
    #             joint_orientations[b_id] = (R.from_quat(joint_orientations[joint_parents[b_id]]) * lcl_rot).as_quat()
    #             joint_positions[b_id] = joint_positions[joint_parents[b_id]] + R(joint_orientations[joint_parents[b_id]]).apply(joint_offset[b_id])
    #
    #         end_pose = joint_positions[path[-1]]
    return joint_positions, joint_orientations


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations