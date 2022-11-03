import os
import math
import numpy as np
import open3d as o3d
import scipy.stats


def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)


def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
      Args:
        pose_path: (Complete) filename for the pose file
      Returns:
        A numpy array of size nx4x4 with n poses as 4x4 transformation
        matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']

    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')

    return np.array(poses)

def getDistance(x1,y1,x2,y2):
    distance= math.sqrt((x2-x1)**2+(y2-y1)**2)
    return distance

def getEulerAngles(R):
    x = math.atan2(R[2,1],R[2,2])
    y = math.atan2(-R[2,0], math.sqrt(np.multiply(R[2,1],R[2,1]) + np.multiply(R[2,2],R[2,2])))
    z = math.atan2(R[1, 0], R[0, 0])
    return x,y,z

def getRotation(R1, R2):
    [x1, y1, z1] = getEulerAngles(R1)
    [x2, y2, z2] = getEulerAngles(R2)
    theta = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    return theta

def load_labels(label_path):
  """ Load semantic and instance labels in SemanticKitti format.
  """
  label = np.fromfile(label_path, dtype=np.uint32)
  label = label.reshape((-1))

  sem_label = label & 0xFFFF  # semantic label in lower half
  inst_label = label >> 16  # instance id in upper half

  # sanity check
  assert ((sem_label + (inst_label << 16) == label).all())

  return sem_label, inst_label


def make_open3d_point_cloud(xyz, color=None, tile=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if tile:
            if len(color) != len(xyz):
                color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def downsample_point_cloud(xyz, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd_ds, ds_trace, ds_ids = pcd.voxel_down_sample_and_trace(
        voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
    inv_ids = [ids[0] for ids in ds_ids]

    return pcd_ds.points, inv_ids

# def find_neighbors_in_radius(xyz, voxel_size=0.05):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#     pcd_tree = o3d.geometry.KDTreeFlann(pcd)
#     [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], )
#     # 该函数search_knn_vector_3d返回点的k个最近邻居的索引列表
#
#     [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2
#     clean_xyz
#
#     return pcd_ds.points, inv_ids

def mutual_info(cell, cell_list):
    cell_mut_info = []

    # x = [0, 0.3, 0, 0.7]
    # y = [0, 0.4, 0, 0.6]
    # a = scipy.stats.entropy(x, y)
    range = np.max(cell) - np.min(cell)
    cell = (cell - np.min(cell)) / range
    # cell = cell / np.sum(cell)

    for cell_i in cell_list[1:]:

        range = np.max(cell_i) - np.min(cell_i)
        cell_i = (cell_i - np.min(cell_i)) / range
        # cell_i = cell_i / np.sum(cell_i)
        KL1 = scipy.stats.entropy(cell, cell_i)
        kl2 = scipy.stats.entropy(cell_i, cell)
        mut_info = 0.5*(KL1+kl2)

        # M = (cell + cell_i) / 2
        # mut_info = 0.5 * scipy.stats.entropy(cell, M) + 0.5 * scipy.stats.entropy(cell_i, M)
        cell_mut_info.append(mut_info)
    min_val = min(cell_mut_info)
    min_val_index = cell_mut_info.index(min_val)+1
    return min_val, min_val_index
