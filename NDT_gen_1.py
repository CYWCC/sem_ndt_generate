import open3d as o3d
import numpy as np
from utils import *
import tqdm

path = '/media/cyw/Backup Plus/datasets/SemanticKitti/sequences/00/'
semantic ='labels'
laser='velodyne' # lms_front 、lms_rear
laser_dir= path+laser+'/'
label_dir = path+semantic+'/'
pc_output_folder='NDT_cells_1104/'
label_output_folder = 'cells_labels_1104/'

#pc output forder
out_pc_dir = path + pc_output_folder
isExists=os.path.exists(out_pc_dir) #判断路径是否存在，存在则返回true
if not isExists:
    os.mkdir(out_pc_dir)

out_label_dir = path + label_output_folder
isExists=os.path.exists(out_label_dir)
if not isExists:
    os.mkdir(out_label_dir)

# ## get poses
pose_file = path + 'poses.txt'
poses = np.array(load_poses(pose_file))


# Parameters
k = 1.25 # 过采样因子
target_cell_size = 2048
target_tolerance_size = target_cell_size * k
scan_num = len(poses)

for i in tqdm.tqdm(range(scan_num)):
    scan_path = laser_dir + str(i).zfill(6) + '.bin'
    scan1 = np.fromfile(scan_path, dtype=np.float32)
    scan = scan1.reshape((-1, 4))
    scan_xyz = scan[:, 0:3]

    # scan_xyz4 = np.append(scan_xyz, np.ones((len(scan_xyz), 1)), axis=1)

    # semantic labels #
    semantic_path = label_dir + str(i).zfill(6) + '.label'
    sem_label, inst_label = load_labels(semantic_path)

    # Ground Plane Removal
    not_ground_mask = np.ones(len(scan_xyz), np.bool)
    raw_pcd = make_open3d_point_cloud(scan_xyz, color=None)
    _, inliers = raw_pcd.segment_plane(0.2, 3, 250)
    not_ground_mask[inliers] = 0
    clean_xyz = scan_xyz[not_ground_mask]
    clean_labels = sem_label[not_ground_mask]

    # dowsampling
    voxel_xyz = clean_xyz
    # vox_sz = 1.001

    # voxel size查找
    left = 0.5
    right = 2.5
    while left < right:
        mid = (left + right)/2
        voxel_xyz, inv_ids = downsample_point_cloud(clean_xyz, mid)
        if len(voxel_xyz) > target_tolerance_size+100:
            left = mid
        elif len(voxel_xyz) < target_tolerance_size-100:
            right = mid
        else:
            vox_sz = mid
            break

    # while len(voxel_xyz) > target_tolerance_size:
    #     voxel_xyz, inv_ids = downsample_point_cloud(clean_xyz, vox_sz)
    #     vox_sz += 0.01

    # NDT representation
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(clean_xyz)
    pcd_tree1 = o3d.geometry.KDTreeFlann(pcd1)
    voxel_indx = []

    r = 0.8
    for center_id in inv_ids:
        [k, idx, _] = pcd_tree1.search_radius_vector_3d(pcd1.points[center_id], r)
        if k>=3:
            voxel_indx.append(center_id)

    while len(voxel_indx) < target_cell_size:
        vox_sz -= 0.01
        voxel_xyz, inv_ids = downsample_point_cloud(clean_xyz, vox_sz)
        voxel_indx = []
        for center_id in inv_ids:
            [k, idx, _] = pcd_tree1.search_radius_vector_3d(pcd1.points[center_id], r)
            if k >= 3:
                voxel_indx.append(center_id)

    if len(voxel_indx) >= target_cell_size:
        NDT_cells_list = []
        cells_label_list = []
        for voxel_i in voxel_indx:
            [k, idx, _] = pcd_tree1.search_radius_vector_3d(pcd1.points[voxel_i], r)
            # ndt
            cell_points = clean_xyz[idx, :]
            cell_mean = np.mean(cell_points, axis=0)
            cell_var = (np.cov(cell_points.T)).flatten()
            cell_ndt = np.concatenate((cell_mean, cell_var), axis=0)
            NDT_cells_list.append(cell_ndt)
            # sem
            cell_labels = clean_labels[idx]
            cell_label = max(cell_labels, key=list(cell_labels).count)
            cells_label_list.append(cell_label)
        NDT_cells = np.array(NDT_cells_list)
        cells_label = np.array(cells_label_list)
    else:
        raise ValueError("The target points are not met")


        # mutual information
    if len(voxel_indx) > target_cell_size:
        out_num = len(voxel_indx) - target_cell_size
        pcd2 = o3d.geometry.PointCloud()
        NDT_cells_xyz = NDT_cells[:, :3]
        pcd2.points = o3d.utility.Vector3dVector(NDT_cells_xyz)
        pcd_tree2 = o3d.geometry.KDTreeFlann(pcd2)
        mut_info_list = []
        mut_info_index = []
        for j in range(len(NDT_cells_list)):
            [kj, idxj, _] = pcd_tree2.search_knn_vector_3d(pcd2.points[j], 7)
            neighbors = NDT_cells[idxj]
            min_val, min_val_index = mutual_info(NDT_cells_list[j], neighbors)
            mut_info_list.append(min_val)
            mut_info_index.append(idxj[min_val_index])
        mutualinfo = np.argsort(mut_info_list)
        delete_cells = mutualinfo[:out_num]
        target_NDT_cells = np.delete(NDT_cells, delete_cells, axis=0)
        target_cells_labels = np.delete(cells_label, delete_cells, axis=0)
    else:
        target_NDT_cells = NDT_cells
        target_cells_labels = cells_label

    # save data
    cell_save_path = out_pc_dir + str(i).zfill(6) + '.bin'
    label_save_path = out_label_dir + str(i).zfill(6) + '.label'
    target_NDT_cells = target_NDT_cells.astype('float32')
    target_NDT_cells.tofile(cell_save_path)
    target_cells_labels = target_cells_labels.astype('uint32')
    target_cells_labels.tofile(label_save_path)

    # 可视化
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(target_NDT_cells[:,:3])
    # o3d.visualization.draw_geometries([pcd1])



