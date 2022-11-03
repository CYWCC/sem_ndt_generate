import math

import open3d as o3d
import numpy as np
import os
import pandas as pd
import csv
from numpy.linalg import inv
import pcl
# import pclpy
# from pclpy import pcl


from utils import load_calib,load_poses,getDistance,getRotation,load_labels

path = 'D:/Study/datasets/kitti/sequences/00/'
semantic ='labels'
laser='velodyne' # lms_front 、lms_rear
laser_dir= path+laser+'/'
label_dir = path+semantic+'/'
pc_output_folder='pointcloud_cells2000_10m/'

#pc output forder
output_dir = path + pc_output_folder
isExists=os.path.exists(output_dir) #判断路径是否存在，存在则返回true
if not isExists:
    os.mkdir(output_dir)


# #Load extrinsics
# calib_file = path + 'calib.txt'
# T_cam_velo = load_calib(calib_file)
# T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
# T_velo_cam = np.linalg.inv(T_cam_velo)

# ## get poses
pose_file = path + 'poses.txt'
poses = np.array(load_poses(pose_file))
# inv_frame0 = np.linalg.inv(poses[0])
# new_poses = []
# for pose in poses:
#     new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
# global_poses = np.array(new_poses)

# Parameters
k = 1.025 # 过采样因子
target_cell_size = 2000
target_tolerance_size = target_cell_size * k
scan_num = len(poses)

# submap generation
submap_cover_distance = 10.0 # 每段submap覆盖轨迹的长度20.0
laser_reading_distance = 0.025
laser_reading_angle = 30
dist_start_next_frame = 5  # 两个submap间隔, 训练集l0m, 测试集20m

# Set up CSV file 建立CSV文件
csv_file_name = 'pointcloud_cells2000_locations10.csv '
fid_locations = path + csv_file_name
frame= pd.DataFrame({'scan_idx', 'northing', 'easting', 'down'})

# Counter Variables
frame_start = 0
frame_end = frame_start + 1
frames = []
i = frame_start
j = i
start_next_frame = frame_start
got_next = 0

while frame_end<scan_num:
    # A=global_poses[i][0,0]
    scan_dis = getDistance(global_poses[i][0,3], global_poses[i][1,3],global_poses[frame_start][0,3],global_poses[frame_start][1,3])
    while scan_dis<submap_cover_distance:
        if (j > (scan_num - 1)):
            break
        j = j + 1
        # a = global_poses[i][0:3,0:3]
        scan_dis = getDistance(global_poses[i][0, 3], global_poses[i][1, 3], global_poses[j][0, 3],
                               global_poses[j][1, 3])
        scan_rot = getRotation(global_poses[i][0:3,0:3], global_poses[j][0:3,0:3])* (180/math.pi)
        while (scan_dis< laser_reading_distance) and (scan_rot< laser_reading_angle):
            j = j + 1
            if (j > (scan_num - 1)):
                break

        frames.append(j)
        if (j > (scan_num - 1)):
            break

        submap_dis = getDistance(global_poses[frame_start,0,3], global_poses[frame_start,0,3],global_poses[j,0,3],global_poses[j,1,3])
        if ( submap_dis>dist_start_next_frame and got_next == 0):
            start_next_frame = len(frames) + 1
            got_next = 1
    i = j
    if (j > (scan_num - 1)):
        break
    frame_end = j

# Build submap
    pointcloud = []
    pointcloud_sem = []
    for i in frames:
        scan_path = laser_dir + str(i).zfill(6) + '.bin'
        scan1 = np.fromfile(scan_path, dtype=np.float32)
        scan = scan1.reshape((-1, 4))
        scan_xyz = scan[:,0:3]
        scan_xyz4=np.append(scan_xyz,np.ones((len(scan_xyz),1)),axis=1)
        scan_xyz_new = inv(global_poses[frame_start]).dot(global_poses[i]).dot(scan_xyz4.T).T# lidar局部坐标系

        # semantic labels #
        semantic_path =label_dir +str(i).zfill(6)+'.label'
        sem_label, inst_label = load_labels(semantic_path)

        scan_xyz_new[:,3] = sem_label
        # pointcloud.append(scan_xyzs[1:3,:])
        if i == frames[0]:
            pointcloud=scan_xyz_new[:,0:3]
            pointcloud_sem=sem_label
        else:
            pointcloud=np.vstack((pointcloud,scan_xyz_new[:,0:3]))
            pointcloud_sem=np.hstack((pointcloud_sem,sem_label))


    # Ground Plane Removal
    cloud = pcl.PointCloud(pointcloud) #　类型的转换
    filter_vox = cloud.make_voxel_grid_filter()   # 体素滤波器
    filter_vox.set_leaf_size(0.005, 0.005, 0.005) # 体素的大小，米
    cloud_filtered = filter_vox.filter() # 滤波，得到的数据类型为点云






