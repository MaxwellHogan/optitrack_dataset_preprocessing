import pandas as pd
import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def make_transformation_matrix(rotation,translation):
    Transformation = np.hstack((rotation,translation.reshape(3,1)))
    Transformation = np.vstack((Transformation ,np.array([0, 0, 0, 1])))
    return Transformation

# Function to create point clouds
def create_point_cloud(points, color):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
    return point_cloud

###### LOAD DATA ###################################################################

# Load CSV data
csv_file = 'take_1_finished.csv'
data = pd.read_csv(csv_file)

# Load Point Cloud data
lidar_folder = f'take_1/point_clouds'
#Lidar_file_name = '13_03_53_420811.png'
#Lidar_file_name = '13_03_57_374339.png'
#Lidar_file_name = '13_04_06_374732.png' 
#Lidar_file_name = '13_04_19_575031.png' 
Lidar_file_name = '13_04_29_773257.png' 

lidar_file = os.path.join(lidar_folder, Lidar_file_name.replace('.png', '.pcd'))

pcd = o3d.io.read_point_cloud(lidar_file)

# Extract Lidar and Human paths
Lidar_path = data[['S_x', 'S_y', 'S_z']].values 
Human_path = data[['H_x', 'H_y', 'H_z']].values 

# Find the row corresponding to the given LiDAR file name
row = data[data['Closest Image'] == Lidar_file_name].iloc[0]

# Extract human position and orientation (quaternion) in the OptiTrack frame
human_position = np.array([row['H_x'], row['H_y'], row['H_z']])
human_orientation = np.array([row['H_qx'], row['H_qy'], row['H_qz'], row['H_qw']])

# Extract lidar position and orientation (quaternion) in the OptiTrack frame
lidar_position = np.array([row['S_x'], row['S_y'], row['S_z']])
lidar_orientation = np.array([row['S_qx'], row['S_qy'], row['S_qz'], row['S_qw'],])


###### MODIFY DATA ###############################################################

# Optitrack to lidar Rotation matrix
R_O2L = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]])

# Transform lidar and human positions to lidar frame
lidar_position_lidar_frame = R_O2L @ (lidar_position - lidar_position) 
human_position_lidar_frame = R_O2L @ (human_position - lidar_position)

lidar_rotation = R.from_quat(lidar_orientation).as_matrix()
human_rotation = R.from_quat(human_orientation).as_matrix()

lidar_transformation = make_transformation_matrix(lidar_rotation, lidar_position_lidar_frame)
human_transformation = make_transformation_matrix(human_rotation, human_position_lidar_frame)

lidar_to_human_transformation = np.matmul(np.linalg.inv(lidar_transformation), human_transformation)
lidar_to_human_translation = lidar_to_human_transformation[:3, 3]
lidar_to_human_rotation = lidar_to_human_transformation[:3, :3]

# Transform paths to lidar frame
Lidar_path_lidar_frame = (R_O2L @ (Lidar_path - lidar_position).T).T
Human_path_lidar_frame = (R_O2L @ (Human_path - lidar_position).T).T


###### ADD DATA TO VISUALISER #######################################################

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Draw the paths of the two objects
Lidar_point_cloud_object = create_point_cloud(Lidar_path_lidar_frame, [1, 0, 0]) # Red
Human_point_cloud_object = create_point_cloud(Human_path_lidar_frame, [0, 0, 1]) # Blue

print(Lidar_path_lidar_frame[0])

# Highlight the Start of the lidar path
Lidar_origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
Lidar_origin.translate(Lidar_path_lidar_frame[0])
Lidar_origin.paint_uniform_color([1, 0, 0])  # Red

# Highlight the current lidar positon on the path
Lidar_pos = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
Lidar_pos.translate(lidar_position_lidar_frame)
Lidar_pos.paint_uniform_color([1, 1, 0])  # Yellow

# Highlight the current human positon on the path
human_pos = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
human_pos.translate(human_position_lidar_frame)
human_pos.paint_uniform_color([0, 1, 1])  # Cyan

# Add the geometries to the visualizer
vis.add_geometry(Lidar_point_cloud_object)
vis.add_geometry(Human_point_cloud_object)
vis.add_geometry(Lidar_origin)
vis.add_geometry(Lidar_pos)
vis.add_geometry(human_pos)
vis.add_geometry(pcd)

# Run the visualizer
vis.run()
vis.destroy_window()
