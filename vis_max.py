import pandas as pd
import open3d as o3d
import numpy as np
import os
import time 
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 

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

def dict2transMat(in_dict : dict) -> np.ndarray:
    # Extract human position and orientation (quaternion) in the OptiTrack frame
    position = np.array([in_dict['loc_x'], in_dict['loc_y'], in_dict['loc_z']])

    ### check xyzw is the correct convention  
    orientation = np.array([in_dict['rot_x'], in_dict['rot_y'], in_dict['rot_z'], in_dict['rot_w']])
    rotation = R.from_quat(orientation).as_matrix()

    # construct transformation mat
    transMat = np.identity(4, dtype=rotation.dtype)
    transMat[:3,:3] = rotation
    transMat[:3, 3] = position

    return transMat

def row2transMat(row : pd.Series) -> list[np.ndarray, np.ndarray]:

    human_data = {key : row[value] for key, value in zip(["loc_x","loc_y","loc_z", "rot_w", "rot_x", "rot_y", "rot_z"], ["H_x", "H_y","H_z", "H_qw","H_qx", "H_qy", "H_qz"])}
    # print(human_data)
    human_transformation = dict2transMat(human_data)

    lidar_data = {key : row[value] for key, value in zip(["loc_x","loc_y","loc_z", "rot_w", "rot_x", "rot_y", "rot_z"], ["S_x", "S_y","S_z", "S_qw","S_qx", "S_qy", "S_qz"])}
    lidar_transformation = dict2transMat(lidar_data)
    
    return human_transformation, lidar_transformation

def transMat2Dict(transMat : np.ndarray) -> dict:
    rotMat = transMat[:3,:3]
    orientation = R.from_matrix(rotMat).as_quat()

    xyz = transMat[:3, 3]

    out_dict = {key : value for key,value in zip(["loc_x","loc_y","loc_z"], xyz)}

    out_dict = out_dict | {key : value for key,value in zip(["rot_w", "rot_x", "rot_y", "rot_z"], orientation)}

    return out_dict


def getMakersFromTracking(fn : str) -> list[np.ndarray, int]:
    '''
    This function will return an array with the tranformation mats for each marker
    belonging to the rigid body relative to the rigid body's pivot point.

    Note many variables are described as 'human', however, this function will work
    with any rigid body as long as: 
        * rigid body params appear first, in the column order.
        * the tracking info was exported with header info enabled.
        * the tracking info was exported with quartonians - is that how you spell that?
        * there is no missing information - I'm not sure what has to be missing for it to fail.
        * If it doesn't work you are likely smarter than the person writing this so you'll figure it out.

    args:
        fn : str (path to optitrack file with rigid body and marker info)

    return:
        human_makers : np array (as described above)
        marker_count : int (number of markers)

    '''

    ## these lines open a file with the unmoving rigid body and the marker location 
    human_description = pd.read_csv(fn, skiprows=list(range(6))).dropna()
    human_description.drop(human_description.columns[list(range(2))], axis=1, inplace=True)
    human_description = human_description.mean()

    ## get the rotation and loc of the rigid body and convert to lidar coordinate system 
    ## going to lidar you swap y and z and multiply x by -1... 
    human_piv_rot = human_description[["X", "Z", "Y","W"]].to_numpy()
    human_piv_rot[1] = -1*human_piv_rot[0]
    human_piv_rot = R.from_quat(human_piv_rot).as_matrix()

    human_piv_loc = human_description[["X.1", "Z.1", "Y.1"]].to_numpy()
    human_piv_loc[0] = -1*human_piv_loc[0]

    marker_count = int(list(human_description.keys())[-1].split(".")[-1])

    human_makers = np.stack((np.identity(4, dtype=human_piv_rot.dtype), )*(marker_count-1))

    for idx, marker_i in enumerate(range(2, marker_count+1)):
        # print("On Marker", marker_i)
        ## copy rotation matrix - same for all makers 
        # human_makers[idx, :3,:3] = human_piv_rot
        
        ## grab the maker loc and convert to lidar coord system 
        marker_i_loc = human_description[["X.%i"%marker_i, "Z.%i"%marker_i, "Y.%i"%marker_i]].to_numpy()
        marker_i_loc[0] = -1*marker_i_loc[0]

        ## calc maker location relative to rigid body pivot point 
        human_makers[idx, :3, 3] = marker_i_loc - human_piv_loc

    return human_makers, marker_count

human_makers, marker_count = getMakersFromTracking("human_description.csv")

# print(human_makers)
# kill
###### LOAD DATA ###################################################################

set_name = "take_2"

## path to lidar folder 
lidar_folder = os.path.join(set_name,f'point_clouds')

# Load CSV data
csv_file = '{}_filtered.csv'.format(set_name)
data = pd.read_csv(csv_file).dropna()

## this is the output dataframe 
data_out = pd.DataFrame(columns=["timestamp","point_cloud_fn", "img_fn", "loc_x","loc_y","loc_z","rot_w", "rot_x", "rot_y", "rot_z"])

## GOTO lidar coordinate system 
## swap loc y and z 
data[["S_z","S_y", "H_z","H_y"]] = data[["S_y","S_z", "H_y","H_z"]]
## swap rot y and z 
data[["S_qz","S_qy", "H_qz","H_qy"]] = data[["S_qy","S_qz", "H_qy","H_qz"]]

## flip x 
data['S_x'] = data['S_x'].apply(lambda x: x*-1)
data['H_x'] = data['H_x'].apply(lambda x: x*-1)
# data['S_qx'] = data['S_qx'].apply(lambda x: x*-1)
# data['H_qx'] = data['H_qx'].apply(lambda x: x*-1)

## calc relative poses and append to data_out 
for row in data.to_dict(orient="records"):
    try:
        human_transformation, lidar_transformation = row2transMat(row)
        human_rel_posMat = np.matmul(np.linalg.inv(lidar_transformation),human_transformation)
        
        relDict = transMat2Dict(human_rel_posMat)
        relDict["timestamp"] = row['Time Elapsed']
        relDict["img_fn"] = row['Closest Image']
        relDict["point_cloud_fn"] = row['Closest Image'].replace('.png', '.pcd')

        rel_df = pd.DataFrame([relDict])
        data_out = pd.concat([data_out, rel_df], ignore_index=True)

    except:
        print(row)
    # break

###### visulaise results ###################################################################

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

for idx, row in enumerate(data_out.to_dict(orient="records")):
    if idx != 0:
        vis.remove_geometry(pcd) 
        vis.remove_geometry(human_pos)
        for marker_geo in marker_geometry_objs: vis.remove_geometry(marker_geo)

    ## load the point cloud data
    lidar_file = os.path.join(lidar_folder, row["point_cloud_fn"])
    pcd = o3d.io.read_point_cloud(lidar_file)
    pcd.paint_uniform_color([0, 0, 1]) ## blue 

    ## get translation matrix of the human 
    human_TMat = dict2transMat(row)

    # ## indicate current human position 
    human_pos = o3d.geometry.TriangleMesh.create_sphere(radius=.1)
    human_pos.translate(human_TMat[:3, 3])
    human_pos.paint_uniform_color([1, 1, 0])  # Yellow

    marker_geometry_objs = [] 
    ## no rotation for makers 
    # human_TMat[:3, :3] = np.identity(3)
    for marker_mat in human_makers:
        ## apply transform to move marker to correct loc 
        # marker_mat = np.matmul(human_TMat, marker_mat)
        marker_geo = o3d.geometry.TriangleMesh.create_sphere(radius=.075)
        marker_geo.translate(marker_mat[:3, 3])
        marker_geo.paint_uniform_color([1, 0, 0])  # Red
        vis.add_geometry(marker_geo)
        marker_geometry_objs.append(marker_geo)

    ## setup render 
    vis.add_geometry(pcd)
    vis.add_geometry(human_pos)

    # # ## update render 
    vis.poll_events()
    vis.update_renderer()

    time.sleep(.1)

    if idx == 30: break 
    break

vis.run()
vis.destroy_window()


###### unused code below ###################################################################

# fig_rows = 3

# fig, axs = plt.subplots(fig_rows,2)

# ## row 0 is lidar input 
# data.plot.line(x="S_x", y="S_z", ax = axs[0,0])
# data.plot.line(x="S_x", y="S_y", ax = axs[0,1])

# ## row 1 is human input 
# data.plot.line(x="H_x", y="H_z", ax = axs[1,0])
# data.plot.line(x="H_x", y="H_y", ax = axs[1,1])

# ## row 2 is the human relative to the lidar 
# data_out.plot.line(x="loc_x", y="loc_z", ax = axs[2,0])
# data_out.plot.line(x="loc_x", y="loc_y", ax = axs[2,1])

# axs[2,1].plot(data_out["loc_x"][0], data_out["loc_y"][0], 'ro')

# # print(data_out.head())

# ## set the same lims for every plot
# for i in range(2):
#     axs[i,0].set_xlim((data["S_x"].max(), data["S_x"].min()))
#     axs[i,0].set_ylim((data["S_z"].max(), data["S_z"].min()))

#     axs[i,1].set_xlim((data["S_x"].max(), data["S_x"].min()))
#     axs[i,1].set_ylim((data["S_y"].max(), data["S_y"].min()))

# plt.show()

