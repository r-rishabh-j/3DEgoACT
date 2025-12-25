# reference implementation: https://github.com/mattcorsaro1/mj_pc
# with personal modifications


import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from typing import List
import open3d as o3d

"""
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""
def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


# 
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""
class PointCloudGenerator(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self, model, data, cam_ids:List, img_size=84):
        super(PointCloudGenerator, self).__init__()

        self.model = model
        self.data = data

        # this should be aligned with rgb
        self.img_width = img_size
        self.img_height = img_size

        self.cam_ids = cam_ids
        
        # List of camera intrinsic matrices
        self.cam_mats = []

        for cam_id in self.cam_ids:
            # get camera id
            # cam_id = self.model.camera_name2id(cam_id)
            fovy = math.radians(self.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)
        
        self.camExtMat = [self.get_world_transform(cam_id) for cam_id in self.cam_ids]

    def get_world_transform(self, cam_id):
    # 1. Get the computed World Position & Rotation
    # 'data.cam_xpos' is the vector position of the camera in World Frame.
    # 'data.cam_xmat' is the rotation matrix (flat 9-array) from Camera to World.
        cam_pos = self.data.cam_xpos[cam_id]
        cam_rot = self.data.cam_xmat[cam_id].reshape(3, 3)

        # 2. Build the MuJoCo-Frame Matrix (4x4)
        # This matrix describes the camera's pose in the world (looking down -Z)
        c2w_mujoco = np.eye(4)
        c2w_mujoco[:3, :3] = cam_rot
        c2w_mujoco[:3, 3] = cam_pos

        # 3. FIX THE COORDINATE SYSTEM (The Optical Correction)
        # MuJoCo cameras look down -Z. Open3D/PointClouds expect +Z.
        # We rotate 180 degrees around X to flip Y and Z.
        reorient_mat = np.array([
            [1,  0,  0,  0],
            [0, -1,  0,  0],
            [0,  0, -1,  0],
            [0,  0,  0,  1]
        ])

        # 4. Final Matrix: Optical Frame -> World Frame
        c2w_optical = c2w_mujoco @ reorient_mat
        
        return c2w_optical

    def compute_c2w_manually(self, cam_id):
        # --- 1. Get Positions from Model (Static XML values) ---
        # cam_id = self.model.camera_name2id(cam_name)
        target_body_id = self.model.cam_targetbodyid[cam_id]

        # Camera Position (Assuming defined in worldbody or static)
        # If the camera is a child of a moving body, this logic gets harder without 'data'.
        # Assuming standard static camera setup:
        cam_pos = self.model.cam_pos[cam_id]

        # Target Position
        target_pos = self.model.body_pos[target_body_id]

        # --- 2. Calculate the Look-At Rotation Vectors ---
        # In MuJoCo, the camera looks down the **Negative Z** axis.
        # So, the Z-axis vector points FROM Target TO Camera.
        forward_z = cam_pos - target_pos
        forward_z /= np.linalg.norm(forward_z) # Normalize
        
        # Global Up Vector (Standard Z-up world)
        global_up = np.array([0.0, 0.0, 1.0])
        
        # Right Vector (X-axis) = Cross(Global Up, Forward Z)
        right_x = np.cross(global_up, forward_z)
        if np.linalg.norm(right_x) < 1e-6:
            # Edge case: Looking straight down/up. X becomes arbitrary (usually +Y or +X).
            # Standard fallback:
            right_x = np.array([1.0, 0.0, 0.0])
        else:
            right_x /= np.linalg.norm(right_x)
            
        # Orthogonal Up Vector (Y-axis) = Cross(Forward Z, Right X)
        up_y = np.cross(forward_z, right_x)
        up_y /= np.linalg.norm(up_y)
        
        # --- 3. Construct the Rotation Matrix (3x3) ---
        # R = [X_col, Y_col, Z_col]
        cam_rot = np.stack([right_x, up_y, forward_z], axis=1)

        # --- 4. Build 4x4 Matrix (MuJoCo Frame) ---
        c2w_mujoco = np.eye(4)
        c2w_mujoco[:3, :3] = cam_rot
        c2w_mujoco[:3, 3] = cam_pos

        # --- 5. Optical Correction (MuJoCo -> Open3D) ---
        # Rotate 180 deg around X to flip Y and Z
        reorient_mat = np.array([
            [1,  0,  0,  0],
            [0, -1,  0,  0],
            [0,  0, -1,  0],
            [0,  0,  0,  1]
        ])

        c2w_optical = c2w_mujoco @ reorient_mat
        
        return c2w_optical

    def generateCroppedPointCloud(self, depth, color_img, save_img_dir=None):
        o3d_clouds = []
        depths = []

        for i, cam_id in enumerate(self.cam_ids):
            # 1. Capture Images
            # color_img should be (H, W, 3) uint8
            # depth should be (H, W) float32 (in meters)
            # color_img, depth = self.captureImage(renderer, color_renderer)
            depths.append(depth)
            # 2. Convert to Open3D format
            od_cammat = cammat2o3d(self.cam_mats[i], self.img_width, self.img_height)

            # Ensure memory layout and types are correct for Open3D
            od_color = o3d.geometry.Image(np.ascontiguousarray(color_img).astype(np.uint8))
            od_depth = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))

            # [FIX 2] Create RGBD Image to guarantee point/color alignment
            # depth_scale=1.0 is vital because MuJoCo returns meters. 
            # Default Open3D expects millimeters (scale=1000).
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                od_color, 
                od_depth, 
                depth_scale=1.0, 
                depth_trunc=20.0, 
                convert_rgb_to_intensity=False # Keep color info
            )

            # 3. Generate Cloud from RGBD
            o3d_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                od_cammat
            )

            # 4. Transform to World Frame
            c2w= self.camExtMat[i]
            transformed_cloud = o3d_cloud.transform(c2w)
            o3d_clouds.append(transformed_cloud)

        # 5. Combine Clouds
        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        
        # 6. Extract Numpy Arrays
        # points: (N, 3), colors: (N, 3) in range [0, 1]
        points = np.asarray(combined_cloud.points)
        colors = np.asarray(combined_cloud.colors)
        # 7. Concatenate
        # Output shape: (TARGET_NUM_POINTS, 6) -> [x, y, z, r, g, b]
        combined_cloud = np.concatenate((points, colors), axis=1)
        depths = np.array(depths).squeeze()
        
        return combined_cloud, depths

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, renderer, color_renderer, capture_depth=True):
        # rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=camera_name, depth=capture_depth, device_id=device_id)
        if capture_depth:
            img=color_renderer.render()
            depth=renderer.render()
            return img, depth
        else:
            img = color_renderer.render()
            return img

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img/img.max()*255
        normalized_image = normalized_image.astype(np.uint8)
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".jpg")
