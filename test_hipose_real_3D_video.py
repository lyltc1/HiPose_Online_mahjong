import sys
import os
import json
import cv2
import argparse
import numpy as np
import torch
import open3d as o3d
from open3d.camera import PinholeCameraIntrinsic

import normalSpeed
from ultralytics import YOLO

sys.path.append('/home/HiPose_Online_mahjong/docker/hipose')
from models.RandLA.helper_tool import DataProcessing as DP
from models.ffb6d import FFB6D
from models.common import ConfigRandLA

from bop_dataset_3d_convnext_backbone import bop_dataset_single_obj_3d, get_K_crop_resize, padding_Bbox, get_final_Bbox, get_roi
from binary_code_helper.CNN_output_to_pose_region import load_dict_class_id_3D_points
from binary_code_helper.generate_new_dict import generate_new_corres_dict_and_region
from binary_code_helper.CNN_output_to_pose_region_for_test_with_region_v3 import CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v7


def load_yolo_model(model_path):
    model = YOLO(model=model_path)
    model.eval()
    model.to(device='cuda', dtype=torch.bfloat16)
    return model

def load_hipose_model(args):
    rndla_cfg = ConfigRandLA
    pose_model = FFB6D(
        n_classes=1, n_pts=480 * 640 // 24 , rndla_cfg=rndla_cfg,
        number_of_outputs=16 + 1, fusion=False,
        convnext="convnext_base"
    )
    pose_model=pose_model.cuda()
    checkpoint = torch.load(args.ckpt_file)
    pose_model.load_state_dict(checkpoint['model_state_dict'])
    pose_model.eval()
    return pose_model

def load_rgb_depth_path(rgb_folder, depth_folder):
    rgb_paths = sorted([os.path.join(rgb_folder, file)  for file in os.listdir(rgb_folder) if file.endswith(('.jpg', '.png'))])
    depth_paths = sorted([os.path.join(depth_folder, file)  for file in os.listdir(depth_folder) if file.endswith('.png')])
    assert len(rgb_paths) == len(depth_paths), "RGB and depth images count mismatch"
    return rgb_paths, depth_paths

def load_3d_boxes(model_info_path):
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    min_x = model_info['1']['min_x']
    min_y = model_info['1']['min_y']
    min_z = model_info['1']['min_z']
    size_x = model_info['1']['size_x']
    size_y = model_info['1']['size_y']
    size_z = model_info['1']['size_z']
    max_x = min_x + size_x
    max_y = min_y + size_y
    max_z = min_z + size_z
    box3d = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z]
    ])
    return box3d

def prepare_input_dict(img_path, depth_path, Cam_K, detected_bbox):
    BoundingBox_CropSize_image = 256
    resize_method = "crop_square_resize"

    rgb_img = cv2.imread(img_path)
    Bbox = padding_Bbox(detected_bbox, padding_ratio=1.5)
    roi_rgb = get_roi(rgb_img, Bbox, 256, interpolation=cv2.INTER_LINEAR, resize_method="crop_square_resize")
    Bbox = get_final_Bbox(Bbox, "crop_square_resize", rgb_img.shape[1], rgb_img.shape[0])
    depth_image_mm = bop_dataset_single_obj_3d.read_depth(depth_path, 1.0)
    depth_image_m = depth_image_mm / 1000.
    cam_param_new = get_K_crop_resize(Cam_K, Bbox, 256, 256)
    roi_depth = get_roi(depth_image_m, Bbox, BoundingBox_CropSize_image, interpolation=cv2.INTER_NEAREST, resize_method = "crop_square_resize")
    roi_dpt_xyz = bop_dataset_single_obj_3d.dpt_2_pcld(roi_depth, 1.0, cam_param_new, BoundingBox_CropSize_image,BoundingBox_CropSize_image)
    roi_dpt_xyz[np.isnan(roi_dpt_xyz)] = 0.0
    roi_dpt_xyz[np.isinf(roi_dpt_xyz)] = 0.0
    roi_depth_mm_int = (1000*roi_depth).astype(np.uint16)
    roi_nrm_map = normalSpeed.depth_normal(
        roi_depth_mm_int, cam_param_new[0,0], cam_param_new[1,1], 5, 2000, 20, False
    )
    mask_dp = roi_depth > 1e-6
    valid_depth_idx = mask_dp.flatten().nonzero()[0].astype(np.uint64)  # index of all valid points
    if len(valid_depth_idx) == 0:
        return 0,0,{}

    n_points = int(BoundingBox_CropSize_image*BoundingBox_CropSize_image/24)

    selected_idx = np.array([i for i in range(len(valid_depth_idx))])  # from 0 to length
    if len(selected_idx) > n_points:
        c_mask = np.zeros(len(selected_idx), dtype=int)
        c_mask[:n_points] = 1
        np.random.shuffle(c_mask)
        selected_idx = selected_idx[c_mask.nonzero()]  # if number of points are enough, random choose n_sample_points
    else:
        selected_idx = np.pad(selected_idx, (0, n_points-len(selected_idx)), 'wrap') 

    selected_point_idx = np.array(valid_depth_idx)[selected_idx]  # index of selected points, which has number of n_sample_points

    # shuffle the idx to have random permutation
    sf_idx = np.arange(selected_point_idx.shape[0])
    np.random.shuffle(sf_idx)
    selected_point_idx = selected_point_idx[sf_idx]   

    roi_cld = roi_dpt_xyz.reshape(-1, 3)[selected_point_idx, :]    # random selected points from all valid points

    roi_pt_rgb = roi_rgb.reshape(-1, 3)[selected_point_idx, :].astype(np.float32)
    roi_pt_nrm = roi_nrm_map[:, :, :3].reshape(-1, 3)[selected_point_idx, :]

    selected_point_idx = np.array([selected_point_idx])
    roi_cld_rgb_nrm = np.concatenate((roi_cld, roi_pt_rgb, roi_pt_nrm), axis=1).transpose(1, 0)

    h = w = BoundingBox_CropSize_image

    xyz_list = [roi_dpt_xyz.transpose(2, 0, 1)]  # c, h, w
    mask_list = [roi_dpt_xyz[2, :, :] > 1e-8]

    for i in range(4):   # add different scaled input into the list
        scale = pow(2, i+1)
        nh, nw = h // pow(2, i+1), w // pow(2, i+1)
        ys, xs = np.mgrid[:nh, :nw]
        xyz_list.append(xyz_list[0][:, ys*scale, xs*scale])    
        mask_list.append(xyz_list[-1][2, :, :] > 1e-8)

    scale2dptxyz = {
        pow(2, ii): item.reshape(3, -1).transpose(1, 0)
        for ii, item in enumerate(xyz_list)
    }     # c x h x w to h*w x 3 

    rgb_downsample_scale = [4, 8, 8, 8]
    rgb_downsample_scale = [4, 8, 16, 16]
    n_ds_layers = 4
    pcld_sub_sample_ratio = [4, 4, 4, 4]

    inputs = {}
    # DownSample stage
    for i in range(n_ds_layers):
        nei_idx = DP.knn_search(
            roi_cld[None, ...], roi_cld[None, ...], 16
        ).astype(np.int64).squeeze(0)  # find 16 neiborhood for each point in the selected point cloud
        sub_pts = roi_cld[:roi_cld.shape[0] // pcld_sub_sample_ratio[i], :]    # can dowmsample , due to the index is schuffeled
        pool_i = nei_idx[:roi_cld.shape[0] // pcld_sub_sample_ratio[i], :]
        up_i = DP.knn_search(
            sub_pts[None, ...], roi_cld[None, ...], 1
        ).astype(np.int64).squeeze(0)
        inputs['cld_xyz%d' % i] = roi_cld.astype(np.float32).copy()  # origin xyz
        inputs['cld_nei_idx%d' % i] = nei_idx.astype(np.int64).copy()  # find 16 neiborhood
        inputs['cld_sub_idx%d' % i] = pool_i.astype(np.int64).copy()  # sub xyz neiborhood
        inputs['cld_interp_idx%d' % i] = up_i.astype(np.int64).copy()  # origin xyz find 1 neiborhoood in sub xyz
        nei_r2p = DP.knn_search(
            scale2dptxyz[rgb_downsample_scale[i]][None, ...], sub_pts[None, ...], 16
        ).astype(np.int64).squeeze(0)  # sub xyz find 16 neiborhood in downsampled depth
        inputs['r2p_ds_nei_idx%d' % i] = nei_r2p.copy()
        nei_p2r = DP.knn_search(
            sub_pts[None, ...], scale2dptxyz[rgb_downsample_scale[i]][None, ...], 1
        ).astype(np.int64).squeeze(0)
        inputs['p2r_ds_nei_idx%d' % i] = nei_p2r.copy()
        roi_cld = sub_pts

    n_up_layers = 3
    rgb_up_sr = [4, 2, 2]
    rgb_up_sr = [8, 4, 4]
    for i in range(n_up_layers):
        r2p_nei = DP.knn_search(
            scale2dptxyz[rgb_up_sr[i]][None, ...],
            inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
        ).astype(np.int64).squeeze(0)
        inputs['r2p_up_nei_idx%d' % i] = r2p_nei.copy()
        p2r_nei = DP.knn_search(
            inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
            scale2dptxyz[rgb_up_sr[i]][None, ...], 1
        ).astype(np.int64).squeeze(0)
        inputs['p2r_up_nei_idx%d' % i] = p2r_nei.copy()

    roi_rgb, roi_cld_rgb_nrm, selected_point_idx, roi_depth = bop_dataset_single_obj_3d.transform_pre_inputs(roi_rgb, roi_cld_rgb_nrm, selected_point_idx, roi_depth)

    for key in inputs:
        inputs[key] = torch.from_numpy(inputs[key])

    inputs.update( 
        dict(
        rgb=roi_rgb,  # [c, h, w]
        cld_rgb_nrm=roi_cld_rgb_nrm,  # [9, npts]
        choose=selected_point_idx,  # [1, npts]
        dpt_map_m=roi_depth,  # [h, w]
        Bbox = Bbox
    )
    )

    inputs["cam_param_new"] = cam_param_new

    return inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONLY for testing wan7 of mahjong')
    parser.add_argument('--obj_name', type=str, default='obj01')
    parser.add_argument('--yolo_path', default="/home/HiPose_Online_mahjong/ultralytics/runs/detect/mahjong/weights/best.pt")
    parser.add_argument('--ckpt_file', type=str, default='/home/HiPose_Online_mahjong/output/0_9051step164000')
    parser.add_argument('--mesh_path', type=str, default='/home/HiPose_Online_mahjong/output/dataset/models/obj_000001.ply')
    parser.add_argument('--model_info_path', default='/home/HiPose_Online_mahjong/output/dataset/models/models_info.json')
    parser.add_argument('--camera_path', default='/home/HiPose_Online_mahjong/output/dataset/camera.json')
    parser.add_argument('--Class_CorresPoint_path', default='/home/HiPose_Online_mahjong/output/dataset/models_GT_color/Class_CorresPoint000001.txt')
    parser.add_argument('--region_bit', type=int, default=10)
    parser.add_argument('--rgb_path', default='/home/HiPose_Online_mahjong/output/dataset/test/rgb')
    parser.add_argument('--depth_path', default='/home/HiPose_Online_mahjong/output/dataset/test/depth')
    parser.add_argument('--output_video', type=str, default="/home/HiPose_Online_mahjong/output_video_real_pcd.mp4")
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--frame_width', type=int, default=1920)
    parser.add_argument('--frame_height', type=int, default=1080)
    args = parser.parse_args()

    detection_model = load_yolo_model(args.yolo_path)
    pose_model = load_hipose_model(args)
    rgb_paths, depth_paths = load_rgb_depth_path(args.rgb_path, args.depth_path)
    box3d = load_3d_boxes(args.model_info_path)  # [8, 3]
    total_numer_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(args.Class_CorresPoint_path)

    bit2class_id_center_and_region = {}
    for bit in range(args.region_bit + 1, 17):
        bit2class_id_center_and_region[bit] = generate_new_corres_dict_and_region(dict_class_id_3D_points, 16, bit)

    # complete bit2class_id_center_and_region so that all regions share the same shape, default: 32
    region_max_points = pow(2, 15 - args.region_bit)
    for bit in range(args.region_bit + 1, 17):
        for center_and_region in bit2class_id_center_and_region[bit].values():
            region = center_and_region['region']
            assert region.shape[0] <= region_max_points
            if region.shape[0] < region_max_points:
                region_new = np.zeros([region_max_points, 3])
                region_new[:region.shape[0]] = region
                region_new[region.shape[0]:] = region[0]
                center_and_region['region'] = region_new

    with open(args.camera_path, 'r') as f:
        cam_data = json.load(f)

    K = np.array([
        [cam_data["fx"], 0, cam_data["cx"]],
        [0, cam_data["fy"], cam_data["cy"]],
        [0, 0, 1]
    ])
    
    video_path = args.output_video
    frame_width, frame_height = args.frame_width, args.frame_height
    fps = args.fps
    
    # Use MP4V codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer at {video_path}")
        exit(1)
    
    print(f"Writing video to {video_path} at {fps} FPS with resolution {frame_width}x{frame_height}")

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])

    # get detected bbox
    for rgb_path, depth_path in zip(rgb_paths, depth_paths):
        image_array = cv2.imread(rgb_path)
        results = detection_model.predict(image_array, conf=0.6)
        id2labels = results[0].names
        cls = results[0].boxes.cls
        conf = results[0].boxes.conf
        xyxy = results[0].boxes.xyxy

        # use open3d to generate point cloud from rgb_path and depth_path
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(frame_width, frame_height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        rgb_image = o3d.io.read_image(rgb_path)
        depth_image = o3d.io.read_image(depth_path)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image, depth_image, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        # Visualize the point cloud
        vis.add_geometry(pcd)
        # Write the frame to the video
        view_control = vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        camera_params.intrinsic = PinholeCameraIntrinsic(1920, 1080, 935.30743609, 935.30743609, 959.5, 539.5)
        # 设置相机外参矩阵，调整相机位置使对象更近
        camera_params.extrinsic = np.array([[1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.15757],
                                        [0.0, 0.0, 1.0, -0.24235],
                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        view_control.convert_from_pinhole_camera_parameters(camera_params)
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        for i, box in enumerate(xyxy):
            label = id2labels[int(cls[i])]
            confidence = conf[i].item()
            if confidence < 0.7:
                continue
            x1, y1, x2, y2 = map(int, box)
            detected_bbox = [x1, y1, x2, y2]

            inputs = prepare_input_dict(img_path=rgb_path, depth_path=depth_path, Cam_K=K, detected_bbox=[x1, y1, x2-x1, y2-y1])
            if not inputs:
                continue

            if torch.cuda.is_available():
                for key in inputs:
                    if not isinstance(inputs[key], torch.Tensor):
                        inputs[key] = torch.tensor(inputs[key])
                    inputs[key] = torch.unsqueeze(inputs[key], 0).cuda()
            pred_masks_prob, pred_code_prob = pose_model(inputs)
            pred_masks_probability = torch.sigmoid(pred_masks_prob).detach().cpu().numpy()
            pred_codes_probability = torch.sigmoid(pred_code_prob).detach().cpu().numpy()
            inputs_pc = inputs['cld_xyz0'].detach().cpu().numpy()
            R_predict, t_predict, success = CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v7(
                inputs_pc[0], 
                pred_masks_probability[0], 
                pred_codes_probability[0], 
                bit2class_id_center_and_region=bit2class_id_center_and_region,
                dict_class_id_3D_points=dict_class_id_3D_points,
                region_bit=args.region_bit,
                threshold=50,
                mean=False,
                uncertain_threshold = 0.02
                ) 
            if success:
                # Transform the 3D bounding box points from object to camera frame
                transformed_points_camera_frame = R_predict @ box3d.T + t_predict  # Apply rotation and translation
                transformed_points_camera_frame = transformed_points_camera_frame.T * 0.001  # [8, 3]

                transformed_points_world_frame = transformed_points_camera_frame
                # Create a LineSet for the 3D bounding box in world frame
                lines = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
                ]
                colors = [[0, 1, 0] for _ in range(len(lines))]  # Green color for all edges

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(transformed_points_world_frame)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                # Create a LineWidth option for wider lines
                line_set.paint_uniform_color([0, 1, 0])  # Ensure the lines are green
                vis.add_geometry(line_set)
                
                # Define cylinder parameters
                cylinder_radius = 0.0005  # Adjustable radius
                cylinder_height = 0.1  # Default height, will be adjusted per edge
                
                # Create cylinders for each edge of the bounding box
                for i, line in enumerate(lines):
                    start_idx, end_idx = line
                    start_point = transformed_points_world_frame[start_idx]
                    end_point = transformed_points_world_frame[end_idx]
                    
                    # Calculate edge vector and length
                    edge_vector = end_point - start_point
                    edge_length = np.linalg.norm(edge_vector)
                    
                    # Create cylinder geometry
                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                        radius=cylinder_radius,
                        height=edge_length
                    )
                    
                    # Position and orient the cylinder
                    # Translate to midpoint
                    midpoint = (start_point + end_point) / 2
                    cylinder.translate(midpoint)
                    
                    # Rotate to align with edge vector
                    rotation_matrix = np.eye(3)
                    if edge_length > 0:
                        # Create rotation matrix to align z-axis with edge vector
                        z_axis = edge_vector / edge_length
                        # Find a vector perpendicular to z_axis in the original plane
                        # Using cross product with an arbitrary vector not parallel to z_axis
                        arbitrary_vector = np.array([0, 0, 1]) if not np.allclose(z_axis, np.array([0, 0, 1])) else np.array([0, 1, 0])
                        x_axis = np.cross(arbitrary_vector, z_axis)
                        x_axis = x_axis / np.linalg.norm(x_axis)
                        y_axis = np.cross(z_axis, x_axis)
                        
                        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
                    
                    cylinder.rotate(rotation_matrix)
                    
                    # Set color
                    if i == 0:  # Change the first line's color to red
                        cylinder.paint_uniform_color([1, 0, 0])
                    else:
                        cylinder.paint_uniform_color([0, 1, 0])
                    
                    # Add to visualizer
                    vis.add_geometry(cylinder)
        # Write the frame to the video
        view_control = vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        camera_params.intrinsic = PinholeCameraIntrinsic(1920, 1080, 935.30743609, 935.30743609, 959.5, 539.5)
        # 设置相机外参矩阵，调整相机位置使对象更近
        camera_params.extrinsic = np.array([[1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.15757],
                                        [0.0, 0.0, 1.0, -0.24235],
                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        view_control.convert_from_pinhole_camera_parameters(camera_params)
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        # 锁定视图控制，防止自动调整
        view_control.set_constant_z_far(10)
        view_control.set_constant_z_near(0.01)
        # Use Open3D's LineWidth option to make the lines wider
        vis.poll_events()
        vis.update_renderer()
        # Capture the current view as an image
        # Capture the current view as an image and save it to a local variable
        image = vis.capture_screen_float_buffer(True)
        vis.clear_geometries()
        image_array_pcd = (255 * np.asarray(image)).astype(np.uint8)
        resized_frame = cv2.resize(image_array_pcd, (frame_width, frame_height))
        video_writer.write(resized_frame[:,:, ::-1])  # Convert RGB to BGR for OpenCV
    # 确保释放视频写入器
    video_writer.release()
    print(f"Video saved to {video_path}")
    vis.destroy_window()
