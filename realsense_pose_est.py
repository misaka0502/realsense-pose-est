import sys
sys.path.append('foundationpose')
sys.path.append('segment_anything')
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
from foundationpose.estimater import *
from foundationpose.datareader import *
import argparse
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import time
input_point = np.array([[860, 440]])
input_label = np.array([1])
color_image = None
mask_flag = False

def get_camera_intrinsics(pipeline):
    # get camera intrinsics
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])
    np.savetxt('cam_K.txt', camera_matrix)

def mouse_callback(event, x, y, flags, param):
    global input_point, input_label, color_image, mask_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        input_point = np.array([[x, y]])
        print(f"mouse click at {x}, {y}")
        mask_flag = True
        # image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # plt.imshow(image_rgb)
        # show_points(input_point, input_label, plt.gca())

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def prepare_pose(pipeline):
    global color_image, mask_flag
    sam = sam_model_registry["vit_b"](checkpoint="/home/yumio/Code/realsense_pose_est/segment_anything/checkpoint/sam_vit_b_01ec64.pth")
    sam.to("cuda")
    predictor = SamPredictor(sam)
    final_color_image = None
    mask_id = 0
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            final_color_image = color_image.copy()
            cv2.namedWindow("Color Image")
            cv2.setMouseCallback("Color Image", mouse_callback)
            cv2.imshow('Color Image', color_image)
            if mask_flag:
                predictor.set_image(color_image)
                masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
                )
                # for i, (mask, score) in enumerate(zip(masks, scores)):
                #     plt.figure(figsize=(10,10))
                #     show_mask(mask, plt.gca())
                #     show_points(input_point, input_label, plt.gca())
                #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                #     plt.axis('off')
                #     plt.show()  
                # mask_id = input("选择掩码：")
                # mask_id = int(mask_id) - 1
                # print(f"选择掩码第{mask_id+1}个掩码")
                max_id = np.argmax(scores)
                show_mask(masks[max_id], plt.gca())
                show_points(input_point, input_label, plt.gca())
                plt.title(f"Mask {max_id+1}, Score: {scores[max_id]:.3f}", fontsize=18)
                plt.show()
                mask_flag = False
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 按 ESC 退出
                cv2.destroyAllWindows()
                break
    binary_mask = (masks[max_id] * 255).astype(np.uint8)
    return final_color_image, binary_mask, depth_image

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
devoce_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

profile = pipeline.start(config)
logging.info("realsense initialization done")
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

mesh_file = '/home/yumio/Code/realsense_pose_est/foundationpose/demo_data/square_table_leg/mesh/square_table_leg.obj'
test_scene_dir = '/home/yumio/Code/realsense_pose_est/foundationpose/demo_data/square_table_leg'
code_dir = os.path.dirname(os.path.realpath(__file__))
debug = 1
debug_dir = f'{code_dir}/debug'
est_refine_iter = 5
track_refine_iter = 2
mesh = trimesh.load(mesh_file)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=360, zfar=np.inf)
logging.info("estimator initialization done")

color_image, mask, depth_image = prepare_pose(pipeline)
# while True:
#     cv2.imshow('color', color_image)
#     cv2.imshow('mask', mask)
#     cv2.imshow('depth', depth_image)
#     key = cv2.waitKey(1)
color = reader.get_color(color_image)
mask = reader.get_mask(mask)
depth = reader.get_depth(depth_image)
pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
center_pose = pose@np.linalg.inv(to_origin)
vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
cv2.imshow('1', vis[...,::-1])
cv2.waitKey(1)
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color = reader.get_color(color_image)
        depth = reader.get_depth(depth_image).astype(np.float32)
        pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=2)
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)
finally:
    pipeline.stop()