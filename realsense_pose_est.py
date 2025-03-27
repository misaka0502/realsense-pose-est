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
num_poses = 2

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
    mask_list = []
    color_images = []
    depth_images = []
    for i in range(num_poses):
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
                    max_id = np.argmax(scores)
                    show_mask(masks[max_id], plt.gca())
                    show_points(input_point, input_label, plt.gca())
                    plt.title(f"Mask {max_id+1}, Score: {scores[max_id]:.3f}", fontsize=18)
                    plt.show()
                    mask_flag = False

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # 按 ESC 退出
                    binary_mask = (masks[max_id] * 255).astype(np.uint8)
                    mask_list.append(binary_mask)
                    color_images.append(color_image)
                    depth_images.append(depth_image)
                    cv2.destroyAllWindows()
                    break

    del sam, predictor
    torch.cuda.empty_cache()
    return color_images, mask_list, depth_images

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

mesh_files = []
mesh_files.append('/home/yumio/Code/realsense_pose_est/foundationpose/demo_data/square_table_leg/mesh/square_table_leg.obj')
mesh_files.append('/home/yumio/Code/realsense_pose_est/foundationpose/demo_data/square_table/mesh/square_table.obj')
test_scene_dirs = []
test_scene_dirs.append('/home/yumio/Code/realsense_pose_est/foundationpose/demo_data/square_table_leg')
test_scene_dirs.append('/home/yumio/Code/realsense_pose_est/foundationpose/demo_data/square_table')
code_dir = os.path.dirname(os.path.realpath(__file__))
debug = 1
debug_dir = f'{code_dir}/debug'
est_refine_iter = 5
track_refine_iter = 2

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()

meshs = []
to_origins = []
extents = []
bboxs = []
ests = []
readers = []
for i in range(num_poses):
    meshs.append(trimesh.load(mesh_files[i], force='mesh'))
    to_origin, extent = trimesh.bounds.oriented_bounds(meshs[i])
    to_origins.append(to_origin)
    extents.append(extent)
    bboxs.append(np.stack([-extent/2, extent/2], axis=0).reshape(2,3))
    ests.append(FoundationPose(model_pts=meshs[i].vertices, model_normals=meshs[i].vertex_normals, mesh=meshs[i], scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx))
    readers.append(YcbineoatReader(video_dir=test_scene_dirs[i], shorter_side=360, zfar=np.inf))

logging.info("estimator initialization done")

color_image, mask, depth_image = prepare_pose(pipeline)
# while True:
#     cv2.imshow('color', color_image)
#     cv2.imshow('mask', mask)
#     cv2.imshow('depth', depth_image)
#     key = cv2.waitKey(1)
colors = [readers[i].get_color(color_image[i]) for i in range(num_poses)]
masks = [readers[i].get_mask(mask[i]) for i in range(num_poses)]
depths = [readers[i].get_depth(depth_image[i]) for i in range(num_poses)]
poses = [ests[i].register(K=readers[i].K, rgb=colors[i], depth=depths[i], ob_mask=masks[i], iteration=est_refine_iter) for i in range(num_poses)]
center_poses = [poses[i]@np.linalg.inv(to_origins[i]) for i in range(num_poses)]
vis = draw_posed_3d_box(readers[0].K, img=colors[0], ob_in_cam=center_poses[0], bbox=bboxs[0])
vis = draw_xyz_axis(vis, ob_in_cam=center_poses[0], scale=0.1, K=readers[0].K, thickness=3, transparency=0, is_input_rgb=True)
for i in range(1, num_poses):
    vis = draw_posed_3d_box(readers[i].K, img=vis, ob_in_cam=center_poses[i], bbox=bboxs[i])
    vis = draw_xyz_axis(vis, ob_in_cam=center_poses[i], scale=0.1, K=readers[i].K, thickness=3, transparency=0, is_input_rgb=True)
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
        color = readers[0].get_color(color_image)
        depth = readers[0].get_depth(depth_image)
        poses = [ests[i].track_one(rgb=color, depth=depth, K=readers[i].K, iteration=2) for i in range(num_poses)]
        center_poses = [poses[i]@np.linalg.inv(to_origins[i]) for i in range(num_poses)]
        vis = draw_posed_3d_box(readers[0].K, img=color, ob_in_cam=center_poses[0], bbox=bboxs[0])
        vis = draw_xyz_axis(vis, ob_in_cam=center_poses[0], scale=0.1, K=readers[0].K, thickness=3, transparency=0, is_input_rgb=True)
        for i in range(1, num_poses):
            vis = draw_posed_3d_box(readers[1].K, img=vis, ob_in_cam=center_poses[1], bbox=bboxs[1])
            vis = draw_xyz_axis(vis, ob_in_cam=center_poses[1], scale=0.1, K=readers[1].K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)
finally:
    pipeline.stop()