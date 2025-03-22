from segment_anything import SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np

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

sam = sam_model_registry["vit_b"](checkpoint="/home/yumio/Code/FoundationPose/segment-anything/checkpoint/sam_vit_b_01ec64.pth")
sam.to("cuda")
predictor = SamPredictor(sam)
image = cv2.imread("/home/yumio/Code/FoundationPose/demo_data/square_table_leg/rgb/1581120424100262102.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_point = np.array([[860, 440]])
input_label = np.array([1])
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()
predictor.set_image(image)
plt.figure(figsize=(10,10))
plt.imshow(image)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
max_id = np.argmax(scores)
show_mask(masks[max_id], plt.gca())
show_points(input_point, input_label, plt.gca())
plt.title(f"Mask {max_id+1}, Score: {scores[max_id]:.3f}", fontsize=18)
plt.axis('off')
plt.show()