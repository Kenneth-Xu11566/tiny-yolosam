import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

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

import sys
sys.path.append("..")
from tinysam import sam_model_registry, SamPredictor

model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint="./weights/tinysam_42.3.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)

predictor = SamPredictor(sam)

# Process all 3 images
images_config = [
    ('fig/picture1.jpg', np.array([[400, 400]]), 'demo_picture1.png'),
    ('fig/picture2.jpg', np.array([[400, 400]]), 'demo_picture2.png'),
    ('fig/picture3.jpg', np.array([[400, 400]]), 'demo_picture3.png'),
]

for img_path, input_point, output_path in images_config:
    print(f"Processing {img_path}...")
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictor.set_image(image)
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
    )
    
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks[scores.argmax(),:,:], plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")