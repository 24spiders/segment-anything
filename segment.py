# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:46:21 2023

@author: Labadmin
"""

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

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
    


from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import matplotlib.pyplot as plt
im = Image.open('Lobstick_Prop_34_35_38_Ent_06-07-22_trimmed.tif')
im = np.array(im)[5705:5705+2189,3470:3470+3031,0:3]
plt.figure(figsize=(10,10))
plt.imshow(im)
plt.title('Input Image', fontsize=18)
plt.axis('off')
plt.show()

sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(im)
i=0
for mask in masks:
    m = mask['segmentation']
    score = mask['predicted_iou']
    plt.figure(figsize=(10,10))
    plt.imshow(im)
    show_mask(m, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  
    i+=1

# from segment_anything import SamPredictor
# sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
# predictor = SamPredictor(sam)
# predictor.set_image(im)
# point_coords = np.array([[1500,1250],[1000,1250]])
# point_labels = np.array([1,1])
# masks, scores, logits = predictor.predict(point_coords = point_coords, point_labels = point_labels)

# i=0
# for mask in masks:
#     m = mask
#     plt.figure(figsize=(10,10))
#     plt.imshow(im)
#     show_mask(m, plt.gca())
#     show_points(point_coords, point_labels, plt.gca())
#     plt.title(f"Mask {i+1}", fontsize=18)
#     plt.axis('off')
#     plt.show()  
#     i+=1