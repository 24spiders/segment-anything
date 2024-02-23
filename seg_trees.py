# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:51:15 2023

@author: Labadmin
"""

from itamtsupport.utils.img_annot_utils import read_pascalvoc
import matplotlib.pyplot as plt
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None

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
    
# xml_file = 'Lobstick_Prop_34_35_38_Ent_06-07-22_trimmed_preds.xml'
# img_file = 'Lobstick_Prop_34_35_38_Ent_06-07-22_trimmed.tif'
img_file = 'crop1.png'
xml_file = 'crop1_preds.xml'

import random
name, boxes = read_pascalvoc(xml_file)
random.shuffle(boxes)

im = Image.open(img_file)
im = np.array(im)
im = im[:,:,0:3]

# Init Segment
from segment_anything import SamPredictor
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

for box in boxes:
    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[2])
    ymax = int(box[3])
    

    predictor.set_image(im)
    input_box = np.array([xmin, ymin, xmax, ymax])
    point_labels = np.array([1])
    masks, scores, logits = predictor.predict(point_coords = None, point_labels = None, box = input_box)
    
    mask = masks[np.where(scores == max(scores))]
    plt.figure(figsize=(10,10))
    plt.imshow(im)
    show_mask(mask, plt.gca())
    # show_points(point_coords, point_labels, plt.gca())
    show_box(input_box, plt.gca())
    plt.title("Mask", fontsize=18)
    plt.axis('off')
    plt.show()

