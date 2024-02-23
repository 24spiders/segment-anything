# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:01:59 2023

@author: Labadmin
"""
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage import io
import numpy as np
from tqdm import tqdm
import pickle

def read_pascalvoc(xml_file: str):
  '''This function takes in an xml file, and returns a nested list of box coordinates, and the box's type
  
  Arguments:
      xml_file: str, path to annotation file to be read
   
  Returns:
      filename: str, name of the file that was read
      list_with_all_boxes: list, A list of lists with format [xmin, ymin, xmax, ymax, label, prediction_confidence, extra1, extra2]
  
  '''
  # This was modified from: https://stackoverflow.com/questions/53317592/reading-pascal-voc-annotations-in-python
  import xml.etree.ElementTree as ET
  tree = ET.parse(xml_file)
  root = tree.getroot()
  # Initialize list
  list_with_all_boxes = []
  # For each box listed in the .xml file
  filename = root.find('filename').text
  for boxes in root.iter('object'):
    # Initialize bounds, class
    ymin, xmin, ymax, xmax, label = None, None, None, None, None
    # Read the information from the .xml
    ymin = float(boxes.find("bndbox/ymin").text)
    xmin = float(boxes.find("bndbox/xmin").text)
    ymax = float(boxes.find("bndbox/ymax").text)
    xmax = float(boxes.find("bndbox/xmax").text)
    label = boxes.find("name").text
    conf = boxes.find('pose').text
    xtra1 = boxes.find("truncated").text
    xtra2 = boxes.find("difficult").text
    # Put the information into a list, append that list to master list to be output
    list_with_single_boxes = [xmin, ymin, xmax, ymax, label, conf, xtra1, xtra2]
    list_with_all_boxes.append(list_with_single_boxes)
  # Return the filename, and the list of all box bounds/classes
  return filename, list_with_all_boxes
import matplotlib.pyplot as plt


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
def _get_tiles(ortho_path: str, tile_size, overlap):
  ''' Uses a python package (slidingwindow) to generate windows over the orthophoto. 
  
  Arguments: 
      ortho_path: str, the orthomosaic for which predictions will be generated
      tile_size: the size of the tiles in px 
      overlap: the % amount the tiles will overlap
      
  Returns:
      windows: a list of windows to be used to slice the image
  '''
  # Imports
  import slidingwindow as sw
  # Open the image, convert to array
  Image.MAX_IMAGE_PIXELS = None
  im = io.imread(ortho_path)
  im = Image.fromarray(im)
  im = im.convert('RGB')
  im_array = np.array(im)
  # Returns a list of windows, each window as (self.x, self.y, self.w, self.h)
  windows = sw.generate(im_array, sw.DimOrder.HeightWidthChannel, tile_size, overlap)
  im.close()
  return windows

xml_file = 'Lobstick_Prop_34_35_38_Ent_06-07-22_trimmed_preds.xml'
ortho_path = 'Lobstick_Prop_34_35_38_Ent_06-07-22_trimmed.tif'

# ortho_path = 'Coleman_Flight3_Mini_06-12-20_trimmed.tif'
# xml_file = 'J12F3 - RGB_preds.xml'

# ortho_path = './surv2/Lobstick_Prop_13_Mini_06-07-22_trimmed.tif'
# xml_file = './surv2/Lobstick_Prop_13_Mini_06-07-22_trimmed_preds.xml'
# out_path = "./surv2/Lobstick_Prop_13_Mini_06-07-22_trimmed_pred_masks.pkl"

# ortho_path = './surv3/J10F2 - RGB.tif'
# xml_file = './surv3/J10F2 - RGB_preds.xml'
# out_path = "./surv3/J10F2 - RGB_pred_masks.pkl"

# ortho_path = './cynthia/608068_5906903_aoi_ht_reproj.tif'
# xml_file = './cynthia/608068_5906903_aoi_ht_reproj_preds.xml'
# out_path = "./cynthia/608068_5906903_aoi_ht_reproj_pred_masks.pkl"

tile_size = 800
overlap = 0.3

name, boxes = read_pascalvoc(xml_file)
boxes = np.array(boxes)

im = io.imread(ortho_path)
im = Image.fromarray(im)
im = im.convert('RGB')

windows = _get_tiles(ortho_path, tile_size, overlap)

# Init Segment
import torch

# from segment_anything import SamPredictor, sam_model_registry
# sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
from segment_anything_hq import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_l"](checkpoint="sam_hq_vit_l.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
sam.to(device=device)
predictor = SamPredictor(sam)

out = []
pbar = tqdm(total = len(windows))
ii = 0
# try:
    # while True:
for window in windows:
    # if ii < 80:
    #     pbar.update(1)
    #     ii+=1
    #     continue
    # else:
        window_xmin = window.x
        window_ymin = window.y
        window_xmax = window.x + window.w
        window_ymax = window.y + window.h
        # Tile image
        cropped = im.crop((window_xmin,window_ymin,window_xmax,window_ymax))
        cropped = np.array(cropped)
        # Get boxes in the tile
        m = (boxes[:,0].astype(float) > window_xmin) & (boxes[:,2].astype(float) < window_xmax) & (boxes[:,1].astype(float) > window_ymin) & (boxes[:,3].astype(float) < window_ymax)
        tile_boxes = boxes[m]
        plt.figure(figsize=(10,10))
        plt.imshow(cropped)
        
        for box in tile_boxes:
            
            xmin = int(float(box[0])) - window_xmin
            ymin = int(float(box[1])) - window_ymin
            xmax = int(float(box[2])) - window_xmin
            ymax = int(float(box[3])) - window_ymin
            
            predictor.set_image(cropped)
            input_box = np.array([xmin, ymin, xmax, ymax])
            point_coords = np.array([(xmax - xmin) / 2 + xmin, (ymax - ymin) / 2 + ymin])
            point_labels = np.array([1])
            masks, scores, logits = predictor.predict(point_coords = None, point_labels = None, box = input_box)
            mask = masks[np.where(scores == max(scores))]
            show_mask(mask, plt.gca())
            # show_points(point_coords, point_labels, plt.gca())
            show_box(input_box, plt.gca())
            
            outmask = mask[0, ymin:ymax, xmin:xmax]
    
            out.append([box, outmask, window])
        plt.title("Mask", fontsize=18)
        plt.axis('off')
        plt.show()
        pbar.update(1)
        
        ii += 1
                
# except KeyboardInterrupt:
#     pass
    

with open(out_path,"wb") as f:
    pickle.dump(out,f)

    f.close()