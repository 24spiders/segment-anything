# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:43:52 2023

@author: Liam
"""

from PIL import Image
from skimage import io
import numpy as np
import torch
from torchvision.ops import batched_nms
from lsnms import nms
import os
from tqdm import tqdm
from pascal_voc_writer import Writer
import cv2
from itamtsupport.figures.draw_annots import draw_boxes_on_tif
from itamtsupport.utils.img_annot_utils import read_pascalvoc

## Tile into folder
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

def tile_and_predict(ortho_path: str, windows, path_to_model):

    # Open the ortho
    Image.MAX_IMAGE_PIXELS = None
    im = io.imread(ortho_path)
    im = Image.fromarray(im)
    im = im.convert('RGB')


    model = YOLO(path_to_model)
    # Crop the image based on the windows
    
    x = torch.empty(0,6)
    x = x.to(0)
    # global boxes
    print('Making predictions...')
    pbar = tqdm(total = len(windows))
    for window in windows:
        window_xmin = window.x
        window_ymin = window.y
        window_xmax = window.x + window.w
        window_ymax = window.y + window.h
        cropped = im.crop((window_xmin,window_ymin,window_xmax,window_ymax))
        # cropped = im.resize((640,640))
        
        cropped = np.array(cropped)
        cropped = Image.fromarray(cropped)
        preds = model.predict(source = cropped, save = False, save_txt = False, verbose = False, conf = 0.1, imgsz = 800, device = 0, max_det = 3000, iou = 0.1, show_labels = False, show_conf = False)
        boxes = preds[0].boxes.data
        boxes = torch.clone(boxes)
        boxes[:,0] = (boxes[:,0])
        boxes[:,1] = (boxes[:,1])
        boxes[:,2] = (boxes[:,2])
        boxes[:,3] = (boxes[:,3])
        boxes[:,0] = boxes[:,0] + window_xmin
        boxes[:,1] = boxes[:,1] + window_ymin
        boxes[:,2] = boxes[:,2] + window_xmin
        boxes[:,3] = boxes[:,3] + window_ymin
        x = torch.cat((x, boxes), 0)
        
        pbar.update(1)

    
    print('Predictions complete!')

    # global scores
    # global idxs
    boxes = x[:,0:4]
    scores = x[:,4]
    idxs = x[:,5]
    
    print('Performing NMS...')
    # global nms_boxes
    print(len(boxes))
    
    boxes = np.array(boxes.cpu())
    scores = np.array(scores.cpu())
    A = nms(boxes, scores, iou_threshold = 0.1)
    nms_boxes = x[A]
    

    nms_boxes = nms_boxes.cpu()
    nms_boxes = np.array(nms_boxes)
    print('NMS complete!')
    
    classes = ['con','dec','snag']
    
    # Init writer
    xml_path = ortho_path.replace('.png','_preds.xml')
    writer = Writer(xml_path[:-4], 0,0)


    # Iterate through the boxes
    print(len(nms_boxes))
    for box in nms_boxes:
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        label = classes[int(box[5])]
        conf = float(box[4])

        # Add box to the output
        writer.addObject(label, xmin, ymin, xmax, ymax, conf,)
    
    # Save the file
    writer.save(xml_path)

if __name__ == '__main__':
    path_to_model = 'best.pt'
    # image_path = 'Lobstick_Prop_34_35_38_Ent_06-07-22_trimmed.tif'
    image_path = 'crop1.png'
    from ultralytics import YOLO
    windows = _get_tiles(image_path, 800, 0.3)
    tiles = tile_and_predict(image_path, windows, path_to_model)
    name, boxes = read_pascalvoc(image_path.replace('.png','_preds.xml'))
    draw_boxes_on_tif(boxes, image_path,class_color=True)
    