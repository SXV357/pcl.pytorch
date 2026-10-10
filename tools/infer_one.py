import _init_paths # pylint: disable=unused-import
import torch
import cv2
import pickle
import numpy as np

from modeling.model_builder import Generalized_RCNN
from core.config import cfg, merge_cfg_from_file, assert_and_infer_cfg
from core.test import im_detect_all, box_results_with_nms_and_limit
from datasets.voc_eval import parse_rec
import utils.net as net_utils, utils.boxes as box_utils

YAML_PATH = 'configs/baselines/vgg16_voc2007.yaml'
CHECKPOINT_PATH = 'Outputs/vgg16_2007.pth'

IMAGE_PATH = 'data/VOCdevkit/VOC2007/JPEGImages/000001.jpg'
TRUE_ANNOTATIONS_PATH = 'data/VOCdevkit/VOC2007/Annotations/000001.xml'

PROPOSALS_PATH = 'data/selective_search_data/voc_2007_test.pkl'

merge_cfg_from_file(YAML_PATH)
cfg.MODEL.NUM_CLASSES = 20
cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False
cfg.TEST.BBOX_AUG.ENABLED = False

assert_and_infer_cfg()

model = Generalized_RCNN()
model.eval()

def unwrap(data, rois, labels):
    '''
    RCNN forward()
        -> data: image as numbers (1, channels, H, W)
        -> rois: proposal boxes
        -> labels: placeholder and all zeros (required)
    
    test.py calls model(**inputs)
    but inputs there is constructed like [tensor] which this func gets, extracts
    first elem and runs forward pass so it doens't complain
    '''
    return model(data[0], rois[0], labels[0])

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
net_utils.load_ckpt(model, checkpoint['model'])

im = cv2.imread(IMAGE_PATH)
assert im is not None

img_shape = im.shape

with open(PROPOSALS_PATH, 'rb') as f:
    proposals = pickle.load(f)

# could specify any test image and this will get the ID
image_id = int(IMAGE_PATH.split("/")[4].split(".")[0])
indexes, all_candidate_boxes = proposals['indexes'], proposals['boxes']

# this is what we'll use to lookup the proposals
target_idx = indexes.index(image_id)

# all the original proposal boxes
target_boxes = all_candidate_boxes[target_idx].astype(np.float32)
print(f"Number of original proposals for image id {image_id}: {target_boxes.shape[0]}")

# bit of cleanup
boxes = box_utils.clip_boxes_to_image(target_boxes, img_shape[0], img_shape[1])
keep = box_utils.unique_boxes(boxes)

boxes = boxes[keep, :]
keep = box_utils.filter_small_boxes(boxes, 2)

final_cleaned_boxes = boxes[keep, :]

print(f"Number of final proposals for image id {image_id} post cleaning: {final_cleaned_boxes.shape[0]}")

initial_detections = im_detect_all(unwrap, im, final_cleaned_boxes)
scores, boxes = initial_detections['scores'], initial_detections['boxes']

'''
scores: one row per proposal and one column per class (how confident is model that proposal i is class j)
col 0 is background and cols 1-20 are actual VOC classes

boxes: each class 1 box and box described by 4 numbers (so per proposal 21 boxes as well)
'''
print(f"Scores shape: {scores.shape}\n Boxes shape: {boxes.shape}")

filtered_predictions = box_results_with_nms_and_limit(scores, boxes)
cls_boxes = filtered_predictions[2]

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

for i in range(1, len(cls_boxes)):
    print(f"Class {i} = {classes[i-1]} predictions\n\n")
    predictions = cls_boxes[i]
    print(f"{predictions}\n")

    print(f"Number of predictions: {predictions.shape[0]}\n")
    print(f"Highest score: {predictions[:, 4].max()}\n")
    print("****************************")

true_annotations = parse_rec(TRUE_ANNOTATIONS_PATH)

for obj in true_annotations:
    name, bbox = obj['name'], obj['bbox']
    x1, y1, x2, y2 = bbox

    cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=4)

    cv2.putText(
        im, 
        f"GT {name}", 
        (x1, y1 - 10),               # Text origin (x, y) -> 10 pixels above the box
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=0.5, 
        color=(0, 0, 255), 
        thickness=1,
        lineType=cv2.LINE_AA          # Makes the text look smooth
    )

cutoff = 0.5
for j in range(1, 21):
    cl_boxes = cls_boxes[j]

    for x1, y1, x2, y2, score in cl_boxes:
        if score >= cutoff:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(im, (ix1, iy1), (ix2, iy2), color=(0, 255, 0), thickness=2)

            cv2.putText(
                im, 
                classes[j-1], 
                (ix1, iy1 - 10),               # Text origin (x, y) -> 10 pixels above the box
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, 
                color=(0, 255, 0), 
                thickness=1,
                lineType=cv2.LINE_AA          # Makes the text look smooth
            )

cv2.imwrite("Outputs/1.jpg", im)