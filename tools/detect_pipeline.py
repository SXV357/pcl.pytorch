from pathlib import Path
import sys
from typing import Union
import cv2
import torch
import pickle
import numpy as np

# root of this fork essentially
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

# settings file describing how OICR is built + run: backbone to use (VGG16), test image size, NMS threshold
YAML_PATH = ROOT / "configs/baselines/vgg16_voc2007.yaml"

# trained OICR weights: VGG16 backbone initialized from imagenet and fine tuned on VOC 2007
# plus OICR's detection heads
CHECKPOINT_PATH = ROOT / "Outputs/vgg16_2007.pth"

# lookup: since OICR cant find boxes by itself and only scores candidate boxes
# that selective search made beforehand - these files have candidates for every image
TEST_PROPOSALS_PATH = ROOT / "data/selective_search_data/voc_2007_test.pkl"
TRAINVAL_PROPOSALS_PATH = ROOT / "data/selective_search_data/voc_2007_trainval.pkl"

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

from modeling.model_builder import Generalized_RCNN
from core.config import cfg, merge_cfg_from_file, assert_and_infer_cfg
from core.test import im_detect_all, box_results_with_nms_and_limit
import utils.net as net_utils, utils.boxes as box_utils


class DetectPipeline():
    '''
    OICR (VGG16) wrapper for VOC 2007

    Make sure to only create one instance per process since the 
    global `cfg` variable is locked after init. Additionally this runs
    at single scale without flip so the clean mAP will be slightly lower than
    the published one of 52.9.
    '''

    def __init__(self, device: Union[str, torch.device]):
        self.device = device

        merge_cfg_from_file(YAML_PATH)
        cfg.MODEL.NUM_CLASSES = 20
        cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False

        # test time augmentation: if on each image run 10 times (5 sizes and more) results averaged
        # and how authors got 52.9 mAP; since off, only 1 run per image at 1 size which is faster but
        # bit lower mAP
        cfg.TEST.BBOX_AUG.ENABLED = False

        assert_and_infer_cfg()

        self.model = Generalized_RCNN()

        # in this repo: makes forward return detections rather than training losses + disable gradients
        self.model.eval()

        # location to load checkpoint tensors (weights in CPU RAM)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        net_utils.load_ckpt(self.model, checkpoint['model']) 

        # copy weights to GPU or leave in CPU if device is "cpu"
        self.model.to(self.device) 

        with open(TEST_PROPOSALS_PATH, 'rb') as tp:
            test_proposals = pickle.load(tp)  

        with open(TRAINVAL_PROPOSALS_PATH, 'rb') as tvp:
            trainval_proposals = pickle.load(tvp)

        # building these mappings since on RL side detect() will be called frequently
        # and don't wanna have to do .index() every single time adding latency
        test_proposal_lookup = dict(zip(test_proposals['indexes'], test_proposals['boxes']))
        trainval_proposal_lookup = dict(zip(trainval_proposals['indexes'], trainval_proposals['boxes']))

        self.proposals_lookup = test_proposal_lookup | trainval_proposal_lookup

    def unwrap(self, data: list[torch.Tensor], rois: list[torch.Tensor], labels: list[torch.Tensor]):
        '''
        RCNN forward()
        - data: image as numbers (1, channels, H, W)
        - rois: proposal boxes
        - labels: placeholder and all zeros (required)
            
        test.py calls model(**inputs)
        but inputs there is constructed like [tensor] which this func gets, extracts
        first elem and runs forward pass so it doens't complain

        test.py creates fresh CPU tensors for every image and if we want inference
        to run on GPU just models weights on GPU isn't enough these params have to
        also be moved every time since they differ
        '''
        
        modified_data = data[0].to(self.device)
        modified_rois = rois[0].to(self.device)
        modified_labels = labels[0].to(self.device)

        return self.model(modified_data, modified_rois, modified_labels)
    
    def detect(self, image: torch.Tensor, image_id: int) -> dict[str, np.ndarray]:
        '''
        Args:
            - `image`: float tensor (3, H, W), RGB, 0-1, no normalization, original
            VOC size, any device. This image will have some noise added to it though
            but this function doesn't modify the noise part at all.

            - `image_id`: int VOC 2007 ID (test or trainval)
        
        Returns:
            - `boxes`: float32 `(N, 4), x1 y1 x2 y2` top left/bottom right
            - `scores`: float32 `(N,)` confidence
            - `labels`: int64 `(N,)`, 0-19, name = `VOC_CLASSES[label]`
            
            - The above 3 are returned as a dictionary where boxes/scores/labels are keys and all detections are included. There could be cases sometimes when detection count is 0
        
        Raises:
            - `ValueError` for a bad image shape
            - `KeyError` if the image id passed in is non-existent
        '''

        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected shape (3, H, W), got {image.shape}")

        if image_id not in self.proposals_lookup:
            raise KeyError(f"{image_id} is not a valid VOC 2007 image ID")

        # drop gradient record, move tensor to CPU memory and convert
        np_converted = image.detach().cpu().numpy()

        # switches from (3, H, W) pytorch format -> (H, W, 3) np format
        transformed = np.transpose(np_converted, (1, 2, 0))

        # reverse the last axis (RGB -> BGR)
        bgr_array = transformed[..., ::-1]

        # scaling it since 0-1 values are what come from img to which noise is added
        im = (bgr_array * 255.0).astype(np.float32)

        # all the original proposal boxes: O(1) lookup
        target_boxes = self.proposals_lookup[image_id].astype(np.float32)

        # *************************** Proposal cleanup ***************************
        # clipping: boxes poking outside image get trimmed to the egde
        clipped_boxes = box_utils.clip_boxes_to_image(target_boxes, im.shape[0], im.shape[1])

        # removes exact duplicate boxes so model doesn't score same box twice
        keep = box_utils.unique_boxes(clipped_boxes)
        unique_boxes = clipped_boxes[keep, :]

        # drops boxes 2 pixels wide or smaller since no way they can contain an object
        keep = box_utils.filter_small_boxes(unique_boxes, 2)
        final_cleaned_boxes = unique_boxes[keep, :]

        initial_detections = im_detect_all(self.unwrap, im, final_cleaned_boxes)

        # for each proposal which is a candidate box from selective search
            # scores: 21 (how likely it is to be each class)
            # boxes: 21 (84 cols) adjusted onex (where it would be if it were each class)
        scores, boxes = initial_detections['scores'], initial_detections['boxes']

        '''
        [0] scores: (N,) every detections score
        [1] boxes: (N, 4) every detections box
        [2] cls_boxes: list of 21 2D arrays

        drop near 0 scores, NMS removes overlapping duplicates keeping highest score then 
        top 100 overall; output is confident non duplicate detections
        '''
        filtered_predictions = box_results_with_nms_and_limit(scores, boxes)
        scores, boxes, cls_boxes = filtered_predictions

        '''
        scores and boxes are for every single detection already
        sum of how many detections/class in cls_boxes equals len(scores) = len(boxes)

        cls_boxes has indices from 0->20 where [0] is background
        actual class names represented only from index 1 in cls_boxes
        basically: cls_boxes[j] <-> VOC_CLASSES[j-1]

        every detection in cls_boxes[1->20] is for one class so add that classes label
        to the labels array repeated those many times. labels are 0->19 indices into
        VOC_CLASSES
        '''
        labels = []
        for j in range(1, 21):
            labels.extend([j - 1] * len(cls_boxes[j]))

        return {
            "scores": scores,
            "boxes": boxes,
            "labels": np.array(labels, dtype=np.int64)
        }

if __name__ == "__main__":
    IMG_ID = 1
    IMG_PATH = ROOT / 'data/VOCdevkit/VOC2007/JPEGImages/000001.jpg'

    im = cv2.imread(IMG_PATH)
    assert im is not None

    # BGR -> RGB conversion
    rgb_array = im[..., ::-1].copy()

    converted_tensor = torch.from_numpy(rgb_array)

    # (H, W, 3) -> (3, H, W): 2, 0, 1
    permuted_tensor = converted_tensor.permute(2, 0, 1)

    # 0-255 -> 0-1
    t = permuted_tensor.float() / 255

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    pipeline = DetectPipeline(device)

    def run_detect(img: torch.Tensor, cloned_img: torch.Tensor, image_id: int) -> None:
        results = pipeline.detect(img, image_id)

        boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
        
        assert len(boxes) == len(scores) == len(labels)
        assert boxes.shape[1] == 4
        
        if not torch.equal(img, cloned_img):
            raise ValueError("The original tensor was somehow modified")
        
        idx = np.argmax(scores)
        
        assert scores[idx] >= 0.98
        
        assert labels[idx] == 14
        assert VOC_CLASSES[labels[idx]] == "person"

    run_detect(t, t.clone(), IMG_ID)

    gpu_img = t.to(device)
    run_detect(gpu_img, gpu_img.clone(), IMG_ID)

    print("Both runs successful...")