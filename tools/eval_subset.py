import random
import argparse
import cv2
import torch

from detect_pipeline import ROOT, VOC_CLASSES, DetectPipeline
from datasets.voc_eval import voc_eval

parser = argparse.ArgumentParser(description="Script to compute subset/full VOC 2007 test set mAP")
parser.add_argument("--full", action="store_true")
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

args = parser.parse_args()

# deterministic
random.seed(42)

IMAGE_ID_PATH = ROOT / 'data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
with open(IMAGE_ID_PATH, 'r') as tf:
    test_ids = tf.read().splitlines()

if not args.full:
    IMAGE_ID_PATH = ROOT / 'data/VOCdevkit/VOC2007/ImageSets/Main/test100.txt'

    # sample 100 random test IDs with fixed seed
    test_ids = random.sample(test_ids, 100)
    test_ids.sort()
    
    with open(IMAGE_ID_PATH, 'w') as stf:
        for img_id in test_ids:
            stf.write(f"{img_id}\n")

pipeline = DetectPipeline(args.device)

cl_mappings = {}
for cl in VOC_CLASSES:
    cl_mappings[cl] = []

for iid in test_ids:
    path = ROOT / f"data/VOCdevkit/VOC2007/JPEGImages/{iid}.jpg"

    im = cv2.imread(path)
    assert im is not None
    
    # BGR -> RGB conversion
    rgb_array = im[..., ::-1].copy()
    
    converted_tensor = torch.from_numpy(rgb_array)
    
    # (H, W, 3) -> (3, H, W): 2, 0, 1
    permuted_tensor = converted_tensor.permute(2, 0, 1)
    
    # 0-255 -> 0-1
    t = permuted_tensor.float() / 255

    results = pipeline.detect(t, int(iid))

    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box

        cl_mappings[VOC_CLASSES[label]].append(f"{iid} {score:.3f} {x1 + 1:.1f} {y1 + 1:.1f} {x2 + 1:.1f} {y2 + 1:.1f}")

path = ROOT / "Outputs/subset_eval" if not args.full else ROOT / "Outputs/full_eval"
path.mkdir(parents=True, exist_ok=True)

total_ap = 0
for k in cl_mappings:
    with open(f"{path}/det_{k}.txt", "w") as f:
        for det in cl_mappings[k]:
            f.write(f"{det}\n")

    _, _, ap = voc_eval(
        detpath=str(path / "det_{}.txt"),
        annopath=str(ROOT / "data/VOCdevkit/VOC2007/Annotations/{}.xml"),
        imagesetfile=str(IMAGE_ID_PATH),
        classname=k,
        cachedir=str(path / "annots_cache"),
        use_07_metric=True
    )

    print(f"AP for class {k}: {ap}")

    total_ap += ap

print(f"mAP: {total_ap / len(VOC_CLASSES)}")