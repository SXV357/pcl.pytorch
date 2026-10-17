# OICR `detect()` Plan: Fri 10/16 → Sun 10/18

Goal: get Kanad and Utkarsh a working `detect()` they can call from the RL code, plugged into the main project repo. Then check that it's correct (subset mAP) and start the eval script.

The repo plumbing is already done: the port is merged to `master` (PR #1) and `tools/infer_one.py` works end to end. Everything below is reorganizing that work plus a few new pieces.

---

## Assumptions (state these in the docstring and in the handoff message)

| Topic | Assumption | Status |
|---|---|---|
| Input image | float tensor `(3, H, W)`, RGB, values 0–1, no normalization, original VOC size, any device | ✅ Utkarsh confirmed |
| Image ID | `detect(image, image_id)`; proposals looked up by VOC ID | assumed, Kanad said "up to u" |
| Output | dict of numpy arrays: `boxes (N,4)` x1 y1 x2 y2 in pixels (top-left, bottom-right), `labels (N,)`, `scores (N,)`. All detections, **no cutoff**; caller filters | assumed |
| Device | model forward on `device` (`cuda`/`cpu`); pre/post-processing stays numpy on CPU | Kanad: "use numpy cuda" |
| Splits | agent trains on VOC07 **trainval** (5,011), eval on VOC07 **test** (4,952) | assumed, matches how OICR was trained |
| Test setting | single scale, no flip (`TEST.BBOX_AUG.ENABLED = False`) | your choice. Clean mAP will be a bit below the published 52.9 |

---

## Phase 0: Prep (Fri, ~30 min)

- [ ] Download `voc_2007_trainval.pkl` from the same Drive proposals folder into `data/selective_search_data/`.
- [ ] Download and untar `VOCtrainval_06-Nov-2007.tar` (same server/mirror as the test tar). It merges into the same `data/VOCdevkit/VOC2007/`.
- [ ] Sanity check: trainval pkl has 5,011 `indexes`, and no ID appears in both pickles.
- [ ] Create a branch, e.g. `git checkout -b detect-wrapper`.

---

## Phase 1: `detect()` v1 (Fri, 1 evening)

Mostly moving `infer_one.py` code into a class.

### Where it lives
Put the wrapper **in this fork** (e.g. `tools/` or a new top-level file), so the main repo imports **one** entry point. Only the wrapper file should add `lib/` to `sys.path`. The repo's generic package names (`utils`, `core`, `datasets`, `nn`, `ops`, `modeling`) collide with other code if they leak out.

### `__init__` (runs once)
- [ ] Inputs: yaml path, checkpoint path, proposal pkl path(s), device.
- [ ] Merge yaml → 3 overrides (`NUM_CLASSES=20`, `LOAD_IMAGENET_PRETRAINED_WEIGHTS=False`, `BBOX_AUG.ENABLED=False`) → `assert_and_infer_cfg()`.
- [ ] Build `Generalized_RCNN`, load the checkpoint, `.eval()`, move to `device`.
- [ ] Load **both** pickles once. Build a lookup from image ID to boxes, so each call doesn't run `list.index()` over ~5k entries.
- [ ] Keep the class-name list here.

### `detect(image, image_id)` (runs every call)
1. [ ] **Validate**: shape is `(3, H, W)` and `image_id` exists, with clear error messages.
2. [ ] **Convert the format**: tensor → CPU → numpy, `(3,H,W)` → `(H,W,3)`, RGB → BGR, ×255.
   - Keep it **float**. Do **not** cast to uint8; rounding erases noise smaller than 1/255.
   - Do **not** apply mean/std. The repo subtracts its own BGR `PIXEL_MEANS` (`config.py:434`) inside `prep_im_for_blob` (`blob.py:92`), which assumes BGR 0–255.
3. [ ] **Proposals**: look up by ID, cast to float32, clip / unique / filter small (min size 2). Same as `infer_one.py`.
4. [ ] **Forward**: `im_detect_all` with your `unwrap`. Change: `unwrap` must move the three tensors to `device` before calling the model (`test.py:102-104` creates them on CPU).
5. [ ] **NMS**: `box_results_with_nms_and_limit` → `cls_boxes`.
6. [ ] **Flatten**: loop classes 1–20 and stack into `boxes`, `labels`, `scores`.
   - Zero detections → `boxes` shape `(0,4)`, the others `(0,)`. Never `[]` or `None`.
   - Labels: pick int (0–19 or 1–20) or name and document it.
7. [ ] Wrap the call in `torch.no_grad()` (black-box RL, no gradients needed).

### Check 1: does it match the script?
- [ ] Load `000001.jpg` with cv2 → convert to RGB, 0–1, `(3,H,W)` tensor (i.e., simulate Utkarsh's input) → `detect(img, 1)`.
- [ ] Top person ≈ **0.99**, top dog ≈ **0.84**, same boxes as `infer_one.py`. If the numbers differ, the format conversion is wrong (most likely a BGR/RGB or scale mistake).
- [ ] Try a trainval ID too (proves the second pickle lookup works).
- [ ] Bad `image_id` → your error message, not a raw traceback.

### Check 2: device
- [ ] No CUDA on the Mac → test `device="cpu"` locally.
- [ ] Test `device="cuda"` once on Colab (or ask Kanad to run Check 1). The number of detections should match; scores can differ by about 1e-5.

---

## Phase 2: Into the main project repo (Sat morning)

- [ ] Push the branch to the fork, open a PR, merge to `master`.
- [ ] In the main project repo: `git submodule add <fork url> <path, e.g. third_party/oicr>`, and commit the `.gitmodules` + pointer.
- [ ] README section (short):
  - clone with `--recurse-submodules` (or `git submodule update --init`)
  - env setup: Python 3.11, `torch torchvision opencv-python "numpy<2" scikit-learn pyyaml cython six pycocotools`, and build `cython_bbox` (`cd lib && python setup.py build_ext --inplace`)
  - downloads: checkpoint → `Outputs/`, both pkls, VOC07 test + trainval tars
  - a 5-line usage example of `detect()` and the output format
- [ ] Remember: updating the fork means you **also** commit the new submodule pointer in the main repo.
- [ ] Send the handoff message with the assumptions table and "tell me what to change."

---

## Phase 3: Subset mAP (Sat–Sun)

Checks that what you handed off is correct. Builds most of the eval script.

- [ ] Pick a fixed list of ~100 test IDs (save it to a file so it's reproducible).
- [ ] Loop `detect()` over them.
- [ ] Write one results file per class in the format `voc_eval` expects. See `_write_voc_results_files` (`voc_dataset_evaluator.py:63`) for the line format (`image_id score x1 y1 x2 y2`) and the **+1** on coordinates (VOC is 1-based).
- [ ] Write an imageset file listing your ~100 IDs (six-digit, one per line).
- [ ] Call `voc_eval(detpath, annopath, imagesetfile, classname, cachedir, ovthresh=0.5, use_07_metric=True)` (`voc_eval.py:86`) for each class, then average the 20 APs.
- [ ] Expect **~45–55**. ~10–20 means a bug: check the conversion, then the +1, then the class order.
- [ ] Later (Colab/Kanad GPU): full 4,952 test images → the real clean baseline.

---

## Phase 4: Eval script + reward list (Sun / next week)

- [ ] Eval script = Phase 3 run twice: clean images and perturbed images (same IDs, same proposals). Report the mAP drop and the noise size (L2, L∞ in 0–1 pixel units).
- [ ] Reward list (write it yourself; run it past me to check reasoning). Questions to answer per candidate:
  - What does "the attack worked" mean: label gone, score down, or box moved (H3)?
  - How do you penalize noise size?
  - Does it give a signal per image/step (mAP doesn't)?
  - How could the agent cheat it?

---

## Known pitfalls

- **cv2 `rectangle`** takes two corner tuples, `(x1,y1)` and `(x2,y2)`. One 4-tuple is read as `(x, y, w, h)`.
- **Proposal position ≠ image ID**: always look up by ID.
- **Global `cfg`**: one detector config per Python process. A second WSOD model (PCL/MIST) means a separate process.
- **Single-scale vs 52.9**: always say which setting a number comes from.
- **Course policy**: you write all the code; AI explains, points and reviews reasoning. Disclose AI-assisted setup and explanations.

## Done = 
- [ ] `detect()` matches `infer_one.py` on image 1 (CPU) and runs on CUDA
- [ ] Submodule in the main repo, README updated, handoff message sent
- [ ] Subset mAP number in hand (or a known bug being chased)
