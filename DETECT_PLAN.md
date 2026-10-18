# OICR `detect()` Plan: Sat 10/17 → Sun 10/18

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

## Phase 0: Prep ✅ done Sat 10/17

- [x] Download `voc_2007_trainval.pkl` from the same Drive proposals folder into `data/selective_search_data/`.
- [x] Download and untar `VOCtrainval_06-Nov-2007.tar` (same server/mirror as the test tar). It merges into the same `data/VOCdevkit/VOC2007/`.
- [x] Sanity check: trainval pkl has 5,011 `indexes`, and no ID appears in both pickles.
  - Verified: 9,963 JPEGs + 9,963 XMLs; `trainval.txt` (5,011) and `test.txt` (4,952) IDs exactly match each pickle's `indexes`.
- [x] Create a branch (`pipeline`).

---

## Phase 1: `detect()` v1 ✅ done Sat 10/17

Mostly moving `infer_one.py` code into a class.

### Where it lives ✅ `tools/detect_pipeline.py`
Put the wrapper **in this fork** (e.g. `tools/` or a new top-level file), so the main repo imports **one** entry point. Only the wrapper file should add `lib/` to `sys.path`. The repo's generic package names (`utils`, `core`, `datasets`, `nn`, `ops`, `modeling`) collide with other code if they leak out.

### `__init__` (runs once) ✅ done Sat 10/17
- [x] Inputs: `device` only. Paths are module-level constants built from `ROOT = Path(__file__).resolve().parent.parent`; `lib/` is added to `sys.path` at module level, replacing `_init_paths`.
- [x] Merge yaml → 3 overrides (`NUM_CLASSES=20`, `LOAD_IMAGENET_PRETRAINED_WEIGHTS=False`, `BBOX_AUG.ENABLED=False`) → `assert_and_infer_cfg()`.
- [x] Build `Generalized_RCNN`, load the checkpoint, `.eval()`, move to `device`.
- [x] Load **both** pickles once. Build a lookup from image ID to boxes, so each call doesn't run `list.index()` over ~5k entries.
- [x] Class-name list as module constant `VOC_CLASSES`.
- [ ] Class docstring: add "one instance per process (global cfg is locked)" + single-scale note. *(device type hint done)*

### `detect(image, image_id)` (runs every call) ✅
1. [x] **Validate**: shape is `(3, H, W)` and `image_id` exists, with clear error messages.
2. [x] **Convert the format**: tensor → CPU → numpy, `(3,H,W)` → `(H,W,3)`, RGB → BGR, ×255.
   - Keep it **float**. Do **not** cast to uint8; rounding erases noise smaller than 1/255.
   - Do **not** apply mean/std. The repo subtracts its own BGR `PIXEL_MEANS` (`config.py:434`) inside `prep_im_for_blob` (`blob.py:92`), which assumes BGR 0–255.
3. [x] **Proposals**: look up by ID, cast to float32, clip / unique / filter small (min size 2). Same as `infer_one.py`.
4. [x] **Forward**: `im_detect_all` with your `unwrap`. Change: `unwrap` must move the three tensors to `device` before calling the model (`test.py:102-104` creates them on CPU).
5. [x] **NMS**: `box_results_with_nms_and_limit` → `cls_boxes`.
6. [x] **Flatten** → returns dict: `boxes` float32 (N,4), `scores` float32 (N,), `labels` int64 (N,) 0–19: loop classes 1–20 and stack into `boxes`, `labels`, `scores`.
   - Zero detections → `boxes` shape `(0,4)`, the others `(0,)`. Never `[]` or `None`.
   - Labels: pick int (0–19 or 1–20) or name and document it.
7. [x] ~~`torch.no_grad()`~~ not needed: in this repo `.eval()` already turns off gradient tracking in `forward`.

### Check 1: does it match the script? ✅ (your `__main__` checks person on image 1; dog 0.843 + trainval image 5 verified in a throwaway run)
- [x] Load `000001.jpg` with cv2 → convert to RGB, 0–1, `(3,H,W)` tensor (i.e., simulate Utkarsh's input) → `detect(img, 1)`.
- [x] Top person ≈ **0.99**, top dog ≈ **0.84**, same boxes as `infer_one.py`. If the numbers differ, the format conversion is wrong (most likely a BGR/RGB or scale mistake).
- [x] Try a trainval ID too (image 5 → chair 0.968) (proves the second pickle lookup works).
- [ ] Bad `image_id` → your error message, not a raw traceback.

### Check 2: device
- [x] No CUDA on the Mac → test `device="cpu"` locally. (~1 s/image)
- [x] ✅ Colab T4: CPU-input and GPU-input runs both pass. Originally: test `device="cuda"` once on Colab (or ask Kanad to run Check 1). The number of detections should match; scores can differ by about 1e-5.

---

## Phase 2: Into the main project repo (Sun 10/18, first)

- [x] Push the branch to the fork, open a PR, merge to `master`. (fork `master` contains `e12bf61`, verified Sun 10/18)
- Main project repo = `SXV357/CS557-WSOD-Neural-Perturbation` (local: `~/Desktop/CS 557/WSOD`, branch `main`, only README + .gitignore so far).
- [ ] In the main project repo: `git submodule add <fork url> <path, e.g. third_party/oicr>`, and commit the `.gitmodules` + pointer.
- Tested environments (for README):
  | | Python | numpy | torch | torchvision | opencv |
  |---|---|---|---|---|---|
  | Mac M2, CPU (`pcl` conda env) | 3.11.17 | 1.26.4 | 2.14.1 | 0.29.1 | 4.11.0 |
  | Colab T4, CUDA | Colab default | 2.1.3 | 2.11.0+cu130 | 0.26.0+cu130 | 4.14.0 |

  Colab needed only `pip install pycocotools` + the `cython_bbox` build; no numpy downgrade.
- [ ] README section (short):
  - clone with `--recurse-submodules` (or `git submodule update --init`)
  - env setup: Python 3.11, `torch torchvision opencv-python "numpy<2" scikit-learn pyyaml cython six pycocotools`, and build `cython_bbox` (`cd lib && python setup.py build_ext --inplace`)
  - downloads: checkpoint → `Outputs/`, both pkls, VOC07 test + trainval tars
  - a 5-line usage example of `detect()` and the output format
- [ ] Remember: updating the fork means you **also** commit the new submodule pointer in the main repo.
- [ ] Send the handoff message with the assumptions table and "tell me what to change."

---

## Phase 3: Subset mAP baseline (Sun 10/18, after Phase 2)

Checks that what you handed off is correct. Builds most of the eval script.

- [ ] Pick a fixed list of ~100 test IDs (save it to a file so it's reproducible).
- [ ] Loop `detect()` over them.
- [ ] Write one results file per class in the format `voc_eval` expects. See `_write_voc_results_files` (`voc_dataset_evaluator.py:63`) for the line format (`image_id score x1 y1 x2 y2`) and the **+1** on coordinates (VOC is 1-based).
- [ ] Write an imageset file listing your ~100 IDs (six-digit, one per line).
- [ ] Call `voc_eval(detpath, annopath, imagesetfile, classname, cachedir, ovthresh=0.5, use_07_metric=True)` (`voc_eval.py:86`) for each class, then average the 20 APs.
- [ ] Expect **~45–55**. ~10–20 means a bug: check the conversion, then the +1, then the class order.
- [ ] Later (Colab/Kanad GPU): full 4,952 test images → the real clean baseline.

---

## Phase 4: Eval script + reward list (week of 10/19)

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
