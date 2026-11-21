# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

Always reply to the user in Vietnamese (luôn trả lời người dùng bằng tiếng Việt). This covers explanations, summaries, questions and plans. Code identifiers, commands and file paths stay as they are.

## Project

This is an image-processing course project (đồ án Xử lý ảnh). It detects and tracks several people in video from a fixed camera. Later stages are planned but not built yet: door-crossing events, ORB/SIFT features, and Re-ID across cameras (cam_a ↔ cam_b). The README, docstrings, comments, config comments and error messages are all in Vietnamese. Keep them in Vietnamese, because some tests match parts of the Vietnamese messages (for example `match="Thiếu section"` and `"ngoài khung"`).

### Hard constraint: no YOLO, no deep learning, no machine learning
Use only classical image processing, estimation and combinatorial optimisation. That covers optical flow, morphology, connected components, histograms, RANSAC, Hungarian, Kalman and hand-crafted descriptors.
- HOG with OpenCV's pre-trained people SVM counts as ML. It was removed from the code (QĐ-04), so don't bring it back.
- The teacher allowed MOG2 on 2026-11-21. MOG2 is an adaptive per-pixel Gaussian-mixture background model.
  - Don't claim flatly that it is or isn't ML.
  - It detects foreground, not people.
  - The redesign implements both MOG2 and a median background that is updated selectively with `cv2.accumulateWeighted`, and picks the default from dev results.
- Don't add k-means, `createBackgroundSubtractorKNN`, classifiers, fitted weights, or labelling done by a model. `BFMatcher.knnMatch` is fine, because it is a nearest-neighbour search and not a classifier.
- Parameters are tuned by a manual grid search on dev data. Call this "hiệu chỉnh tham số", never "train" or "huấn luyện".

## Planned redesign (read before changing the architecture)

On 2026-11-21 the team approved a redesign that follows XLA.md. It is described in Vietnamese in `docs/`:
- `docs/de_xuat_phuong_an.md`: the target pipeline, module layout, roadmap and risks.
- `docs/quyet_dinh_thiet_ke.md`: the decision log (QĐ-01…). Add an entry for every new design decision.
- `docs/hoi_dap_bao_ve.md`: defense Q&A. Add questions whenever a module is finished.
- `docs/tu_dien_thuat_ngu.md`: a plain-Vietnamese glossary. The user has never studied image processing. When talking to them, explain technical terms in Vietnamese, and add every new technical term to this file.
- `docs/cau_hoi_thay.md`: questions for the teacher. All were answered on 2026-11-21:
  - Allowed: AI-assisted coding, MOG2, "auto SIFT", library functions instead of hand-written algorithms, Kalman, and the floor homography.
  - IN/OUT door counting is **required**.
  - At most 4 fixed cameras.
- `docs/ke_hoach_quay_video.md`: the plan for re-shooting with a tripod, two overlapping cameras and a floor homography.
- `docs/tim_video_thuc_te.md`: requirements, a checklist of corridor conditions, search keywords and public datasets for extra real test video from the internet. Put that video in `data/real/internet/<name>/cam_a/` together with a `nguon.txt` that records the source and licence.

Team choices (2026-11-21):
- **Cameras:** 2. Config and Re-ID code must still support up to 4, matching cameras pair by pair.
- **People:** 4 per scene.
- **"Auto SIFT":** compare three branches with equal keypoint counts: plain SIFT, `sift_auto` (lower `contrastThreshold` and retry), and ASIFT via `cv2.AffineFeature_create`. ASIFT was verified only on OpenCV 5.1.0, so pin that version.
- **Floor homography:** an auxiliary cue only. It assumes a near-planar floor and can be wrong, so always report appearance-only Re-ID results separately.

The code described under **Architecture** below is the **old** design: an LK-only tracker and a MOG2 detector. Phase 2 began on 2026-11-21 (QĐ-20). The user wants the system code built now and data work skipped. The old detector and tracker stay untouched so the report can compare old with new.

Groundwork that doesn't need real video was built on 2026-11-21. Each piece is a standalone, tested module with a doc in `docs/01_…06_*.md`:
- `src/homography.py` and `tools/calibrate_floor.py`: floor homography.
- `src/assignment.py`: shared Hungarian assignment (`assign`, `iou_cost`, `FORBIDDEN_COST`).
- `src/kalman.py`: `ConstantVelocityKalman`, configured by the `kalman` section.
- `src/door.py`: `DoorCounter` using foot point, signed distance and hysteresis. It is configured by `events.crossing_margin` and `events.inside_side`.
- `src/mot_metrics.py`: MOTA, MOTP and IDF1, also printed by `python -m src.metrics track`.
- Appearance features (step F):
  - `src/gallery.py`: picks the K best clean, non-border, well-spaced observations per track.
  - `src/keypoint_features.py`: normalised 128×256 crop plus mask, then SIFT, ORB, sift_auto or ASIFT capped at `features.max_keypoints`.
  - `src/color_features.py`: gray-world colour balance and upper/lower-body H-S histograms.
- Re-ID (step G):
  - `src/reid_similarity.py`: keypoint score (ratio test, vertical constraint, optional RANSAC), colour score, floor-geometry score, and the weighted combination.
  - `src/reid.py`: describes tracks, builds the score tables `ngoai_hinh` and `ngoai_hinh_san` (always report them separately), and assigns global IDs with Hungarian and `reid.match_threshold`.
  - `tools/run_reid.py` runs oracle mode on a synthetic scene and scores it with `src.metrics.reid_metrics` and `rank1`. `tools/explain_reid.py` draws the tables and keypoint matches. End-to-end mode waits for the new tracker.
- `src/door_matching.py`: links door events across two synchronised cameras by time and direction, using Hungarian on the time gap with `reid.door_event_max_gap_s` (QĐ-18).
- Synthetic video v2:
  - `tools/synthetic_scene.py`: classes and geometry (`Person`, `Layout`, `Scenario`).
  - `tools/synthetic_scenarios.py`: data (`LAYOUTS`, `SCENARIOS`).
  - `tools/synthetic_render.py`: drawing.
  - `tools/make_synthetic_v2.py`: the CLI.

The new end-to-end system (phase 2) wires these modules together. It is documented in `docs/10_tim_nguoi.md`, `docs/11_tracker_moi.md` and `docs/12_he_thong.md`:
- `src/background.py` and `src/detection.py`: step C. `PersonDetector` uses the median background with selective update (or MOG2), and is configured by the `detection` section.
- Step D, configured by the `people_tracker` section:
  - `src/people_tracker.py` holds the `PeopleTracker` state machine: tentative, confirmed, merged, static, lost.
  - `src/track.py`, `src/optical_flow.py`, `src/blob_analysis.py` and `src/recovery.py` are its parts.
  - The rules found by testing: implausible-size detections that overlap another track don't update position or size; split pieces are joined only when vertically stacked; during a merge only the front track (largest bottom y) keeps LK.
- `src/pipeline.py`: `process_camera` runs C → D → E for one camera and writes `detections.csv` and `tracks.csv`, plus `debug.mp4` with `--debug-video`.
- `src/system_eval.py`: end-to-end scoring. It maps each track to a GT ID by majority IoU vote, uses the main track per person, and scores door events and Re-ID.
- `run.py --scene <dir> [--out] [--debug-video]` runs both cameras, Re-ID, door links and the summary, writing to `outputs/he_thong/<scene>/`.
- `src/metrics.py` now scores track states `confirmed`, `merged` and `static` (`SCORED_TRACK_STATES`).

The team writes code with AI help and has to defend every line in front of the teacher. Rules for new code:
- One concept per file, under about 200 lines, with short functions.
- Never put several statements on one line with `;`.
- Docstrings in Vietnamese that explain *why* and give the formula.
- Every threshold lives in config, with a comment giving its meaning and unit.
- Keep diagnostics and logging out of the algorithm code.
- A finished module ships with three things: `docs/NN_<module>.md` (problem → algorithm → formula → parameters → failure cases → likely teacher questions), an `--explain` visualisation of each step, and new entries in `docs/hoi_dap_bao_ve.md`. After that, explain the module to the user in chat in plain Vietnamese and ask only the 1–2 core questions they must be able to answer at the defense (for example "what does this step do and why" or "is this ML"). Skip detail questions.

## Commands

Run everything from the repo root inside `.venv` (Python 3.14, OpenCV 5.1). The README asks for Python ≥3.10.

```bash
python -m pytest                       # use `python -m`: plain `pytest` can't import `src`
python -m pytest tests/test_tracker_blob.py
python -m pytest tests/test_tracker.py::test_two_separated_objects_keep_distinct_ids

python run_detector.py --video data/synthetic/cross/cam_a/video.mp4 --out outputs/detections.csv --debug-video outputs/detector_debug.mp4
python run_tracking.py --video data/synthetic/cross/cam_a/video.mp4 --out outputs/tracks.csv --debug-video outputs/tracking_debug.mp4
python -m src.metrics det   outputs/detections.csv data/synthetic/cross/gt_tracking.csv
python -m src.metrics track outputs/tracks.csv     data/synthetic/cross/gt_tracking.csv [--iou-thr 0.5]

# Tracking run that also writes JSONL diagnostics and a provenance manifest; it never overwrites existing files
python tools/trace_tracking.py --video <video> --gt <gt.csv> --tracks-out outputs/<run>/tracks.csv --diagnostics-out outputs/<run>/diagnostics.jsonl

python -m tools.make_synthetic --scenario {cross,occlude,stop} --out data/synthetic/<name> [--frames N --seed S --write-gt-reid]  # must use -m
python tools/annotate.py  --video <video> --out <gt.csv> [--size 640 360]   # opens an interactive OpenCV window; key o cycles occlusion
python tools/verify_gt.py --video <video> --gt <gt.csv> --out outputs/gt_check.mp4 [--size 640 360]

# Redesign groundwork (standalone, not yet wired into the tracker)
python -m tools.make_synthetic_v2 --out data/synthetic_v2 [--scenario s1 s2 s3 s4 door] --explain outputs/explain/video_tong_hop   # ~1 min
python tools/calibrate_floor.py --cam-a <video_a> --cam-b <video_b> --out <homography.json> [--explain <png>]
python -m tools.explain_assignment    # writes outputs/explain/ghep_cap/
python -m tools.explain_kalman        # writes outputs/explain/kalman/
python -m tools.explain_door          # writes outputs/explain/dem_qua_cua/
python -m tools.explain_door_matching # writes outputs/explain/ghep_su_kien_cua/
python run.py --scene data/synthetic_v2/hanh_lang_cua [--debug-video]   # WHOLE SYSTEM, both cameras, writes outputs/he_thong/<scene>/summary.json
python -m tools.run_reid --scene data/synthetic_v2/s4 [--method sift|orb|sift_auto|asift]   # oracle Re-ID + scores
python -m tools.explain_reid --scene data/synthetic_v2/s4 --person 3   # writes outputs/explain/nhan_dang_lai/
```

The video in `data/synthetic_v2` is 640×360, but `config.yaml` still resizes to 640×480. So pass `--size 640 360` to `verify_gt.py` and `annotate.py` when you use them on that data.

No linter, formatter or type checker is set up.

## Architecture

This section describes the code as it is now, which is the old design (see **Planned redesign** above). Every entry point runs the same chain:
`utils.iter_video` (resizes each frame once to `scene.resize_width × resize_height`) → `Detector.detect(frame_bgr)` → `Tracker.update(gray, detections, frame_bgr=frame)` → CSV → `src.metrics`.

Every box is `(x, y, w, h)` in integer pixels of the **resized** frame. All pixel thresholds in `config.yaml` are measured at that resolution.

### Config
- `load_config` checks that the sections `scene, detector, tracker, events, features, reid, benchmark` exist, and checks the keys inside `scene`. It returns an `AppConfig` in which every section except `scene` is a raw dict. `SceneConfig` drops `ignore_regions`, so the entry points read the YAML a second time and pass `Detector({"detector": cfg.detector, "scene": raw["scene"]})`.
- `metrics` is not a section of `AppConfig`. The `src/metrics.default_*` functions read `config.yaml` from the repo root directly and ignore `--config`.
- `Detector` and `Tracker` accept either a Mapping or an object with attributes. Each checks every key itself and has **no defaults**. When you add a `tracker.*` key, also add it to the config dicts built by the helpers (`config()`, `cfg()`, `_cfg()`) at the top of each `tests/test_tracker*.py`. Otherwise those tests raise `ValueError: Thiếu tracker.<key>`.
- The `events`, `features`, `reid` and `benchmark` sections have no code that reads them yet. Door counting, ORB/SIFT, Re-ID and benchmarking are not implemented. Re-ID scoring already exists in `src/metrics` (`reid_metrics`, `rank1`, `confusion_matrix`).

### Detector (`src/detector.py`)
The steps run in this order:
1. Gaussian blur.
2. MOG2 with `detectShadows=True`. Only pixels equal to 255 are kept, which removes shadow pixels (127).
3. The ROI and ignore regions are masked out.
4. Opening, closing and dilation.
5. The ROI and ignore regions are masked out again.
6. Connected components are found and filtered by area and by h/w ratio.

`get_foreground_mask()` returns the most recent processed mask.

### Tracker (`src/tracker.py`): one `update()` call, in order
1. The previous frame's boxes are saved in `_ownership_snapshot`.
2. Each track is predicted with pyramidal LK and a forward-backward check. A point pair is kept only if its *previous* position was inside the track's own saved box and inside no neighbour's box. The box then moves by the median displacement of the kept points. If too few points remain, `_extrapolate_exit` continues the last motion, but only when the track is heading out of the frame. Otherwise the box does not move.
3. `_classify_detections` runs **before** any matching. It labels each detection as one of three kinds:
   - `merged`: the detection contains two or more well-separated predicted tracks. This uses coverage or IoU, together with whether each track's centre lies inside the detection.
   - `ambiguous`: more than one track may be involved, but the `merged` test fails.
   - `clean`: everything else.

   Only clean detections can update a box or a histogram, or start a new track.
4. Recovery: confirmed tracks that were occluded in the previous frame are matched against clean detections. Hungarian runs on the cost `1 − HSV_corr × IoU`, and a pair must pass both `hist_corr_min` and `iou_gate`. If a clean detection looks like an occluded track but fails the IoU gate, it is held back (`_unresolved_recovery_detections`) so that it doesn't get a new ID.
5. Normal matching: the remaining tracks and clean detections are matched by Hungarian on `1 − IoU`, or greedily when `matcher: greedy`. Forbidden pairs get a cost of `1e6`, and the gate is checked again after assignment. Don't use `np.inf`.
6. Lifecycle:
   - A track moves from `tentative` to `confirmed` after `confirm_hits` hits in a row.
   - A confirmed track inside a merged blob is marked `occluded` and moves only by its LK prediction. It is kept for up to `blob_grace_frames` frames after the blob ends.
   - Any other track becomes `lost` after `max_misses` missed frames, or as soon as it leaves the frame.
   - IDs only increase and are never reused.
7. LK points are refreshed with `goodFeaturesToTrack`, searching the track's own box minus the neighbours' predicted boxes. A refresh happens every `refresh_every` frames, or when the track has fewer than `min_points` points. Tracks that are merged, ambiguous or were occluded in the previous frame are skipped.

`update()` returns the active tracks **plus** the tracks lost in this frame. That is why `tracks.csv` contains `lost` rows. Metrics count only rows with `state == confirmed`.

### Diagnostics
`Tracker._diagnostic` sends one JSON event at a time to the logger `src.tracker` at INFO level. It does nothing unless INFO is enabled. The events are:
- `frame_start`
- `prediction`
- `detection_classification`
- `recovery_matrix`
- `association_matrix`
- `normal_assignment_result`
- `point_refresh`
- `track_created`
- `track_lifecycle`
- `ownership_replenishment`

`tools/trace_tracking.py` writes these events to JSONL, after a first `run_manifest` record. The manifest holds the git HEAD, a hash of the diff, a copy of the config, and SHA-256 hashes of the source and input files. `tests/test_tracker_diagnostics.py` checks the event fields, so update it whenever you change an event's payload.

### Tests
The tests build textured frames with NumPy and pass in exact ("oracle") detections. The blob tests replace `tracker._predict_track` so that predictions stay fixed and only the matching logic is tested. `test_tracker_crossing.py` is an end-to-end acceptance test, scored with `tracking_metrics`.

## Data and outputs

`src/metrics.read_csv_rows` checks that each CSV has these columns:

| File | Columns |
|---|---|
| detections | `frame,x,y,w,h,area` |
| tracks | `frame,track_id,state,x,y,w,h,n_points` |
| gt_tracking | `frame,gt_id,x,y,w,h` plus, in the new format (QĐ-15), `occluded` (0/1/2) and `keyframe` (1 = hand-labelled) |
| gt_events (synthetic v2) | `frame,gt_id,direction` |
| gt_reid | `person,cam_a_id,cam_b_id` |
| global_ids | `cam,track_id,global_id` |
| reid scores | `cam_a_track,cam_b_track,method,score` |

- Synthetic data lives in `data/synthetic/<scenario>/{cam_a,cam_b}/{video.mp4,metadata.json}` with a `gt_tracking.csv` next to the camera folders. That ground truth covers **cam_a only**. Synthetic video is 640×480 at 20 fps.
- `detection_pr` checks whether the ground truth has the new columns. If it does, it scores only frames where every box is hand-labelled and none has `occluded == 2` (`scored_frames`). Old-format files are scored on every frame, as before.
- Synthetic v2 has two layouts. `san_chung` (QĐ-02) is used by s1–s4 and door. `hanh_lang` (QĐ-17) is a corridor with a side door: cam_a looks down the corridor and cam_b faces the door. It is used by `hanh_lang_cua`.
- Synthetic v2 data lives in `data/synthetic_v2/<s1|s2|s3|s4|door|hanh_lang_cua>/`:
  - `{cam_a,cam_b}/` holds `video.mp4`, `gt_tracking.csv` (one per camera; each box covers the visible part only) and `metadata.json`.
  - `gt_reid.csv` and `gt_events.csv` are shared by both cameras.
  - `scene.json` holds the floor→image homography of each camera, `H_ab`, the door line and inside side in image coordinates, and the tape-mark positions.
  - The video is 640×360 at 30 fps. The first 2 seconds show an empty scene.
  - `s2` has one frame where person 1 is fully hidden, so `verify_gt` reports a "lỗ hổng frame" warning for that person. That warning is expected.
- `data/real/dev/S1/cam_a/video.mp4` is real footage at 1024×576 and 30 fps. Resizing it to the 640×480 `scene` size changes its aspect ratio. Ground truth and thresholds for it must use resized coordinates.
- `outputs/` is git-ignored except for `outputs/README.md`, which describes each folder. Earlier runs are grouped as `outputs/synthetic_d2/NN_*` (in time order) and `outputs/real_dev_S1/{base,lr01,r1}`. Put each new run in its own `outputs/<stage>/<NN_name>/` folder, together with the config it used.
- `.gitignore` covers `.venv/`, Python caches, `outputs/*` and Windows `*:Zone.Identifier` files. `data/real/` is not ignored, but it isn't committed yet either.

## Review findings not yet in the code (as of 2026-11-21; check before relying on them)
These come from the roadmap review (XLA.md, which is not in this repo):
- When LK points are refreshed, the search mask is the track's box minus the neighbours' boxes. It is not intersected with the foreground mask, so corners in the background can drag a box toward zero motion.
- There is no `static` state. When MOG2 absorbs a person who stands still into the background, the person stops producing detections and their track is dropped after `max_misses`, even though LK is still following them.
- Recovery after occlusion relies only on the HSV histogram. It fails when two people wear the same colour (scenario S4). The review suggests adding motion continuity, NCC template matching and a gallery of keypoints.
- For the planned ORB/SIFT work:
  - Pass an eroded foreground mask through the detector's `mask` argument. Don't black out the crop, because that creates fake keypoints along its edge.
  - Lower ORB's `edgeThreshold` and `patchSize` for crops about 128 px wide.
  - Measure memory with RSS or `ru_maxrss`, running each configuration in its own process. Don't use `tracemalloc`.
