# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

Always reply to the user in Vietnamese (luôn trả lời người dùng bằng tiếng Việt). This covers explanations, summaries, questions and plans. Code identifiers, commands and file paths stay as they are.

## Project

This is an image-processing course project (đồ án Xử lý ảnh). It detects and tracks several people walking in a corridor, filmed by two fixed cameras, and re-identifies them across the cameras (cam_a ↔ cam_b). Door IN/OUT counting was dropped (QĐ-21). The README, docstrings, comments, config comments and error messages are all in Vietnamese. Keep them in Vietnamese, because some tests match parts of the Vietnamese messages (for example `match="Thiếu section"` and `"ngoài khung"`).

### Hard constraint: real video only, never computer-generated images or video (QĐ-31)
- Never create, run on, tune on, test on or report computer-drawn images or video. The user made this a hard rule on 2026-11-23. All synthetic data, generators (`tools/make_synthetic*`, `tools/synthetic_*`), synthetic tests and `docs/01_video_tong_hop.md` were deleted; they survive only in git history.
- Tests that need images read real video from `data/canh/` (see `tests/du_lieu_that.py` and `tests/conftest.py`). They skip themselves when the data is missing. Tests on pure numbers (boxes, cost matrices, formulas) are fine. Never build test frames with NumPy drawing, noise or colour patches.
- Use only the real videos the user supplies (QĐ-32): currently the EPFL lab-4p avi originals in `data/goc/epfl_lab4p/` (4 cameras, 360×288, 25 fps, 3915 frames). On 2026-11-23 the user had the downloaded EPFL 6p and Passageway sets (with their ground truth) and the old S1 clip deleted. Never download or add other videos or datasets unless the user asks at that moment.
- Without ground truth, judge changes with the QĐ-34 scorecard (`src/evaluation/khong_dap_an.py`), never with the ID count alone: a change is kept only if IDs drop while coverage (`phu_nguoi`) does not drop and ghosts (`track_ma`) and jumps (`nhay`) do not rise. Tune on scene `epfl_lab4p_c0_c1` and confirm on `epfl_lab4p_c2_c3`. MOG2 learns fast for the first `mog2_warmup_frames` frames (QĐ-33), then slowly.
- No video has ground truth yet, so MOTA, IDF1 and Re-ID accuracy cannot be computed; `run.py` and `tools/hieu_chinh.py` only report ID and track counts. Real scores need the team to hand-label keyframes with `tools/annotate.py`; never let a model label them. Tuning that relied on synthetic scores (QĐ-25 … QĐ-29) must be re-validated once ground truth exists.
- The tests are ground-truth free and use the 4p videos: box height vs the height the EPFL calibration expects (median ratio 0.90), ID counts, and colour similarity on hand-picked real boxes (`tests/du_lieu_that.py`).

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
  - IN/OUT door counting is **required**. But the team **dropped it on 2026-11-22 (QĐ-21)**, against the teacher's answer. The door code is preserved in commit `827121c` on `redesign`. Don't re-add it unless the user asks.
  - At most 4 fixed cameras.
- `docs/ke_hoach_quay_video.md`: the plan for re-shooting with a tripod, two overlapping cameras and a floor homography.
- `docs/13_noi_tracklet.md`: tracklet linking.
- `docs/tim_video_thuc_te.md`: requirements, a checklist of corridor conditions, search keywords and public datasets for extra real test video from the internet. Put that video in `data/video_moi/`, then build a scene `data/canh/internet_<name>/` with `tools/tao_canh.py` and add a `nguon.txt` that records the source and licence.

Team choices (2026-11-21):
- **Cameras:** 2. Config and Re-ID code must still support up to 4, matching cameras pair by pair.
- **People:** 4 per scene.
- **"Auto SIFT":** compare three branches with equal keypoint counts: plain SIFT, `sift_auto` (lower `contrastThreshold` and retry), and ASIFT via `cv2.AffineFeature_create`. ASIFT was verified only on OpenCV 5.1.0, so pin that version.
- **Floor homography:** an auxiliary cue only. It assumes a near-planar floor and can be wrong, so always report appearance-only Re-ID results separately.

Layout (reorganised 2026-11-22): the new system lives in `src/` split by step (`src/common`, `src/detect`, `src/tracking`, `src/reid`, `src/evaluation`, plus `src/pipeline.py`); the old system lives in the top-level `legacy/` package (`legacy/detector.py`, `legacy/tracker.py`, `legacy/config.py`, run with `python -m legacy.run_detector` / `python -m legacy.run_tracking`) and its tests in `tests/legacy/`. `tools/` and `docs/` stay flat; `docs/README.md` is the docs index.

The code described under **Architecture** below is the **old** design: an LK-only tracker and a MOG2 detector. Phase 2 began on 2026-11-21 (QĐ-20). The old detector and tracker stay untouched so the report can compare old with new.

Groundwork built on 2026-11-21. Each piece is a standalone, tested module with a doc in `docs/02_…09_*.md`:
- `src/common/homography.py` and `tools/calibrate_floor.py`: floor homography.
- `src/common/assignment.py`: shared Hungarian assignment (`assign`, `iou_cost`, `FORBIDDEN_COST`).
- `src/common/kalman.py`: `ConstantVelocityKalman`, configured by the `kalman` section.
- `src/evaluation/mot_metrics.py`: MOTA, MOTP and IDF1, also printed by `python -m src.evaluation.metrics track`.
- Appearance features (step F):
  - `src/reid/gallery.py`: picks the K best clean, non-border, well-spaced observations per track.
  - `src/reid/keypoint_features.py`: normalised 128×256 crop plus mask, then SIFT, ORB, sift_auto or ASIFT capped at `features.max_keypoints`.
  - `src/reid/color_features.py`: gray-world colour balance and upper/lower-body H-S histograms.
- Re-ID (step G):
  - `src/reid/reid_similarity.py`: keypoint score (ratio test, vertical constraint, optional RANSAC), colour score, floor-geometry score, and the weighted combination.
  - `src/reid/reid.py`: describes tracks, builds the score tables `ngoai_hinh` and `ngoai_hinh_san` (always report them separately), and assigns global IDs with Hungarian and `reid.match_threshold`.
  - `tools/run_reid.py` runs oracle mode (ground-truth boxes as tracks) and needs a scene with `cam_a/gt_tracking.csv`, `cam_b/gt_tracking.csv` and `gt_reid.csv`; it exits with a message otherwise. It scores with `src.evaluation.metrics.reid_metrics` and `rank1`. `tools/explain_reid.py` draws the tables and keypoint matches and needs the same ground truth (`--scene` is required).
- Real data (QĐ-31, QĐ-32, `docs/15_du_lieu_that_va_dap_an.md`, `data/README.md`):
  - `tools/tao_canh.py` builds a scene folder by copying videos with their original extension (no re-encoding, no symlinks).
  - `tools/epfl_homography.py` builds `homography.json` from `calibration-4p.txt` (H_AB = G_B⁻¹·G_A; `--calib-size` rescales a calibration made at another image size).

The new end-to-end system (phase 2) wires these modules together. It is documented in `docs/10_tim_nguoi.md`, `docs/11_tracker_moi.md` and `docs/12_he_thong.md`:
- `src/detect/background.py` and `src/detect/detection.py`: step C. `PersonDetector` uses MOG2 by default since QĐ-25, with `mog2_learning_rate` 0.001 since QĐ-28 (0.005 absorbed people standing still within about 8 s; the median background with selective update stays for comparison but gives 14–20% false foreground on EPFL), and is configured by the `detection` section.
- `src/detect/stabilization.py` (QĐ-30): `Stabilizer` aligns every frame to the first one (LK on goodFeaturesToTrack points + `estimateAffinePartial2D` RANSAC) when corners move ≥ `stabilization.min_shift_px`; `pipeline.stable_frames` feeds detection/tracking with stabilized frames plus a `valid_mask`, and every box written out is mapped back with `to_original`, so CSVs, GT, Re-ID crops and debug video stay in original video coordinates. It was needed for the deleted S1 clip (drifted ~40 px). The user's 4p cameras are fixed, so frames pass unchanged; no real drifting video remains, so the drift path is only unit-tested with numbers.
- `src/detect/height_model.py` (QĐ-26): `pipeline.estimate_heights` runs the detector without grouping on the first `detection.height_model.frames` frames. It fits the expected person height `h = a·y_foot + b` with Theil–Sen (median of pairwise slopes; least squares was dragged down by a truncated-people bin at the image edge) through the 90th-percentile height per foot-y bin and writes `height_model.json`. Since QĐ-30 the model is discarded when the median relative residual of its points exceeds `height_model.max_residual` (0.15), e.g. the deleted multi-floor S1 clip (22%). The detector then joins vertically stacked pieces (head and legs of a person in dark clothes) only when the joined box is at most `group_max_height_ratio × h`. Plain "stacked" grouping and tracker-side fragment removal (`fragment_max_vertical_overlap`, off with −1) were tried and hurt the corridor scenes, because a far person sits right above a near one.
- Step D, configured by the `people_tracker` section:
  - `src/tracking/people_tracker.py` holds the `PeopleTracker` state machine: tentative, confirmed, merged, static, lost.
  - `src/tracking/track.py`, `src/tracking/optical_flow.py`, `src/tracking/blob_analysis.py` and `src/tracking/recovery.py` are its parts.
  - The rules found by testing: implausible-size detections that overlap another track don't update position or size; split pieces are joined only when vertically stacked; during a merge only the front track (largest bottom y) keeps LK; merge membership has hysteresis (enter at `merge_coverage` 0.6, stay at `merge_keep_coverage` 0.3); recovery after a merge refuses detections with colour similarity < `recover_min_color_similarity` (QĐ-24); when a height model exists, a detection may start a new track only if `tall_enough` (h ≥ `new_track_min_height_ratio` 0.7 since QĐ-29 × expected height, or touching the image border within `new_track_border_px`), which cut EPFL 4p IDs from about 195 to about 75 per camera (QĐ-27). The switches `appearance_weight`, `appearance_update_rate` and `template_search` exist only for ablation; they are off by default because they made results worse.
- `src/pipeline.py`: `process_camera` runs C → D → E for one camera and writes `detections.csv` and `tracks.csv`, plus `debug.mp4` with `--debug-video`.
- `src/evaluation/system_eval.py`: end-to-end scoring. It maps each track to a GT ID by majority IoU vote, uses the main track per person, and scores Re-ID.
- `src/tracking/tracklet_linking.py`: post-hoc linking of broken tracks within one camera (QĐ-23). It uses the time gap, the predicted position from the end velocity and colour similarity, linked with Hungarian, and is configured by the `tracklet_linking` section. `run.py` writes `tracks_linked.csv` and reports MOT before and after linking.
- `run.py --scene <dir> [--out] [--debug-video]` runs both cameras, tracklet linking, Re-ID and the summary, writing to `outputs/he_thong/<scene>/`.
- `src/evaluation/metrics.py` now scores track states `confirmed`, `merged` and `static` (`SCORED_TRACK_STATES`).

The team writes code with AI help and has to defend every line in front of the teacher. Rules for new code:
- One concept per file, under about 200 lines, with short functions.
- Never put several statements on one line with `;`.
- Docstrings in Vietnamese that explain *why* and give the formula.
- Every threshold lives in config, with a comment giving its meaning and unit. The user tunes parameters themselves using `docs/14_hieu_chinh_tham_so.md` and `tools/hieu_chinh.py`; don't change tuned values without being asked, and keep that doc in sync when adding a knob.
- Keep diagnostics and logging out of the algorithm code.
- A finished module ships with three things: `docs/NN_<module>.md` (problem → algorithm → formula → parameters → failure cases → likely teacher questions), an `--explain` visualisation of each step, and new entries in `docs/hoi_dap_bao_ve.md`. After that, explain the module to the user in chat in plain Vietnamese and ask only the 1–2 core questions they must be able to answer at the defense (for example "what does this step do and why" or "is this ML"). Skip detail questions.

## Commands

Run everything from the repo root inside `.venv` (Python 3.14, OpenCV 5.1). The README asks for Python ≥3.10.

```bash
python -m pytest                       # use `python -m`: plain `pytest` can't import `src`; tests need data/canh (else they skip)
python -m pytest tests/test_that_bam_nguoi.py

python run.py --scene data/canh/epfl_lab4p_c0_c1 [--debug-video] [--max-frames N]   # WHOLE SYSTEM on one scene; writes outputs/he_thong/<scene>/ (summary.json has scores only when gt_tracking.csv exists)
python -m tools.hieu_chinh [--set "section.key=[v1, v2]" ...] [--chi-dap-an]   # one-at-a-time sweep over every scene in data/canh: MOTA/IDF1/Re-ID on scenes with ground truth, ID counts elsewhere; writes outputs/hieu_chinh/<time>/bang_diem.csv
python -m tools.cham_khong_dap_an --run outputs/he_thong/<scene> --scene data/canh/<scene>   # ground-truth-free scorecard (QĐ-34): so_id, phu_nguoi, track_ma, nhay, cao_hop_ly; needs camera.json for calibration
python -m tools.run_reid --scene data/canh/<scene with ground truth> [--method sift|orb|sift_auto|asift]   # oracle Re-ID on ground-truth boxes
python -m tools.tao_canh --cam-a <video> [--cam-b <video>] --out data/canh/<scene>       # build a scene (copies videos, keeps extension)
python -m tools.epfl_homography --calib data/goc/epfl_lab4p/calibration-4p.txt --cam-a 0 --cam-b 1 --scene data/canh/epfl_lab4p_c0_c1 [--top-view X0 Y0 W H] [--explain <png>]
python tools/annotate.py  --video <video> --out <gt.csv>      # interactive OpenCV window; key o cycles occlusion; keeps the native video size unless --size
python tools/verify_gt.py --video <video> --gt <gt.csv> --out outputs/gt_check.mp4
python tools/calibrate_floor.py --cam-a <video_a> --cam-b <video_b> --out <homography.json> [--explain <png>]

# Illustrations (all on real video)
python -m tools.explain_assignment     # outputs/explain/ghep_cap/ (track boxes vs detections of a run.py result, 4p frame 800; run run.py first)
python -m tools.explain_kalman --run outputs/he_thong/epfl_lab4p_c0_c1 --scene data/canh/epfl_lab4p_c0_c1   # outputs/explain/kalman/ (run run.py first)
python -m tools.explain_reid --scene data/canh/<scene with ground truth> --person 1   # outputs/explain/nhan_dang_lai/
python -m tools.explain_height_model --video data/canh/epfl_lab4p_c0_c1/cam_b/video.avi --frame 1800   # outputs/explain/chieu_cao/
python -m tools.explain_stabilization --video <video with a drifting camera>                         # outputs/explain/on_dinh_hinh/

# Old system (kept for comparison); it resizes to scene.resize_* (640×480), so score it against native-size ground truth only with a config whose scene size is 360×288
python -m legacy.run_tracking --video data/canh/epfl_lab4p_c0_c1/cam_a/video.avi --out outputs/tracks.csv [--debug-video outputs/tracking_debug.mp4] [--config <yaml>]
python -m src.evaluation.metrics track outputs/tracks.csv <gt_tracking.csv> [--iou-thr 0.5]
python tools/trace_tracking.py --video <video> --gt <gt.csv> --tracks-out outputs/<run>/tracks.csv --diagnostics-out outputs/<run>/diagnostics.jsonl
```

No linter, formatter or type checker is set up.

A teammate runs the project on native Windows with a venv (no WSL). So: always pass `encoding="utf-8"` to text file I/O, never rely on symlinks in tracked or documented data layouts, and keep `PYTHONUTF8=1` for subprocesses whose output goes to files (`tools/hieu_chinh.py` sets it). The Windows setup steps are in README.md.

## Architecture

This section describes the code as it is now, which is the old design (see **Planned redesign** above). Every entry point runs the same chain:
`utils.iter_video` (resizes each frame once to `scene.resize_width × resize_height`) → `Detector.detect(frame_bgr)` → `Tracker.update(gray, detections, frame_bgr=frame)` → CSV → `src.evaluation.metrics`.

Every box is `(x, y, w, h)` in integer pixels of the **resized** frame. All pixel thresholds in `config.yaml` are measured at that resolution.

### Config
- `load_config` checks that the sections `scene, detector, tracker, events, features, reid, benchmark` exist, and checks the keys inside `scene`. It returns an `AppConfig` in which every section except `scene` is a raw dict. `SceneConfig` drops `ignore_regions`, so the entry points read the YAML a second time and pass `Detector({"detector": cfg.detector, "scene": raw["scene"]})`.
- `metrics` is not a section of `AppConfig`. The `src/evaluation/metrics.default_*` functions read `config.yaml` from the repo root directly and ignore `--config`.
- `Detector` and `Tracker` accept either a Mapping or an object with attributes. Each checks every key itself and has **no defaults**. The old tracker's synthetic tests were deleted (QĐ-31); only `tests/legacy/test_config.py` remains.
- In the OLD code, nothing reads the `events`, `features`, `reid` and `benchmark` sections. The new system reads `features`, `reid`, `detection`, `people_tracker`, `kalman`, `tracklet_linking` and `homography`. `events` is now empty (door counting was dropped) and `benchmark` is still unused.

### Detector (`legacy/detector.py`)
The steps run in this order:
1. Gaussian blur.
2. MOG2 with `detectShadows=True`. Only pixels equal to 255 are kept, which removes shadow pixels (127).
3. The ROI and ignore regions are masked out.
4. Opening, closing and dilation.
5. The ROI and ignore regions are masked out again.
6. Connected components are found and filtered by area and by h/w ratio.

`get_foreground_mask()` returns the most recent processed mask.

### Tracker (`legacy/tracker.py`): one `update()` call, in order
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
`Tracker._diagnostic` sends one JSON event at a time to the logger `legacy.tracker` at INFO level. It does nothing unless INFO is enabled. The events are:
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

`tools/trace_tracking.py` writes these events to JSONL, after a first `run_manifest` record. The manifest holds the git HEAD, a hash of the diff, a copy of the config, and SHA-256 hashes of the source and input files. The test that checked these event fields was synthetic and has been deleted (QĐ-31).

### Tests
The tests build textured frames with NumPy and pass in exact ("oracle") detections. The blob tests replace `tracker._predict_track` so that predictions stay fixed and only the matching logic is tested. `test_tracker_crossing.py` is an end-to-end acceptance test, scored with `tracking_metrics`.

## Data and outputs

`src/evaluation/metrics.read_csv_rows` checks that each CSV has these columns:

| File | Columns |
|---|---|
| detections | `frame,x,y,w,h,area` |
| tracks | `frame,track_id,state,x,y,w,h,n_points` |
| gt_tracking | `frame,gt_id,x,y,w,h` plus, in the new format (QĐ-15), `occluded` (0/1/2) and `keyframe` (1 = hand-labelled) |
| gt_reid | `person,cam_a_id,cam_b_id` |
| global_ids | `cam,track_id,global_id` |
| reid scores | `cam_a_track,cam_b_track,method,score` |

- `detection_pr` checks whether the ground truth has the new columns. If it does, it scores only frames where every box is hand-labelled and none has `occluded == 2` (`scored_frames`). `clear_mot`/IDF1 score only frames present in the ground truth, so sparse EPFL ground truth (every 25 frames) works as is.
- Data layout (QĐ-31, `data/README.md`; `data/` is not on GitHub apart from `data/README.md`):
  - `data/canh/<scene>/cam_a/video.<ext>` (+ optional `cam_b/`, `gt_tracking.csv` per camera, `gt_reid.csv`, `homography.json`, `nguon.txt`). `src.common.utils.camera_video` finds `video.*` with ext mp4/avi/mov/mkv/m4v. `run.py` and `tools/hieu_chinh.py` discover scenes there.
  - Scenes: `epfl_lab4p_c0_c1` (cameras 0 + 1) and `epfl_lab4p_c2_c3` (cameras 2 + 3), copied from the user's avi originals, each with a `homography.json` built from `calibration-4p.txt`. No ground truth.
  - `data/goc/epfl_lab4p/` holds `4p-c0..c3.avi`, `calibration-4p.txt` and `nguon.txt` (research-only licence; cite Fleuret et al., TPAMI 2008).
  - `data/video_moi/` is where the team drops new real videos.
- Current counts (2026-11-23, no ground truth): IDs after linking 42 / 26 (c0+c1) and 56 / 75 (c2+c3) for at most 4 people; about 3.2 tracks per frame; the longest track per camera lasts 1400–1900 frames. At 360×288 SIFT finds only 6–9 good matches for the same person a few frames apart. Colour similarity is 0.82–0.86 for the same person and 0.48–0.64 for different people.
- `outputs/` is git-ignored except for `outputs/README.md`, which describes each folder. Old outputs were cleared on 2026-11-23 (QĐ-31, QĐ-32). Put each new run in its own `outputs/<stage>/<NN_name>/` folder, together with the config it used.
- `.gitignore` covers `.venv/`, Python caches, `outputs/*` and Windows `*:Zone.Identifier` files. `data/` is not ignored, but only `data/README.md` is committed: EPFL video is research-only and videos are large.

## Review findings not yet in the code (as of 2026-11-21; check before relying on them)
These come from the roadmap review (XLA.md, which is not in this repo):
- When LK points are refreshed, the search mask is the track's box minus the neighbours' boxes. It is not intersected with the foreground mask, so corners in the background can drag a box toward zero motion.
- There is no `static` state. When MOG2 absorbs a person who stands still into the background, the person stops producing detections and their track is dropped after `max_misses`, even though LK is still following them.
- Recovery after occlusion relies only on the HSV histogram. It fails when two people wear the same colour. The review suggests adding motion continuity, NCC template matching and a gallery of keypoints.
- For the planned ORB/SIFT work:
  - Pass an eroded foreground mask through the detector's `mask` argument. Don't black out the crop, because that creates fake keypoints along its edge.
  - Lower ORB's `edgeThreshold` and `patchSize` for crops about 128 px wide.
  - Measure memory with RSS or `ru_maxrss`, running each configuration in its own process. Don't use `tracemalloc`.
