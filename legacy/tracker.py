"""Theo dõi nhiều đối tượng bằng Lucas-Kanade và ghép box theo IoU."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from math import hypot
from typing import Any
import json
import logging
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.common.utils import iou

_LOGGER = logging.getLogger(__name__)

@dataclass
class Track:
    """Trạng thái một đối tượng cục bộ và các điểm ảnh bám theo nó."""
    id: int
    state: str
    box: tuple[int, int, int, int]
    points: np.ndarray
    age: int = 1
    hits: int = 1
    misses: int = 0
    gallery: list[Any] = field(default_factory=list)
    consecutive_hits: int = 1
    last_displacement: tuple[float, float] | None = None
    exit_extrapolations: int = 0
    appearance_hist: np.ndarray | None = None
    occluded: bool = False
    blob_grace_age: int = 0

@dataclass(frozen=True)
class _DetectionClassification:
    """Kết quả phân loại hình học của một detection so với track dự đoán."""
    kind: str
    track_indices: frozenset[int]

class Tracker:
    """Duy trì track active và cấp ID tăng đơn điệu, không tái sử dụng."""
    def __init__(self, cfg: Any):
        """Đọc cấu hình tracking và chuẩn bị trạng thái LK."""
        section = cfg["tracker"] if isinstance(cfg, Mapping) else cfg.tracker
        self.win_size = _pair(_value(section, "winSize"), "winSize")
        self.max_level = _positive_or_zero_int(section, "maxLevel")
        criteria = _value(section, "criteria")
        if not isinstance(criteria, (list, tuple)) or len(criteria) != 3:
            raise ValueError("tracker.criteria phải có dạng [type, max_count, epsilon]")
        self.criteria = (int(criteria[0]), int(criteria[1]), float(criteria[2]))
        self.use_fb_check = _value(section, "use_fb_check")
        self.fb_thresh = _positive_float(section, "fb_thresh")
        self.min_points = _positive_int(section, "min_points")
        self.max_corners = _positive_int(section, "max_corners")
        self.quality_level = _positive_float(section, "quality_level")
        self.min_distance = _positive_float(section, "min_distance")
        self.block_size = _positive_int(section, "block_size")
        self.refresh_every = _positive_int(section, "refresh_every")
        self.matcher = _value(section, "matcher")
        if self.matcher not in {"hungarian", "greedy"}:
            raise ValueError("tracker.matcher phải là 'hungarian' hoặc 'greedy'")
        self.iou_gate = _unit_interval(section, "iou_gate")
        self.confirm_hits = _positive_int(section, "confirm_hits")
        self.max_misses = _positive_int(section, "max_misses")
        self.box_update_alpha = _unit_interval(section, "box_update_alpha")
        self.blob_iou = _unit_interval(section, "blob_iou")
        self.blob_track_coverage = _unit_interval(section, "blob_track_coverage")
        self.blob_center_separation = _positive_float(section, "blob_center_separation")
        self.blob_grace_frames = _positive_int(section, "blob_grace_frames")
        self.hist_bins = _pair(_value(section, "hist_bins"), "hist_bins")
        self.hist_corr_min = _unit_interval(section, "hist_corr_min")
        self._tracks: list[Track] = []
        self._next_id = 1
        self._previous_gray: np.ndarray | None = None
        self._frame_index = 0
        self._lost_this_frame: list[Track] = []
        self._ownership_snapshot: tuple[tuple[int, str, tuple[int, int, int, int]], ...] = ()
        self._predicted_snapshot: tuple[tuple[int, str, tuple[int, int, int, int]], ...] = ()

    def _diagnostic(self, event: str, **values: Any) -> None:
        """Ghi sự kiện JSON dạng text khi logger diagnostics được bật."""
        if not _LOGGER.isEnabledFor(logging.INFO):
            return
        payload = {"event": event, "frame_index": self._frame_index, **values}
        _LOGGER.info(json.dumps(payload, sort_keys=True, allow_nan=False))

    def update(self, frame_gray: np.ndarray, detections: list[tuple[int, int, int, int]], frame_bgr: np.ndarray | None = None) -> list[Track]:
        """Cập nhật track; BGR thật tùy chọn dùng cho histogram và phục hồi sau blob."""
        if frame_gray.ndim != 2 or frame_gray.dtype != np.uint8:
            raise ValueError("frame_gray phải là ảnh grayscale uint8")
        if frame_bgr is not None and (frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8 or frame_bgr.shape[:2] != frame_gray.shape):
            raise ValueError("frame_bgr phải là BGR uint8 cùng kích thước với frame_gray")
        height, width = frame_gray.shape
        valid_detections = [_clip_box(_validate_box(box), width, height) for box in detections]
        valid_detections = [box for box in valid_detections if box is not None]
        self._lost_this_frame = []
        lifecycle_before = {
            track.id: {
                "state": track.state, "misses": track.misses,
                "occluded": track.occluded,
                "grace_age": track.blob_grace_age,
            }
            for track in self._tracks
        }
        self._ownership_snapshot = tuple(
            (track.id, track.state, track.box) for track in self._tracks
        )
        predictions: list[tuple[int,int,int,int]] = []
        flow_points: list[np.ndarray] = []
        if self._previous_gray is None:
            predictions = [track.box for track in self._tracks]
            flow_points = [track.points for track in self._tracks]
            for track in self._tracks:
                self._diagnostic(
                    "prediction", track_id=track.id, points_before_lk=0,
                    points_after_status_filter=0, points_after_fb_check=0,
                    points_rejected_own_box=0, points_rejected_neighbor=0,
                    points_after_ownership=len(track.points), points_used_for_median=0,
                    median_dx=None, median_dy=None, prediction_applied=False,
                    prediction_fallback_reason="first_frame",
                    exit_extrapolation_applied=False, predicted_box=list(track.box),
                )
        else:
            for track in self._tracks:
                predicted, points, _ = self._predict_track(track, frame_gray, width, height)
                predictions.append(predicted); flow_points.append(points)
        self._predicted_snapshot = tuple(
            (track.id, self._ownership_snapshot[index][1], predictions[index])
            for index, track in enumerate(self._tracks)
        )

        self._diagnostic(
            "frame_start", detection_count=len(valid_detections),
            active_track_count=len(self._tracks), frame_width=width, frame_height=height,
        )

        classifications = self._classify_detections(predictions, valid_detections)
        merged = {j: set(value.track_indices) for j, value in enumerate(classifications) if value.kind == "merged"}
        merged_related = set().union(*merged.values()) if merged else set()
        confirmed_at_start = {
            index for index, snapshot in enumerate(self._ownership_snapshot)
            if snapshot[1] == "confirmed"
        }
        occluded_now = merged_related & confirmed_at_start
        previous_occluded_all = {i for i, track in enumerate(self._tracks) if track.occluded}
        previous_occluded = previous_occluded_all & confirmed_at_start
        ambiguous = {j: set(value.track_indices) for j, value in enumerate(classifications) if value.kind == "ambiguous"}
        ambiguous_related = set().union(*ambiguous.values()) if ambiguous else set()
        ordinary_det_indices = [j for j, value in enumerate(classifications) if value.kind == "clean"]
        clean_det_indices = [j for j, value in enumerate(classifications) if value.kind == "clean"]

        recovery_pairs = self._recover_assignments(
            [i for i in previous_occluded if i not in occluded_now],
            clean_det_indices, predictions, valid_detections, frame_bgr,
        )
        matched = dict(recovery_pairs)
        used_detections = {j for _, j in recovery_pairs}
        unresolved_detections = self._unresolved_recovery_detections(
            [i for i in previous_occluded if i not in occluded_now],
            [j for j in clean_det_indices if j not in used_detections],
            predictions, valid_detections, frame_bgr,
        )
        ordinary_tracks = [i for i in range(len(self._tracks)) if i not in occluded_now and i not in previous_occluded]
        remaining_detections = [j for j in ordinary_det_indices if j not in used_detections]
        normal_pairs = self._assign([predictions[i] for i in ordinary_tracks], [valid_detections[j] for j in remaining_detections])
        self._diagnostic(
            "normal_assignment_result",
            candidate_track_ids=[self._tracks[i].id for i in ordinary_tracks],
            candidate_detection_indices=list(remaining_detections),
            assignments=[
                {"track_id": self._tracks[ordinary_tracks[i]].id,
                 "detection_index": remaining_detections[j]}
                for i, j in normal_pairs
            ],
        )
        for local_i, local_j in normal_pairs:
            matched[ordinary_tracks[local_i]] = remaining_detections[local_j]
        normal_matched = {ordinary_tracks[local_i] for local_i, _ in normal_pairs}
        used_detections.update(remaining_detections[j] for _, j in normal_pairs)
        updated: list[Track] = []
        refresh_candidate_counts: dict[int, int] = {}
        for index, track in enumerate(self._tracks):
            track.age += 1
            previous_box = track.box
            if index in occluded_now:
                track.box = predictions[index]
                track.points = _points_in_box(flow_points[index], track.box)
                track.occluded = True
                track.blob_grace_age = 0
                if _outside(track.box, width, height):
                    track.box = _clip_box(track.box, width, height) or previous_box
                    track.state = "lost"; self._lost_this_frame.append(track)
                else:
                    updated.append(track)
                continue
            if index in matched:
                det_index = matched[index]
                track.box = _blend_box(predictions[index], valid_detections[det_index], self.box_update_alpha, width, height)
                track.hits += 1; track.misses = 0; track.consecutive_hits += 1; track.exit_extrapolations = 0
                if track.consecutive_hits >= self.confirm_hits: track.state = "confirmed"
                track.occluded = False
                track.blob_grace_age = 0
                track.points = _points_in_box(flow_points[index], track.box)
                if (
                    index in normal_matched
                    and index not in previous_occluded_all
                    and index not in merged_related
                    and index not in ambiguous_related
                    and self._should_refresh(track)
                ):
                    refresh_candidate_counts[track.id] = self._refresh_track_points(frame_gray, track, "normal_matched")
                if frame_bgr is not None and classifications[det_index].kind == "clean":
                    appearance = self._appearance_histogram(frame_bgr, valid_detections[det_index])
                    if appearance is not None: track.appearance_hist = appearance
                updated.append(track)
                continue
            if index in previous_occluded:
                track.box = predictions[index]
                track.points = _points_in_box(flow_points[index], track.box)
                track.misses += 1
                track.blob_grace_age += 1
                if track.state == "tentative": track.consecutive_hits = 0
                if _outside(track.box, width, height) or track.blob_grace_age > self.blob_grace_frames:
                    track.box = _clip_box(track.box, width, height) or previous_box
                    track.state = "lost"; self._lost_this_frame.append(track)
                else:
                    updated.append(track)
                continue
            else:
                track.box = predictions[index]
                if self._ownership_snapshot[index][1] == "tentative":
                    track.occluded = False
                    track.blob_grace_age = 0
                track.misses += 1
                if track.state == "tentative": track.consecutive_hits = 0
                if track.misses > self.max_misses or _outside(track.box, width, height):
                    track.box = _clip_box(track.box, width, height) or previous_box
                    track.state = "lost"; self._lost_this_frame.append(track)
                else:
                    track.points = _points_in_box(flow_points[index], track.box)
                    if (
                        index not in merged_related
                        and index not in ambiguous_related
                        and index not in previous_occluded_all
                        and self._should_refresh(track)
                    ):
                        refresh_candidate_counts[track.id] = self._refresh_track_points(frame_gray, track, "normal_unmatched")
                    updated.append(track)
        self._tracks = updated
        for index, box in enumerate(valid_detections):
            if index not in used_detections and index not in unresolved_detections and classifications[index].kind == "clean":
                track = self._new_track(frame_gray, box)
                refresh_candidate_counts[track.id] = len(track.points) if track.points.size else 0
                if frame_bgr is not None:
                    track.appearance_hist = self._appearance_histogram(frame_bgr, box)
                self._tracks.append(track)
                self._diagnostic(
                    "track_created", track_id=track.id, state_after=track.state,
                    box=list(track.box), points_after_refresh=len(track.points),
                )
        current_by_id = {track.id: track for track in self._tracks}
        current_by_id.update({track.id: track for track in self._lost_this_frame})
        for track_id, before in lifecycle_before.items():
            after = current_by_id.get(track_id)
            self._diagnostic(
                "track_lifecycle", track_id=track_id, state_before=before["state"],
                state_after=after.state if after is not None else "removed",
                misses_before=before["misses"], misses_after=after.misses if after is not None else None,
                occluded_previous=before["occluded"], occluded_current=after.occluded if after is not None else None,
                grace_age_before=before["grace_age"], grace_age_after=after.blob_grace_age if after is not None else None,
                grace_remaining=(max(0, self.blob_grace_frames-after.blob_grace_age) if after is not None else 0),
            )
            index = next((i for i, active in enumerate(self._tracks) if active.id == track_id), None)
            if track_id in refresh_candidate_counts:
                attempted=True; safe_count=refresh_candidate_counts[track_id]; suppressed=None
            else:
                attempted=False; safe_count=None
                if index is not None and index in merged_related: suppressed="merged"
                elif index is not None and index in ambiguous_related: suppressed="ambiguous"
                elif index is not None and index in previous_occluded_all: suppressed="prior_occluded"
                else: suppressed="not_due"
            self._diagnostic(
                "ownership_replenishment", track_id=track_id,
                refresh_attempted=attempted, safe_candidate_points=safe_count,
                suppressed_reason=suppressed,
            )
        self._previous_gray = frame_gray.copy()
        self._frame_index += 1
        return list(self._tracks) + list(self._lost_this_frame)

    def _unresolved_recovery_detections(self, track_indices: list[int], det_indices: list[int], predictions: list[tuple[int, int, int, int]], detections: list[tuple[int, int, int, int]], frame_bgr: np.ndarray | None) -> set[int]:
        """Giữ detection sạch đang giống appearance track occluded nhưng thiếu IoU.

        Đây không phải ghép danh tính: box/histogram không được cập nhật và
        detection không tạo ID mới cho đến khi track cũ phục hồi hoặc hết grace.
        """
        if frame_bgr is None or not track_indices or not det_indices: return set()
        unresolved: set[int] = set()
        for track_index in track_indices:
            reference=self._tracks[track_index].appearance_hist
            if reference is None: continue
            for det_index in det_indices:
                candidate=self._appearance_histogram(frame_bgr,detections[det_index])
                if candidate is None: continue
                correlation=float(cv2.compareHist(reference,candidate,cv2.HISTCMP_CORREL))
                overlap=iou(predictions[track_index],detections[det_index])
                if np.isfinite(correlation) and correlation>=self.hist_corr_min and overlap<self.iou_gate:
                    unresolved.add(det_index)
        return unresolved

    def _classify_detections(self, predictions: list[tuple[int, int, int, int]], detections: list[tuple[int, int, int, int]]) -> list[_DetectionClassification]:
        """Phân loại detection trước association bằng IoU, coverage và tâm box.

        Merged cần hai track tách biệt cùng đạt coverage và tâm nằm trong
        detection, hoặc cùng đạt ngưỡng IoU legacy. Nhiều track cùng bị nghi
        liên quan nhưng chưa đủ điều kiện merged được đánh dấu ambiguous.
        """
        results: list[_DetectionClassification] = []
        for detection in detections:
            geometry = [_box_overlap_metrics(detection, prediction) for prediction in predictions]
            centers_inside = {
                index for index, prediction in enumerate(predictions)
                if _center_inside(prediction, detection)
            }
            coverage_candidates = {
                index for index, values in enumerate(geometry)
                if values[1] >= self.blob_track_coverage and index in centers_inside
            }
            iou_candidates = {
                index for index, values in enumerate(geometry)
                if values[0] > self.blob_iou and index in centers_inside
            }
            coverage_distinct = self._distinct_tracks(coverage_candidates, predictions)
            iou_distinct = self._distinct_tracks(iou_candidates, predictions)
            if len(coverage_distinct) >= 2 or len(iou_distinct) >= 2:
                results.append(_DetectionClassification("merged", frozenset(coverage_distinct | iou_distinct)))
            elif len(centers_inside) >= 2 or len(coverage_candidates) >= 2 or len(iou_candidates) >= 2:
                suspect_tracks = centers_inside | coverage_candidates | iou_candidates
                results.append(_DetectionClassification("ambiguous", frozenset(suspect_tracks)))
            else:
                results.append(_DetectionClassification("clean", frozenset(centers_inside)))
        for detection_index, result in enumerate(results):
            overlap_metrics = []
            for track_index, prediction in enumerate(predictions):
                overlap, track_coverage, detection_coverage = _box_overlap_metrics(
                    detections[detection_index], prediction,
                )
                overlap_metrics.append({
                    "track_id": self._tracks[track_index].id if track_index < len(self._tracks) else track_index,
                    "iou": overlap, "track_coverage": track_coverage,
                    "detection_coverage": detection_coverage,
                    "center_inside": _center_inside(prediction, detections[detection_index]),
                })
            center_separations = []
            for left in range(len(predictions)):
                for right in range(left + 1, len(predictions)):
                    center_separations.append({
                        "track_ids": [
                            self._tracks[left].id if left < len(self._tracks) else left,
                            self._tracks[right].id if right < len(self._tracks) else right,
                        ],
                        "separated": _centers_separated(
                            predictions[left], predictions[right], self.blob_center_separation,
                        ),
                    })
            self._diagnostic(
                "detection_classification", detection_index=detection_index,
                detection_box=list(detections[detection_index]), blob_class=result.kind,
                overlap_track_ids=[
                    self._tracks[index].id for index in sorted(result.track_indices)
                    if index < len(self._tracks)
                ],
                overlap_metrics=overlap_metrics, center_separations=center_separations,
            )
        return results

    def _distinct_tracks(self, indices: set[int], predictions: list[tuple[int, int, int, int]]) -> set[int]:
        """Loại các box dự đoán gần trùng tâm khỏi số lượng track độc lập."""
        selected: list[int] = []
        ordered = sorted(indices, key=lambda i: predictions[i][2] * predictions[i][3], reverse=True)
        for index in ordered:
            if all(_centers_separated(predictions[index], predictions[other], self.blob_center_separation) for other in selected):
                selected.append(index)
        return set(selected)

    def _recover_assignments(self, track_indices: list[int], det_indices: list[int], predictions: list[tuple[int,int,int,int]], detections: list[tuple[int,int,int,int]], frame_bgr: np.ndarray | None) -> list[tuple[int,int]]:
        """Ghép track bị che với detection mới chỉ khi cả HSV và IoU đều đạt ngưỡng.

        Chi phí là 1 trừ correlation nhân IoU. Gating độc lập yêu cầu
        correlation >= hist_corr_min và IoU >= iou_gate. Hungarian đảm bảo
        ghép một-một; thiếu BGR hoặc histogram thì không suy đoán danh tính.
        """
        if frame_bgr is None or not track_indices or not det_indices:
            self._diagnostic(
                "recovery_matrix", track_ids=[self._tracks[i].id for i in track_indices],
                detection_indices=list(det_indices), candidates=[],
                hungarian_assignments=[], grace_remaining={
                    str(self._tracks[i].id): max(0, self.blob_grace_frames - self._tracks[i].blob_grace_age)
                    for i in track_indices
                },
            )
            return []
        cost=np.full((len(track_indices),len(det_indices)),1e6,dtype=float)
        candidate_records: list[dict[str, Any]] = []
        trace_enabled = _LOGGER.isEnabledFor(logging.INFO)
        for row,track_index in enumerate(track_indices):
            reference=self._tracks[track_index].appearance_hist
            if reference is None: continue
            for col,det_index in enumerate(det_indices):
                overlap=iou(predictions[track_index],detections[det_index])
                geometry_pass = overlap >= self.iou_gate
                candidate = None
                correlation = None
                if geometry_pass or trace_enabled:
                    candidate=self._appearance_histogram(frame_bgr,detections[det_index])
                    if candidate is not None:
                        measured=float(cv2.compareHist(reference,candidate,cv2.HISTCMP_CORREL))
                        correlation=measured if np.isfinite(measured) else None
                appearance_pass = correlation is not None and correlation >= self.hist_corr_min
                eligible = geometry_pass and appearance_pass
                reason = None
                if not geometry_pass: reason = "below_iou_gate"
                elif candidate is None: reason = "invalid_histogram"
                elif not appearance_pass: reason = "below_hist_corr_min"
                record = {
                    "track_id": self._tracks[track_index].id,
                    "detection_index": det_index, "iou": overlap,
                    "hist_corr": correlation, "geometry_pass": geometry_pass,
                    "appearance_pass": appearance_pass, "candidate_eligible": eligible,
                    "assigned": False, "rejection_reason": reason,
                }
                candidate_records.append(record)
                if eligible:
                    cost[row,col]=1.0-correlation*overlap
        rows,cols=linear_sum_assignment(cost)
        pairs = [(track_indices[int(i)],det_indices[int(j)]) for i,j in zip(rows,cols) if cost[i,j] < 1e6]
        assigned_ids = {(self._tracks[i].id, j) for i, j in pairs}
        for record in candidate_records:
            if (record["track_id"], record["detection_index"]) in assigned_ids:
                record["assigned"] = True
                record["rejection_reason"] = None
            elif record["candidate_eligible"]:
                record["rejection_reason"] = "not_selected_by_hungarian"
        self._diagnostic(
            "recovery_matrix", track_ids=[self._tracks[i].id for i in track_indices],
            detection_indices=list(det_indices), candidates=candidate_records,
            hungarian_assignments=[
                {"track_id": self._tracks[i].id, "detection_index": j}
                for i, j in pairs
            ], grace_remaining={
                str(self._tracks[i].id): max(0, self.blob_grace_frames - self._tracks[i].blob_grace_age)
                for i in track_indices
            },
        )
        return pairs

    def _appearance_histogram(self, frame_bgr: np.ndarray, box: tuple[int,int,int,int]) -> np.ndarray | None:
        """Tính histogram HSV H-S chuẩn hóa từ crop BGR thực tế."""
        x,y,w,h=box; height,width=frame_bgr.shape[:2]
        left=max(0,x); top=max(0,y); right=min(width,x+w); bottom=min(height,y+h)
        if right<=left or bottom<=top: return None
        hsv=cv2.cvtColor(frame_bgr[top:bottom,left:right],cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1],None,list(self.hist_bins),[0,180,0,256])
        if hist is None or not np.isfinite(hist).all() or float(hist.sum())<=0: return None
        return cv2.normalize(hist,hist,alpha=1.0,norm_type=cv2.NORM_L1)

    def _predict_track(self, track: Track, current_gray: np.ndarray, width: int, height: int) -> tuple[tuple[int,int,int,int],np.ndarray,bool]:
        points = track.points
        empty = np.empty((0,1,2), dtype=np.float32)
        points_before = len(points)
        own_box = next((box for track_id, _, box in self._ownership_snapshot if track_id == track.id), track.box)
        neighbor_boxes = [
            box for track_id, state, box in self._ownership_snapshot
            if track_id != track.id and state in {"tentative", "confirmed"}
        ]
        if len(points) == 0:
            predicted, kept, sufficient = self._extrapolate_exit(track, width, height, empty)
            self._diagnostic(
                "prediction", track_id=track.id, points_before_lk=0,
                points_after_status_filter=0, points_after_fb_check=0,
                points_rejected_own_box=0, points_rejected_neighbor=0,
                points_after_ownership=0,
                points_used_for_median=0, median_dx=None, median_dy=None,
                prediction_applied=False,
                prediction_fallback_reason="no_points",
                exit_extrapolation_applied=predicted != track.box,
                predicted_box=list(predicted),
            )
            return predicted, kept, sufficient
        forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(self._previous_gray, current_gray, points, None, winSize=self.win_size, maxLevel=self.max_level, criteria=self.criteria)
        if forward is None or status_forward is None:
            predicted, kept, sufficient = self._extrapolate_exit(track, width, height, empty)
            self._diagnostic(
                "prediction", track_id=track.id, points_before_lk=points_before,
                points_after_status_filter=0, points_after_fb_check=0,
                points_rejected_own_box=0, points_rejected_neighbor=0,
                points_after_ownership=0,
                points_used_for_median=0, median_dx=None, median_dy=None,
                prediction_applied=False,
                prediction_fallback_reason="forward_lk_failed",
                exit_extrapolation_applied=predicted != track.box,
                predicted_box=list(predicted),
            )
            return predicted, kept, sufficient
        old = points.reshape(-1,2); new = forward.reshape(-1,2); valid = status_forward.reshape(-1).astype(bool)
        valid &= np.isfinite(old).all(axis=1) & np.isfinite(new).all(axis=1)
        valid &= (new[:,0] >= 0) & (new[:,0] < width) & (new[:,1] >= 0) & (new[:,1] < height)
        points_after_status = int(valid.sum())
        if self.use_fb_check and valid.any():
            candidate = new[valid].reshape(-1,1,2).astype(np.float32)
            backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(current_gray, self._previous_gray, candidate, None, winSize=self.win_size, maxLevel=self.max_level, criteria=self.criteria)
            if backward is None or status_backward is None:
                valid[:] = False
            else:
                valid_indices = np.flatnonzero(valid); returned = backward.reshape(-1,2); back_status = status_backward.reshape(-1).astype(bool)
                errors = np.linalg.norm(old[valid_indices] - returned, axis=1)
                valid[valid_indices] &= back_status & np.isfinite(returned).all(axis=1) & np.isfinite(errors) & (errors <= self.fb_thresh)
        fb_old, fb_new = old[valid], new[valid]
        good_old, good_new, rejected_own, rejected_neighbor = _filter_owned_correspondences(
            fb_old, fb_new, own_box, neighbor_boxes,
        )
        good_points = good_new.reshape(-1,1,2).astype(np.float32)
        if len(good_new) < self.min_points:
            predicted, kept, sufficient = self._extrapolate_exit(track, width, height, good_points)
            self._diagnostic(
                "prediction", track_id=track.id, points_before_lk=points_before,
                points_after_status_filter=points_after_status,
                points_after_fb_check=len(fb_new),
                points_rejected_own_box=rejected_own,
                points_rejected_neighbor=rejected_neighbor,
                points_after_ownership=len(good_new), points_used_for_median=0,
                median_dx=None, median_dy=None, prediction_applied=False,
                prediction_fallback_reason="insufficient_points",
                exit_extrapolation_applied=predicted != track.box,
                predicted_box=list(predicted),
            )
            return predicted, kept, sufficient
        displacement = np.median(good_new - good_old, axis=0)
        track.last_displacement = (float(displacement[0]), float(displacement[1]))
        track.exit_extrapolations = 0
        x,y,w,h=track.box
        moved=(int(round(x+float(displacement[0]))),int(round(y+float(displacement[1]))),w,h)
        clipped=_clip_box(moved,width,height)
        predicted = moved if clipped is None else clipped
        self._diagnostic(
            "prediction", track_id=track.id, points_before_lk=points_before,
            points_after_status_filter=points_after_status,
            points_after_fb_check=len(fb_new),
            points_rejected_own_box=rejected_own,
            points_rejected_neighbor=rejected_neighbor,
            points_after_ownership=len(good_new), points_used_for_median=len(good_new),
            median_dx=float(displacement[0]), median_dy=float(displacement[1]),
            prediction_applied=True, prediction_fallback_reason=None,
            exit_extrapolation_applied=False, predicted_box=list(predicted),
        )
        return predicted, good_points, True


    def _extrapolate_exit(self, track: Track, width: int, height: int, points: np.ndarray) -> tuple[tuple[int,int,int,int],np.ndarray,bool]:
        """Tiếp tục tối đa max_misses bước theo LK tin cậy khi tiến sát mép ảnh."""
        displacement = track.last_displacement
        if displacement is None or track.exit_extrapolations >= self.max_misses:
            return track.box, points, False
        dx, dy = displacement
        x, y, box_width, box_height = track.box
        candidate = (int(round(x + dx)), int(round(y + dy)), box_width, box_height)
        approaching_edge = (
            (dx > 0 and x + box_width + dx >= width)
            or (dx < 0 and x + dx <= 0)
            or (dy > 0 and y + box_height + dy >= height)
            or (dy < 0 and y + dy <= 0)
        )
        if not approaching_edge:
            return track.box, points, False
        track.exit_extrapolations += 1
        clipped = _clip_box(candidate, width, height)
        return (candidate if clipped is None else clipped), points, False

    def _assign(self, predictions: list[tuple[int,int,int,int]], detections: list[tuple[int,int,int,int]]) -> list[tuple[int,int]]:
        """Ghép track và detection, rồi áp dụng IoU gate cho kết quả."""
        if not predictions or not detections:
            self._diagnostic(
                "association_matrix", branch=self.matcher,
                track_boxes=[list(box) for box in predictions],
                detection_boxes=[list(box) for box in detections],
                iou_matrix=[], cost_matrix=[], candidate_eligible=[], assignments=[],
                rejected_pairs=[],
            )
            return []
        overlaps=np.array([[iou(a,b) for b in detections] for a in predictions],dtype=float)
        eligible = overlaps >= self.iou_gate
        if self.matcher == "greedy":
            candidates=sorted(((1.0-overlaps[i,j],i,j) for i in range(len(predictions)) for j in range(len(detections)) if overlaps[i,j] >= self.iou_gate))
            pairs=[]; used_i=set(); used_j=set()
            for _,i,j in candidates:
                if i not in used_i and j not in used_j: pairs.append((i,j)); used_i.add(i); used_j.add(j)
            costs = 1.0 - overlaps
            self._diagnostic(
                "association_matrix", branch=self.matcher,
                track_boxes=[list(box) for box in predictions],
                detection_boxes=[list(box) for box in detections],
                iou_matrix=overlaps.tolist(), cost_matrix=costs.tolist(),
                candidate_eligible=eligible.tolist(),
                assignments=[[int(i), int(j)] for i, j in pairs],
                rejected_pairs=[
                    {"track_index": i, "detection_index": j,
                     "reason": "below_iou_gate" if not eligible[i, j] else "greedy_one_to_one"}
                    for i in range(len(predictions)) for j in range(len(detections))
                    if (i, j) not in pairs
                ],
            )
            return pairs
        cost=1.0-overlaps; cost[overlaps < self.iou_gate] = 1e6
        rows,cols=linear_sum_assignment(cost)
        pairs = [(int(i),int(j)) for i,j in zip(rows,cols) if overlaps[i,j] >= self.iou_gate]
        self._diagnostic(
            "association_matrix", branch="hungarian",
            track_boxes=[list(box) for box in predictions],
            detection_boxes=[list(box) for box in detections],
            iou_matrix=overlaps.tolist(), cost_matrix=cost.tolist(),
            candidate_eligible=eligible.tolist(),
            assignments=[[int(i), int(j)] for i, j in pairs],
            rejected_pairs=[
                {"track_index": i, "detection_index": j,
                 "reason": "below_iou_gate" if not eligible[i, j] else "not_selected_by_assignment"}
                for i in range(len(predictions)) for j in range(len(detections))
                if (i, j) not in pairs
            ],
        )
        return pairs

    def _new_track(self, frame_gray: np.ndarray, box: tuple[int,int,int,int]) -> Track:
        track=Track(self._next_id,"tentative",box,np.empty((0,1,2),dtype=np.float32))
        self._next_id+=1; self._refresh_track_points(frame_gray,track,"new_track")
        if self.confirm_hits <= 1: track.state="confirmed"
        return track

    def _refresh_track_points(self, frame_gray: np.ndarray, track: Track, branch: str) -> int:
        """Refresh khi đủ điểm an toàn; thiếu thì giữ LK points cũ còn hợp lệ."""
        before = len(track.points)
        neighbor_boxes = [
            box for track_id, state, box in self._predicted_snapshot
            if track_id != track.id and state in {"tentative", "confirmed"}
        ]
        refreshed = self._find_points(frame_gray, track.box, neighbor_boxes)
        candidate_count = len(refreshed)
        skipped = candidate_count < self.min_points
        if not skipped:
            track.points = refreshed
        self._diagnostic(
            "point_refresh", track_id=track.id, refresh_branch=branch,
            points_before_refresh=before, points_after_refresh=len(track.points),
            points_added=(len(track.points)-before) if not skipped else 0,
            safe_candidate_points=candidate_count, refresh_skipped=skipped,
        )
        return candidate_count

    def _find_points(self, frame_gray: np.ndarray, box: tuple[int,int,int,int], neighbor_boxes: list[tuple[int,int,int,int]] | None = None) -> np.ndarray:
        """Tìm góc trong box track trừ union predicted boxes của neighbors."""
        height,width=frame_gray.shape
        clipped=_clip_box(box,width,height)
        if clipped is None: return np.empty((0,1,2),dtype=np.float32)
        x,y,w,h=clipped
        mask=np.zeros(frame_gray.shape,dtype=np.uint8)
        mask[y:y+h,x:x+w]=255
        for neighbor in neighbor_boxes or []:
            clipped_neighbor=_clip_box(neighbor,width,height)
            if clipped_neighbor is None: continue
            nx,ny,nw,nh=clipped_neighbor
            mask[ny:ny+nh,nx:nx+nw]=0
        if not np.any(mask): return np.empty((0,1,2),dtype=np.float32)
        points=cv2.goodFeaturesToTrack(frame_gray,maxCorners=self.max_corners,qualityLevel=self.quality_level,minDistance=self.min_distance,mask=mask,blockSize=self.block_size)
        if points is None: return np.empty((0,1,2),dtype=np.float32)
        points=points.astype(np.float32,copy=False).reshape(-1,1,2)
        finite=np.isfinite(points.reshape(-1,2)).all(axis=1)
        return points[finite]

    def _should_refresh(self, track: Track) -> bool:
        return self._frame_index % self.refresh_every == 0 or len(track.points) < self.min_points

def _value(section: Any,key: str) -> Any:
    try: return section[key] if isinstance(section,Mapping) else getattr(section,key)
    except (KeyError,AttributeError) as exc: raise ValueError(f"Thiếu tracker.{key} trong config") from exc

def _positive_int(section: Any,key: str) -> int:
    value=_value(section,key)
    if type(value) is not int or value<=0: raise ValueError(f"tracker.{key} phải là số nguyên dương")
    return value
def _positive_or_zero_int(section: Any,key: str) -> int:
    value=_value(section,key)
    if type(value) is not int or value<0: raise ValueError(f"tracker.{key} phải là số nguyên không âm")
    return value
def _positive_float(section: Any,key: str) -> float:
    value=_value(section,key)
    if not isinstance(value,(int,float)) or isinstance(value,bool) or value<=0: raise ValueError(f"tracker.{key} phải là số dương")
    return float(value)
def _unit_interval(section: Any,key: str) -> float:
    value=_value(section,key)
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not 0<=value<=1: raise ValueError(f"tracker.{key} phải nằm trong [0, 1]")
    return float(value)
def _pair(value: Any,key: str) -> tuple[int,int]:
    if not isinstance(value,(list,tuple)) or len(value)!=2 or any(type(v) is not int or v<=0 for v in value): raise ValueError(f"tracker.{key} phải gồm hai số nguyên dương")
    return int(value[0]),int(value[1])
def _validate_box(box: Any) -> tuple[int,int,int,int]:
    if len(box)!=4 or not np.isfinite(np.asarray(box,dtype=float)).all(): raise ValueError("Detection box phải có bốn tọa độ hữu hạn")
    x,y,w,h=(int(round(float(v))) for v in box)
    if w<=0 or h<=0: raise ValueError("Detection box cần w,h dương")
    return x,y,w,h
def _clip_box(box: tuple[int,int,int,int],width: int,height: int) -> tuple[int,int,int,int] | None:
    x,y,w,h=box; left=max(0,x); top=max(0,y); right=min(width,x+w); bottom=min(height,y+h)
    if right<=left or bottom<=top: return None
    return left,top,right-left,bottom-top
def _outside(box: tuple[int,int,int,int],width: int,height: int) -> bool:
    x,y,w,h=box
    return x+w<=0 or y+h<=0 or x>=width or y>=height

def _box_overlap_metrics(detection: tuple[int,int,int,int], track_box: tuple[int,int,int,int]) -> tuple[float,float,float]:
    """Trả IoU, coverage theo diện tích track và coverage theo detection."""
    dx,dy,dw,dh=detection; tx,ty,tw,th=track_box
    detection_area=max(0,dw)*max(0,dh); track_area=max(0,tw)*max(0,th)
    if detection_area<=0 or track_area<=0: return 0.0,0.0,0.0
    intersection=max(0,min(dx+dw,tx+tw)-max(dx,tx))*max(0,min(dy+dh,ty+th)-max(dy,ty))
    union=detection_area+track_area-intersection
    return (intersection/union if union>0 else 0.0, intersection/track_area, intersection/detection_area)

def _center_inside(box: tuple[int,int,int,int], container: tuple[int,int,int,int]) -> bool:
    """Kiểm tra tâm box nằm trong hình chữ nhật nửa mở khác."""
    x,y,w,h=box; cx=x+w/2; cy=y+h/2
    left,top,width,height=container
    return left<=cx<left+width and top<=cy<top+height

def _centers_separated(box_a: tuple[int,int,int,int], box_b: tuple[int,int,int,int], ratio: float) -> bool:
    """So sánh khoảng cách tâm với đường chéo nhỏ hơn của hai box."""
    ax,ay,aw,ah=box_a; bx,by,bw,bh=box_b
    diagonal_a=hypot(aw,ah); diagonal_b=hypot(bw,bh)
    scale=min(diagonal_a,diagonal_b)
    if scale<=0: return False
    distance=hypot((ax+aw/2)-(bx+bw/2),(ay+ah/2)-(by+bh/2))
    return distance>=ratio*scale

def _blend_box(prediction: tuple[int,int,int,int],detection: tuple[int,int,int,int],alpha: float,width: int,height: int) -> tuple[int,int,int,int]:
    values=tuple(int(round(alpha*p+(1-alpha)*d)) for p,d in zip(prediction,detection))
    return _clip_box(values,width,height) or detection


def _points_in_box(points: np.ndarray, box: tuple[int,int,int,int]) -> np.ndarray:
    """Giữ feature points nằm trong box cập nhật của Track."""
    if len(points)==0: return np.empty((0,1,2),dtype=np.float32)
    x,y,w,h=box; flat=points.reshape(-1,2)
    keep=(flat[:,0]>=x)&(flat[:,0]<x+w)&(flat[:,1]>=y)&(flat[:,1]<y+h)&np.isfinite(flat).all(axis=1)
    return flat[keep].reshape(-1,1,2).astype(np.float32,copy=False)

def _filter_owned_correspondences(
    points_previous: np.ndarray,
    points_current: np.ndarray,
    own_box: tuple[int,int,int,int],
    neighbor_boxes: list[tuple[int,int,int,int]],
) -> tuple[np.ndarray,np.ndarray,int,int]:
    """Lọc LK correspondences bằng p_prev trong snapshot box đóng frame trước.

    Own box và neighbor box dùng biên inclusive: x0 <= px <= x0+w,
    y0 <= py <= y0+h. Neighbor ownership được ưu tiên khi hai box chồng lấn.
    """
    previous=points_previous.reshape(-1,2)
    current=points_current.reshape(-1,2)
    x,y,w,h=own_box
    in_own=(previous[:,0]>=x)&(previous[:,0]<=x+w)&(previous[:,1]>=y)&(previous[:,1]<=y+h)
    in_neighbor=np.zeros(len(previous),dtype=bool)
    for nx,ny,nw,nh in neighbor_boxes:
        in_neighbor |= (
            (previous[:,0]>=nx)&(previous[:,0]<=nx+nw)
            &(previous[:,1]>=ny)&(previous[:,1]<=ny+nh)
        )
    rejected_neighbor=int(in_neighbor.sum())
    rejected_own=int((~in_own & ~in_neighbor).sum())
    keep=in_own & ~in_neighbor
    return previous[keep],current[keep],rejected_own,rejected_neighbor
