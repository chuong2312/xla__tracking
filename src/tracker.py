"""Theo dõi nhiều đối tượng bằng Lucas-Kanade và ghép box theo IoU."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.utils import iou

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
        self._tracks: list[Track] = []
        self._next_id = 1
        self._previous_gray: np.ndarray | None = None
        self._frame_index = 0
        self._lost_this_frame: list[Track] = []

    def update(self, frame_gray: np.ndarray, detections: list[tuple[int, int, int, int]]) -> list[Track]:
        """Cập nhật track cho frame grayscale hiện tại và các detection cùng tọa độ."""
        if frame_gray.ndim != 2 or frame_gray.dtype != np.uint8:
            raise ValueError("frame_gray phải là ảnh grayscale uint8")
        height, width = frame_gray.shape
        valid_detections = [_clip_box(_validate_box(box), width, height) for box in detections]
        valid_detections = [box for box in valid_detections if box is not None]
        self._lost_this_frame = []
        predictions: list[tuple[int,int,int,int]] = []
        flow_points: list[np.ndarray] = []
        flow_sufficient: list[bool] = []
        if self._previous_gray is None:
            predictions = [track.box for track in self._tracks]
            flow_points = [track.points for track in self._tracks]
            flow_sufficient = [False for _ in self._tracks]
        else:
            for track in self._tracks:
                predicted, points, sufficient = self._predict_track(track, frame_gray, width, height)
                predictions.append(predicted); flow_points.append(points); flow_sufficient.append(sufficient)
        matches = self._assign(predictions, valid_detections)
        matched_track = {i for i, _ in matches}; matched_detection = {j for _, j in matches}
        updated: list[Track] = []
        for index, track in enumerate(self._tracks):
            track.age += 1
            previous_box = track.box
            if index in matched_track:
                det_index = next(j for i,j in matches if i == index)
                track.box = _blend_box(predictions[index], valid_detections[det_index], self.box_update_alpha, width, height)
                track.hits += 1; track.misses = 0; track.consecutive_hits += 1; track.exit_extrapolations = 0
                if track.consecutive_hits >= self.confirm_hits: track.state = "confirmed"
                track.points = _points_in_box(flow_points[index], track.box)
                if self._should_refresh(track): track.points = self._find_points(frame_gray, track.box)
                updated.append(track)
            else:
                track.box = predictions[index]
                track.misses += 1
                if track.state == "tentative": track.consecutive_hits = 0
                if track.misses > self.max_misses or _outside(track.box, width, height):
                    track.box = _clip_box(track.box, width, height) or previous_box
                    track.state = "lost"; self._lost_this_frame.append(track)
                else:
                    track.points = _points_in_box(flow_points[index], track.box)
                    if self._should_refresh(track): track.points = self._find_points(frame_gray, track.box)
                    updated.append(track)
        self._tracks = updated
        for index, box in enumerate(valid_detections):
            if index not in matched_detection:
                track = self._new_track(frame_gray, box)
                self._tracks.append(track)
        self._previous_gray = frame_gray.copy()
        self._frame_index += 1
        return list(self._tracks) + list(self._lost_this_frame)

    def _predict_track(self, track: Track, current_gray: np.ndarray, width: int, height: int) -> tuple[tuple[int,int,int,int],np.ndarray,bool]:
        points = track.points
        empty = np.empty((0,1,2), dtype=np.float32)
        if len(points) == 0:
            return self._extrapolate_exit(track, width, height, empty)
        forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(self._previous_gray, current_gray, points, None, winSize=self.win_size, maxLevel=self.max_level, criteria=self.criteria)
        if forward is None or status_forward is None:
            return self._extrapolate_exit(track, width, height, empty)
        old = points.reshape(-1,2); new = forward.reshape(-1,2); valid = status_forward.reshape(-1).astype(bool)
        valid &= np.isfinite(old).all(axis=1) & np.isfinite(new).all(axis=1)
        valid &= (new[:,0] >= 0) & (new[:,0] < width) & (new[:,1] >= 0) & (new[:,1] < height)
        if self.use_fb_check and valid.any():
            candidate = new[valid].reshape(-1,1,2).astype(np.float32)
            backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(current_gray, self._previous_gray, candidate, None, winSize=self.win_size, maxLevel=self.max_level, criteria=self.criteria)
            if backward is None or status_backward is None:
                valid[:] = False
            else:
                valid_indices = np.flatnonzero(valid); returned = backward.reshape(-1,2); back_status = status_backward.reshape(-1).astype(bool)
                errors = np.linalg.norm(old[valid_indices] - returned, axis=1)
                valid[valid_indices] &= back_status & np.isfinite(returned).all(axis=1) & np.isfinite(errors) & (errors <= self.fb_thresh)
        good_old, good_new = old[valid], new[valid]
        good_points = good_new.reshape(-1,1,2).astype(np.float32)
        if len(good_new) < self.min_points:
            return self._extrapolate_exit(track, width, height, good_points)
        displacement = np.median(good_new - good_old, axis=0)
        track.last_displacement = (float(displacement[0]), float(displacement[1]))
        track.exit_extrapolations = 0
        x,y,w,h=track.box
        moved=(int(round(x+float(displacement[0]))),int(round(y+float(displacement[1]))),w,h)
        clipped=_clip_box(moved,width,height)
        if clipped is None: return moved, good_points, True
        return clipped, good_points, True


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
        if not predictions or not detections: return []
        overlaps=np.array([[iou(a,b) for b in detections] for a in predictions],dtype=float)
        if self.matcher == "greedy":
            candidates=sorted(((1.0-overlaps[i,j],i,j) for i in range(len(predictions)) for j in range(len(detections)) if overlaps[i,j] >= self.iou_gate))
            pairs=[]; used_i=set(); used_j=set()
            for _,i,j in candidates:
                if i not in used_i and j not in used_j: pairs.append((i,j)); used_i.add(i); used_j.add(j)
            return pairs
        cost=1.0-overlaps; cost[overlaps < self.iou_gate] = 1e6
        rows,cols=linear_sum_assignment(cost)
        return [(int(i),int(j)) for i,j in zip(rows,cols) if overlaps[i,j] >= self.iou_gate]

    def _new_track(self, frame_gray: np.ndarray, box: tuple[int,int,int,int]) -> Track:
        track=Track(self._next_id,"tentative",box,np.empty((0,1,2),dtype=np.float32))
        self._next_id+=1; track.points=self._find_points(frame_gray,box)
        if self.confirm_hits <= 1: track.state="confirmed"
        return track

    def _find_points(self, frame_gray: np.ndarray, box: tuple[int,int,int,int]) -> np.ndarray:
        x,y,w,h=box; mask=np.zeros(frame_gray.shape,dtype=np.uint8); mask[y:y+h,x:x+w]=255
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
def _blend_box(prediction: tuple[int,int,int,int],detection: tuple[int,int,int,int],alpha: float,width: int,height: int) -> tuple[int,int,int,int]:
    values=tuple(int(round(alpha*p+(1-alpha)*d)) for p,d in zip(prediction,detection))
    return _clip_box(values,width,height) or detection


def _points_in_box(points: np.ndarray, box: tuple[int,int,int,int]) -> np.ndarray:
    """Giữ feature points nằm trong box cập nhật của Track."""
    if len(points)==0: return np.empty((0,1,2),dtype=np.float32)
    x,y,w,h=box; flat=points.reshape(-1,2)
    keep=(flat[:,0]>=x)&(flat[:,0]<x+w)&(flat[:,1]>=y)&(flat[:,1]<y+h)&np.isfinite(flat).all(axis=1)
    return flat[keep].reshape(-1,1,2).astype(np.float32,copy=False)
