"""Tracker mới của hệ thống (bước D): Kalman + Lucas-Kanade + Hungarian + máy trạng thái (QĐ-05).

Mỗi frame, theo thứ tự:
1. Dự đoán: Kalman predict. Nếu Lucas-Kanade đo được độ dịch (đủ điểm tin cậy), vị trí theo
   LK được dùng làm một phép đo cho Kalman. Khi hai người dính nhau, chỉ người đứng trước
   giữ điểm LK; người đứng sau bỏ điểm để không bị "hút" sang người kia (XLA 2.5).
2. Phân loại detection (src/blob_analysis.py): gộp các mảnh tách, tìm blob gộp.
3. Ghép lại các track vừa ra khỏi blob gộp (trong `recover_window_frames` frame sau khi tách)
   bằng nhiều tín hiệu (src/recovery.py), dùng Hungarian.
4. Ghép các track còn lại với detection sạch: Hungarian trên 1 − IoU + w·(1 − độ giống màu)
   (src/assignment.py). Màu giúp không tráo tên khi hai người ở sát nhau.
5. Cập nhật trạng thái (src/track.py), rồi tạo track mới từ detection sạch còn thừa. Nếu có
   mô hình chiều cao (src/height_model.py), detection quá thấp so với một người (chỉ là mảnh
   đầu hoặc chân) không được tạo track mới (QĐ-27).
Tracker cũ (src/tracker.py) được giữ lại để so sánh.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from src.assignment import assign, iou_cost
from src.blob_analysis import coverage, find_fragments, find_merged, join_split_detections
from src.kalman import ConstantVelocityKalman
from src.optical_flow import EMPTY, find_points, median_shift, track_points
from src.color_features import color_similarity
from src.height_model import HeightModel, tall_enough
from src.recovery import colors_of, make_template, recovery_cost
from src.track import Track
from src.utils import iou


class PeopleTracker:
    """Giữ danh sách track của một camera và cập nhật chúng theo từng frame."""

    def __init__(self, cfg: Mapping, kalman_cfg: Mapping, features_cfg: Mapping, heights: HeightModel | None = None):
        self._cfg = cfg
        self._heights = heights
        self._kalman_cfg = kalman_cfg
        self._features_cfg = features_cfg
        self.tracks: list[Track] = []
        self._next_id = 1
        self._previous_gray: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray, detections: Sequence[Sequence[int]], foreground: np.ndarray | None) -> list[Track]:
        """Xử lý một frame; trả về các track còn hoạt động cùng các track vừa kết thúc (state = lost)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        for track in self.tracks:
            self._predict(track, gray)
        predicted = {track.id: track.box for track in self.tracks}
        detections = join_split_detections([tuple(d) for d in detections], predicted, float(self._cfg["split_coverage"]))
        self._drop_fragments(detections, predicted)
        sticky = {t.id for t in self.tracks if t.state == "merged"}
        merged_blobs = find_merged(detections, predicted, float(self._cfg["merge_coverage"]),
                                   sticky, float(self._cfg["merge_keep_coverage"]))
        merged_ids = set().union(*merged_blobs.values()) if merged_blobs else set()
        clean = [d for i, d in enumerate(detections) if i not in merged_blobs]
        matches = self._associate(clean, merged_ids, gray, frame_bgr)
        for track in self.tracks:
            track.age += 1
            track.frames_since_merge += 1
            if track.id in matches:
                self._on_match(track, clean[matches[track.id]], gray, frame_bgr, foreground)
            elif track.id in merged_ids and track.state != "tentative":
                self._on_merged(track, merged_blobs, predicted)
                if track.state == "merged" and self._cfg["template_search"]:
                    self._locate_by_template(track, gray)
            else:
                self._on_miss(track)
            if self._outside(track, gray.shape):
                track.state = "lost"
        self._create_tracks(clean, matches, gray, frame_bgr, foreground)
        self._previous_gray = gray
        finished = [t for t in self.tracks if t.state == "lost"]
        self.tracks = [t for t in self.tracks if t.state != "lost"]
        return self.tracks + finished

    def _drop_fragments(self, detections: list, predicted: dict[int, tuple]) -> None:
        """Bỏ track chỉ là mảnh (đầu hoặc chân) của một người đã có track, xem find_fragments."""
        fragments = find_fragments(detections, predicted, float(self._cfg["merge_coverage"]),
                                   float(self._cfg["fragment_max_vertical_overlap"]))
        self.tracks = [t for t in self.tracks if t.id not in fragments]
        for track_id in fragments:
            del predicted[track_id]

    def _predict(self, track: Track, gray: np.ndarray) -> None:
        """Kalman dự đoán; nếu LK đo được độ dịch thì dùng làm phép đo nguồn "lk"."""
        before = track.center
        track.kalman.predict()
        track.last_shift = None
        if self._previous_gray is None or len(track.points) == 0:
            return
        old, new = track_points(self._previous_gray, gray, track.points, self._cfg)
        track.points = new.reshape(-1, 1, 2).astype(np.float32)
        shift = median_shift(old, new, int(self._cfg["min_points"]))
        if shift is not None:
            track.last_shift = shift
            track.kalman.correct(before[0] + shift[0], before[1] + shift[1], "lk")

    def _associate(self, clean: list, merged_ids: set[int], gray: np.ndarray, frame_bgr: np.ndarray) -> dict[int, int]:
        """Ghép track với detection sạch; trả về {track id: chỉ số detection}."""
        cfg = self._cfg
        matches: dict[int, int] = {}
        window = int(cfg["recover_window_frames"])
        returning = [t for t in self.tracks if t.id not in merged_ids and t.state != "tentative"
                     and t.frames_since_merge <= window]
        available = list(range(len(clean)))
        if returning and available:
            cost = np.array([[recovery_cost(t, clean[j], gray, frame_bgr, cfg, self._features_cfg) for j in available] for t in returning])
            for i, j in assign(cost, max_cost=float(cfg["recover_max_cost"])):
                matches[returning[i].id] = available[j]
        available = [j for j in available if j not in matches.values()]
        normal = [t for t in self.tracks if t.id not in merged_ids and t.id not in matches]
        if normal and available:
            gate = float(cfg["iou_gate"])
            weight = float(cfg["appearance_weight"])
            cost = iou_cost([t.box for t in normal], [clean[j] for j in available], gate)
            if weight > 0:
                cost += weight * self._color_distance(normal, [clean[j] for j in available], frame_bgr)
            for i, j in assign(cost, max_cost=1.0 - gate + weight):
                matches[normal[i].id] = available[j]
        return matches

    def _color_distance(self, tracks: list[Track], boxes: list, frame_bgr: np.ndarray) -> np.ndarray:
        """Bảng 1 − độ giống màu giữa track và detection; track chưa có màu thì cho 0 (không phạt)."""
        detection_colors = [colors_of(frame_bgr, box, self._features_cfg) for box in boxes]
        table = np.zeros((len(tracks), len(boxes)))
        for i, track in enumerate(tracks):
            if track.colors is None or track.state == "tentative":
                continue
            for j, colors in enumerate(detection_colors):
                table[i, j] = 1.0 - color_similarity(track.colors, colors)
        return table

    def _update_appearance(self, track: Track, gray: np.ndarray, frame_bgr: np.ndarray, box: Sequence[int]) -> None:
        """Cập nhật ảnh mẫu và màu bằng trung bình trượt: mới = (1 − r)·cũ + r·hiện tại.

        Cập nhật từ từ để một lần ghép nhầm không làm track "đổi danh tính" ngay lập tức.
        """
        rate = float(self._cfg["appearance_update_rate"])
        template = make_template(gray, box)
        colors = colors_of(frame_bgr, box, self._features_cfg)
        if track.template is None or template is None or rate >= 1.0:
            track.template = template if template is not None else track.template
        else:
            blended = (1 - rate) * track.template.astype(np.float32) + rate * template.astype(np.float32)
            track.template = blended.astype(np.uint8)
        if track.colors is None or colors is None or rate >= 1.0:
            track.colors = colors if colors is not None else track.colors
        else:
            track.colors = {part: ((1 - rate) * track.colors[part] + rate * colors[part]).astype(np.float32) for part in colors}

    def _size_is_plausible(self, track: Track, box: Sequence[int]) -> bool:
        """Detection có đáng tin để lấy vị trí và kích thước không.

        Không đáng tin khi cả hai điều cùng đúng:
        - diện tích to hơn track quá `max_size_change` lần (detection đã dính thêm người khác),
          hoặc nhỏ hơn 1/max_size_change lần khi bật `guard_shrink` (người bị che một phần);
        - detection chồng lên box của một người khác đang theo dõi (≥ contamination_coverage).
        """
        if track.state == "tentative":
            return True
        ratio = (box[2] * box[3]) / max(track.size[0] * track.size[1], 1.0)
        limit = float(self._cfg["max_size_change"])
        lower = 1.0 / limit if self._cfg["guard_shrink"] else 0.0
        if lower <= ratio <= limit:
            return True
        threshold = float(self._cfg["contamination_coverage"])
        others = [t.box for t in self.tracks if t.id != track.id and t.state != "tentative"]
        return not any(coverage(other, box) >= threshold for other in others)

    def _on_match(self, track: Track, box: Sequence[int], gray: np.ndarray, frame_bgr: np.ndarray, foreground) -> None:
        """Có detection sạch: hiệu chỉnh Kalman, cập nhật kích thước, ảnh mẫu, màu và điểm LK.

        Nếu detection có kích thước bất thường thì vẫn tính là còn thấy người, nhưng chỉ dựa
        vào Lucas-Kanade và Kalman, không lấy vị trí, kích thước, ảnh mẫu hay màu từ detection đó.
        """
        plausible = self._size_is_plausible(track, box)
        x, y, w, h = box
        if plausible:
            track.kalman.correct(x + w / 2, y + h / 2, "detection")
            a = float(self._cfg["size_smoothing"])
            track.size = ((1 - a) * track.size[0] + a * w, (1 - a) * track.size[1] + a * h)
        track.hits += 1
        track.consecutive_hits += 1
        track.misses = 0
        track.merged_frames = 0
        track.static_frames = 0
        if track.state in ("merged", "static"):
            track.state = "confirmed"
        enough_hits = track.consecutive_hits >= int(self._cfg["confirm_hits"])
        moved = track.moved_from_start() >= float(self._cfg["min_confirm_displacement_px"])
        if track.state == "tentative" and enough_hits and moved:
            track.state = "confirmed"
        if not plausible:
            return
        self._update_appearance(track, gray, frame_bgr, box)
        if track.age % int(self._cfg["refresh_every"]) == 0 or len(track.points) < int(self._cfg["min_points"]):
            others = [t.box for t in self.tracks if t.id != track.id]
            track.points = find_points(gray, box, foreground, others, self._cfg)

    def _on_merged(self, track: Track, merged_blobs: dict[int, set[int]], predicted: dict[int, tuple]) -> None:
        """Nằm trong blob gộp: không lấy vị trí từ detection, không chọn thêm điểm LK (XLA 2.5).

        Người đứng trước (đáy box thấp nhất trong ảnh, tức gần camera nhất) vẫn được nhìn thấy
        đầy đủ nên giữ nguyên điểm LK. Người đứng sau bỏ hết điểm LK và chỉ đi theo vận tốc cũ
        của Kalman, vì điểm của họ rất dễ trượt sang người đứng trước (XLA 2.4, 2.5).
        """
        track.state = "merged"
        track.merged_frames += 1
        track.frames_since_merge = 0
        if track.merged_frames > int(self._cfg["merge_max_frames"]):
            track.state = "lost"
            return
        group = next(ids for ids in merged_blobs.values() if track.id in ids)
        front_id = max(group, key=lambda tid: predicted[tid][1] + predicted[tid][3])
        if track.id != front_id:
            track.points = EMPTY

    def _locate_by_template(self, track: Track, gray: np.ndarray) -> None:
        """Trong blob gộp, tìm lại người bằng so khớp ảnh mẫu (NCC) quanh vị trí dự đoán.

        Ảnh mẫu chụp lúc người còn đi riêng, phóng về đúng cỡ box hiện tại, rồi trượt trong vùng
        box nới thêm `template_search_margin` × cỡ box. Chỗ có NCC cao nhất, nếu ≥ template_min_score,
        được dùng làm một phép đo cho Kalman. Nhờ vậy vị trí của từng người không trôi dần trong
        lúc dính nhau, và khi tách ra thì ít bị tráo tên hơn.
        """
        if track.template is None:
            return
        w, h = int(round(track.size[0])), int(round(track.size[1]))
        if w < 8 or h < 16:
            return
        margin = float(self._cfg["template_search_margin"])
        x, y, _, _ = track.box
        left, top = max(0, int(x - margin * w)), max(0, int(y - margin * h))
        right = min(gray.shape[1], int(x + w + margin * w))
        bottom = min(gray.shape[0], int(y + h + margin * h))
        if right - left <= w or bottom - top <= h:
            return
        template = cv2.resize(track.template, (w, h), interpolation=cv2.INTER_LINEAR)
        scores = cv2.matchTemplate(gray[top:bottom, left:right], template, cv2.TM_CCOEFF_NORMED)
        _, best, _, (dx, dy) = cv2.minMaxLoc(scores)
        if np.isfinite(best) and best >= float(self._cfg["template_min_score"]):
            track.kalman.correct(left + dx + w / 2.0, top + dy + h / 2.0, "lk")

    def _on_miss(self, track: Track) -> None:
        """Không có detection: track mới thì bỏ; LK cho thấy đứng yên thì sang static; còn lại tăng misses."""
        cfg = self._cfg
        if track.state == "tentative":
            track.state = "lost"
            return
        if track.state == "merged":
            track.state = "confirmed"
        shift = track.last_shift
        if shift is not None and np.hypot(*shift) <= float(cfg["static_shift_px"]) and len(track.points) >= int(cfg["min_points"]):
            track.state = "static"
            track.static_frames += 1
            if track.static_frames > int(cfg["static_max_frames"]):
                track.state = "lost"
            return
        if track.state == "static":
            track.state = "confirmed"
        track.misses += 1
        if track.misses > int(cfg["max_misses"]):
            track.state = "lost"

    def _create_tracks(self, clean: list, matches: dict[int, int], gray: np.ndarray, frame_bgr: np.ndarray, foreground) -> None:
        """Tạo track tentative từ detection sạch chưa được ghép.

        Không tạo nếu detection chồng nhiều (IoU ≥ new_track_max_iou) lên một track CHƯA được
        ghép ở frame này, vì rất có thể đó chính là người của track ấy. Nếu track kia đã được ghép
        với detection khác rồi thì detection còn thừa là một người khác, ví dụ một người vừa tách
        khỏi nhóm đi cùng.
        """
        limit = float(self._cfg["new_track_max_iou"])
        used = set(matches.values())
        unmatched_boxes = [t.box for t in self.tracks if t.id not in matches]
        for index, box in enumerate(clean):
            if index in used or any(iou(box, other) >= limit for other in unmatched_boxes):
                continue
            if not self._may_start_track(box, gray.shape):
                continue
            x, y, w, h = box
            center = (x + w / 2, y + h / 2)
            track = Track(self._next_id, ConstantVelocityKalman(*center, self._kalman_cfg), (float(w), float(h)), center)
            self._next_id += 1
            track.template = make_template(gray, box)
            track.colors = colors_of(frame_bgr, box, self._features_cfg)
            track.points = find_points(gray, box, foreground, [t.box for t in self.tracks], self._cfg)
            self.tracks.append(track)
            unmatched_boxes.append(track.box)

    def _may_start_track(self, box: Sequence[int], shape: tuple[int, int]) -> bool:
        """Detection có đủ cao để tạo track mới không; luôn đúng khi chưa có mô hình chiều cao."""
        if self._heights is None:
            return True
        return tall_enough(box, self._heights, float(self._cfg["new_track_min_height_ratio"]), shape,
                           float(self._cfg["new_track_border_px"]))

    @staticmethod
    def _outside(track: Track, shape: tuple[int, int]) -> bool:
        """Box đã ra hẳn ngoài ảnh."""
        x, y, w, h = track.box
        height, width = shape
        return x + w <= 0 or y + h <= 0 or x >= width or y >= height
