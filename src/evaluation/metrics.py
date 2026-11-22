"""Đánh giá detection, tracking và nhận dạng lại từ các CSV của project."""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import yaml
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
# Trạng thái track được chấm điểm: confirmed (tracker cũ và mới), merged và static (chỉ tracker mới có).
SCORED_TRACK_STATES = {"confirmed", "merged", "static"}

def read_csv_rows(path: str | Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    """Đọc CSV, xác thực header bắt buộc; cho phép CSV rỗng chỉ khi vẫn có header."""
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"CSV '{path}' rỗng hoặc thiếu header")
            missing = [name for name in required if name not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV '{path}' thiếu cột: {', '.join(missing)}")
            return [row for row in reader]
    except OSError as exc:
        raise ValueError(f"Không đọc được CSV '{path}': {exc}") from exc

def _int(row: dict[str, str], key: str, source: str) -> int:
    try: return int(row[key])
    except (TypeError, ValueError) as exc: raise ValueError(f"Giá trị {key} không hợp lệ trong {source}: {row.get(key)!r}") from exc

def _box(row: dict[str, str], source: str) -> tuple[float,float,float,float]:
    try: box = tuple(float(row[k]) for k in ("x","y","w","h"))
    except (TypeError, ValueError, KeyError) as exc: raise ValueError(f"Box không hợp lệ trong {source}: {row}") from exc
    if box[2] < 0 or box[3] < 0: raise ValueError(f"Box có kích thước âm trong {source}: {row}")
    return box  # type: ignore[return-value]

def _load_boxes(path: str | Path, kind: str) -> dict[int,list[dict[str, Any]]]:
    required = ("frame","x","y","w","h") if kind == "det" else ("frame","gt_id","x","y","w","h")
    if kind == "track": required = ("frame","track_id","state","x","y","w","h")
    rows = read_csv_rows(path, required)
    by_frame: dict[int,list[dict[str,Any]]] = defaultdict(list)
    for row in rows:
        frame = _int(row,"frame",str(path))
        if kind == "track" and row["state"] not in SCORED_TRACK_STATES: continue
        identity = None if kind == "det" else _int(row,"track_id" if kind == "track" else "gt_id",str(path))
        by_frame[frame].append({"id":identity,"box":_box(row,str(path))})
    return by_frame

def iou(box_a: tuple[float,float,float,float], box_b: tuple[float,float,float,float]) -> float:
    """Tính diện tích giao chia diện tích hợp cho hai box dạng x,y,w,h."""
    ax,ay,aw,ah=box_a; bx,by,bw,bh=box_b
    if aw<=0 or ah<=0 or bw<=0 or bh<=0: return 0.0
    iw=max(0.0,min(ax+aw,bx+bw)-max(ax,bx)); ih=max(0.0,min(ay+ah,by+bh)-max(ay,by))
    inter=iw*ih; union=aw*ah+bw*bh-inter
    return inter/union if union>0 else 0.0

def _match_boxes(left: list[dict[str,Any]], right: list[dict[str,Any]], threshold: float) -> list[tuple[int,int]]:
    """Hungarian matching với chi phí 1-IoU, bỏ cặp dưới ngưỡng."""
    if not left or not right: return []
    overlaps=np.array([[iou(a["box"],b["box"]) for b in right] for a in left],dtype=float)
    rows,cols=linear_sum_assignment(1.0-overlaps)
    return [(int(i),int(j)) for i,j in zip(rows,cols) if overlaps[i,j] >= threshold]

def detection_pr(det_csv: str | Path, gt_csv: str | Path, iou_thr: float | None = None) -> dict[str,float|int]:
    """Tính precision=TP/(TP+FP), recall=TP/(TP+FN); Hungarian ghép riêng từng frame có GT.

    Detection ở frame không có hàng GT bị bỏ qua theo hợp đồng chỉ chấm frame có GT.
    Nếu file đáp án có cột occluded và keyframe (QĐ-15) thì chỉ chấm frame gán tay
    và không có box bị che nặng (XLA 3.3), xem `scored_frames`.
    Ghép được tính khi IoU lớn hơn hoặc bằng ngưỡng.
    Mẫu số bằng 0 cho precision/recall được quy ước là 0.
    """
    threshold=default_iou_threshold() if iou_thr is None else _validate_threshold(iou_thr)
    detections=_load_boxes(det_csv,"det"); ground_truth=_load_boxes(gt_csv,"gt")
    allowed=scored_frames(gt_csv)
    tp=fp=fn=0
    for frame,truths in ground_truth.items():
        if allowed is not None and frame not in allowed:
            continue
        found=detections.get(frame,[]); matches=_match_boxes(found,truths,threshold)
        tp+=len(matches); fp+=len(found)-len(matches); fn+=len(truths)-len(matches)
    return {"precision":tp/(tp+fp) if tp+fp else 0.0,"recall":tp/(tp+fn) if tp+fn else 0.0,"tp":tp,"fp":fp,"fn":fn}

def scored_frames(gt_csv: str | Path) -> set[int] | None:
    """Các frame được dùng để chấm precision/recall của detector (XLA 3.3).

    Một frame được chấm khi mọi box của nó đều gán tay (keyframe = 1) và không
    box nào bị che nặng (occluded = 2). Frame nội suy không được chấm vì box do
    máy đoán, không phải đáp án thật.
    Trả về None nếu file đáp án theo định dạng cũ (không có hai cột này): khi đó
    mọi frame có đáp án đều được chấm như trước.
    """
    rows = read_csv_rows(gt_csv, ("frame",))
    if not rows or "keyframe" not in rows[0] or "occluded" not in rows[0]:
        return None
    all_frames: set[int] = set()
    excluded: set[int] = set()
    for row in rows:
        frame = _int(row, "frame", str(gt_csv))
        all_frames.add(frame)
        if _int(row, "keyframe", str(gt_csv)) != 1 or _int(row, "occluded", str(gt_csv)) == 2:
            excluded.add(frame)
    return all_frames - excluded

def count_accuracy(det_csv: str | Path, manual_counts_csv: str | Path) -> dict[str,float|int]:
    """Tỷ lệ frame trong các đoạn đếm thủ công mà số hàng detection đúng bằng count."""
    detections=_load_boxes(det_csv,"det")
    ranges=read_csv_rows(manual_counts_csv,("frame_start","frame_end","count"))
    evaluated=correct=0; covered: set[int]=set()
    for row in ranges:
        start=_int(row,"frame_start",str(manual_counts_csv)); end=_int(row,"frame_end",str(manual_counts_csv)); count=_int(row,"count",str(manual_counts_csv))
        if end<start: raise ValueError("frame_end phải lớn hơn hoặc bằng frame_start")
        for frame in range(start,end+1):
            if frame in covered: raise ValueError(f"Frame {frame} nằm trong nhiều khoảng manual_counts")
            covered.add(frame); evaluated+=1
            correct += len(detections.get(frame,[])) == count
    return {"accuracy":correct/evaluated if evaluated else 0.0,"correct_frames":correct,"total_frames":evaluated}

def tracking_metrics(tracks_csv: str | Path, gt_csv: str | Path, iou_thr: float | None = None, mostly_tracked_ratio: float | None = None) -> dict[str,int|float]:
    """Đánh giá track confirmed bằng ghép Hungarian theo frame và IoU >= ngưỡng.

    ID switch đếm lần một GT ID được ghép với track ID khác lần ghép thành công trước đó,
    kể cả khi giữa hai lần có frame mất ghép. Fragmentation đếm mỗi lần một GT ID chuyển
    từ trạng thái từng được ghép sang mất ghép rồi được ghép lại. Purity mỗi track là số
    lần ghép với GT ID chiếm đa số chia cho tổng quan sát confirmed của track trong frame
    có GT (quan sát không ghép cũng nằm ở mẫu số); purity_mean là trung bình không trọng số
    trên các track quan sát được. Mostly tracked là số GT ID được ghép ở ít nhất tỷ lệ
    `mostly_tracked_ratio` số frame GT của ID đó. Mẫu số rỗng cho tỷ lệ được quy ước là 0.
    """
    threshold=default_iou_threshold() if iou_thr is None else _validate_threshold(iou_thr)
    ratio=default_mostly_tracked_ratio() if mostly_tracked_ratio is None else _validate_ratio(mostly_tracked_ratio)
    tracked=_load_boxes(tracks_csv,"track"); truth=_load_boxes(gt_csv,"gt")
    gt_ids={entry["id"] for entries in truth.values() for entry in entries}
    track_ids={entry["id"] for entries in tracked.values() for entry in entries}
    gt_total=Counter(); gt_matched=Counter(); gt_states: dict[int,list[bool]]=defaultdict(list)
    previous_match: dict[int,int]={}; switches=fragments=0
    track_total=Counter(); track_by_gt: dict[int,Counter[int]]=defaultdict(Counter)
    for frame, truths in sorted(truth.items()):
        detections=tracked.get(frame,[])
        matches=_match_boxes(truths,detections,threshold)
        matched_gt={i for i,_ in matches}; mapping={truths[i]["id"]:detections[j]["id"] for i,j in matches}
        for idx,gt in enumerate(truths):
            gid=gt["id"]; gt_total[gid]+=1; visible=idx in matched_gt; gt_matched[gid]+=int(visible); gt_states[gid].append(visible)
            if visible:
                tid=mapping[gid]
                if gid in previous_match and previous_match[gid]!=tid: switches+=1
                previous_match[gid]=tid; track_by_gt[tid][gid]+=1
        for det_idx,det in enumerate(detections):
            track_total[det["id"]]+=1
    for states in gt_states.values():
        was_matched=False; gap=False
        for visible in states:
            if visible:
                if was_matched and gap: fragments+=1
                was_matched=True; gap=False
            elif was_matched: gap=True
    purity_values=[max(track_by_gt[tid].values(),default=0)/total for tid,total in track_total.items()]
    mostly=sum(gt_matched[gid]/total>=ratio for gid,total in gt_total.items() if total)
    return {"id_switches":switches,"fragmentations":fragments,"purity_mean":float(np.mean(purity_values)) if purity_values else 0.0,"mostly_tracked":mostly,"n_gt_ids":len(gt_ids),"n_track_ids":len(track_ids)}

def reid_metrics(global_ids_csv: str | Path, gt_reid_csv: str | Path) -> dict[str,int|float|str]:
    """So sánh global ID hai track cục bộ trong từng cặp GT; thiếu/-1 là sai, total là số cặp GT."""
    mapping=_global_id_map(global_ids_csv); pairs=_gt_pairs(gt_reid_csv)
    correct=0
    for row in pairs:
        a=mapping.get(("cam_a",_int(row,"cam_a_id",str(gt_reid_csv))))
        b=mapping.get(("cam_b",_int(row,"cam_b_id",str(gt_reid_csv))))
        if a is not None and b is not None and a!=-1 and a==b: correct+=1
    total=len(pairs)
    return {"correct":correct,"total":total,"accuracy":correct/total if total else 0.0,"accuracy_str":f"{correct}/{total}"}

def rank1(scores_csv: str | Path, gt_reid_csv: str | Path, method: str) -> dict[str,int|float|str]:
    """Tính top-1 accuracy: mỗi cam_a_track chọn ứng viên có score cao nhất của method."""
    rows=read_csv_rows(scores_csv,("cam_a_track","cam_b_track","method","score")); pairs=_gt_pairs(gt_reid_csv)
    scores=[]
    for row in rows:
        if row["method"]!=method: continue
        try: score=float(row["score"])
        except ValueError as exc: raise ValueError(f"score không hợp lệ: {row['score']!r}") from exc
        scores.append((row["cam_a_track"],row["cam_b_track"],score))
    best: dict[str,tuple[str,float]]={}
    for a,b,score in scores:
        if a not in best or score>best[a][1]: best[a]=(b,score)
    expected={str(_int(row,"cam_a_id",str(gt_reid_csv))):str(_int(row,"cam_b_id",str(gt_reid_csv))) for row in pairs}
    correct=sum(a in best and best[a][0]==b for a,b in expected.items()); total=len(expected)
    return {"correct":correct,"total":total,"accuracy":correct/total if total else 0.0,"accuracy_str":f"{correct}/{total}"}

def confusion_matrix(global_ids_csv: str | Path, gt_reid_csv: str | Path) -> dict[str,dict[str,int]]:
    """Ma trận đếm GT person theo predicted global_id; '-1' là không ghép hoặc thiếu ID."""
    mapping=_global_id_map(global_ids_csv); matrix: dict[str,dict[str,int]]=defaultdict(lambda:defaultdict(int))
    for row in _gt_pairs(gt_reid_csv):
        person=row["person"]
        for cam,key in (("cam_a","cam_a_id"),("cam_b","cam_b_id")):
            track=_int(row,key,"gt_reid")
            predicted=mapping.get((cam,track),-1)
            matrix[person][str(predicted)]+=1
    return {person:dict(counts) for person,counts in matrix.items()}

def _global_id_map(path: str | Path) -> dict[tuple[str,int],int]:
    rows=read_csv_rows(path,("cam","track_id","global_id")); mapping={}
    for row in rows:
        key=(row["cam"],_int(row,"track_id",str(path))); value=_int(row,"global_id",str(path))
        if key in mapping and mapping[key]!=value: raise ValueError(f"Global ID bị khai báo trùng khác nhau: {key}")
        mapping[key]=value
    return mapping

def _gt_pairs(path: str | Path) -> list[dict[str,str]]:
    return read_csv_rows(path,("person","cam_a_id","cam_b_id"))

def _validate_threshold(value: float) -> float:
    if not 0<=value<=1: raise ValueError("IoU threshold phải nằm trong [0, 1]")
    return float(value)
def _validate_ratio(value: float) -> float:
    if not 0<=value<=1: raise ValueError("mostly_tracked_ratio phải nằm trong [0, 1]")
    return float(value)
def default_iou_threshold() -> float:
    """Đọc ngưỡng IoU mặc định từ section metrics của config.yaml."""
    config=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8")); return _validate_threshold(float(config["metrics"]["iou_threshold"]))
def default_mostly_tracked_ratio() -> float:
    """Đọc tỷ lệ mostly-tracked mặc định từ section metrics của config.yaml."""
    config=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8")); return _validate_ratio(float(config["metrics"]["mostly_tracked_ratio"]))

def _print_table(title: str, values: dict[str,Any]) -> None:
    print(title)
    width=max((len(key) for key in values),default=5)
    for key,value in values.items(): print(f"{key:<{width}} | {value}")

def main() -> None:
    """CLI metrics: det, track hoặc reid với kết quả dạng bảng."""
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    det=sub.add_parser("det"); det.add_argument("detections"); det.add_argument("gt"); det.add_argument("--iou-thr",type=float)
    track=sub.add_parser("track"); track.add_argument("tracks"); track.add_argument("gt"); track.add_argument("--iou-thr",type=float)
    reid=sub.add_parser("reid"); reid.add_argument("global_ids"); reid.add_argument("gt_reid"); reid.add_argument("--scores"); reid.add_argument("--method")
    args=parser.parse_args()
    if args.command=="det": _print_table("Detection metrics",detection_pr(args.detections,args.gt,args.iou_thr))
    elif args.command=="track":
        from src.evaluation.mot_metrics import mot_metrics
        _print_table("Tracking metrics",tracking_metrics(args.tracks,args.gt,args.iou_thr))
        _print_table("MOT metrics (MOTA, MOTP, IDF1)",mot_metrics(args.tracks,args.gt,args.iou_thr))
    else:
        _print_table("Re-ID metrics",reid_metrics(args.global_ids,args.gt_reid))
        _print_table("Confusion matrix",confusion_matrix(args.global_ids,args.gt_reid))
        if args.scores:
            if not args.method: parser.error("--scores cần --method")
            _print_table(f"Rank-1 ({args.method})",rank1(args.scores,args.gt_reid,args.method))
if __name__=="__main__": main()
