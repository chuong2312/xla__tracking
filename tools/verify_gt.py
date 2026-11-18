"""Vẽ nhãn GT lên video và báo các vấn đề hình học/chuỗi ID."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import cv2
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config
from src.utils import iter_video

def read_gt(path: str | Path) -> list[tuple[int,int,int,int,int,int]]:
    """Đọc các hàng GT tracking từ CSV."""
    with Path(path).open(newline="",encoding="utf-8") as fp:
        return [(int(r["frame"]),int(r["gt_id"]),int(r["x"]),int(r["y"]),int(r["w"]),int(r["h"])) for r in csv.DictReader(fp)]

def validate_gt(rows: list[tuple[int,int,int,int,int,int]], width: int, height: int) -> list[str]:
    """Trả cảnh báo box sai biên/kích thước và ID có frame bị khuyết."""
    warnings=[]; by_id: dict[int,list[int]]=defaultdict(list)
    for f,pid,x,y,w,h in rows:
        by_id[pid].append(f)
        if w<=0 or h<=0 or x<0 or y<0 or x+w>width or y+h>height: warnings.append(f"Box ngoài khung hoặc suy biến: frame={f}, gt_id={pid}, box={(x,y,w,h)}")
    for pid,fs in by_id.items():
        uniq=sorted(set(fs))
        if any(b-a>1 for a,b in zip(uniq,uniq[1:])): warnings.append(f"gt_id={pid} có lỗ hổng frame")
    return warnings

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--video",required=True); p.add_argument("--gt",required=True); p.add_argument("--out",required=True); a=p.parse_args()
    cfg=load_config(Path(__file__).parents[1]/"config.yaml"); size=(cfg.scene.resize_width,cfg.scene.resize_height); rows=read_gt(a.gt); warnings=validate_gt(rows,*size); grouped=defaultdict(list)
    for row in rows: grouped[row[0]].append(row)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); writer=None; counts=defaultdict(set)
    source=cv2.VideoCapture(a.video)
    if not source.isOpened():
        source.release()
        raise ValueError(f"Không mở được video: {a.video}")
    fps=source.get(cv2.CAP_PROP_FPS)
    source.release()
    if fps <= 0: raise ValueError(f"Không đọc được FPS hợp lệ từ video: {a.video}")
    try:
        for frame_idx,frame in iter_video(a.video,size):
            if writer is None:
                writer=cv2.VideoWriter(a.out,cv2.VideoWriter_fourcc(*"mp4v"),fps,size)
                if not writer.isOpened(): raise RuntimeError(f"Không tạo được video: {a.out}")
            for _,pid,x,y,w,h in grouped.get(frame_idx,[]):
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2); cv2.putText(frame,str(pid),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2); counts[pid].add(frame_idx)
            writer.write(frame)
    finally:
        if writer is not None: writer.release()
    print(f"Số gt_id: {len(counts)}")
    for pid,fs in sorted(counts.items()): print(f"gt_id={pid}: {len(fs)} frame")
    for warning in warnings: print(f"CẢNH BÁO: {warning}")
if __name__=="__main__": main()
