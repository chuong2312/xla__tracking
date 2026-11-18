"""Gán nhãn hộp người trên video và nội suy giữa các keyframe."""
import argparse
import csv
from pathlib import Path
from typing import Mapping
import cv2
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import yaml
from src.config import load_config
from src.utils import iter_video

Box = tuple[int, int, int, int]
def interpolate_annotations(keyframes: Mapping[int, Mapping[int, Box]]) -> list[tuple[int, int, int, int, int, int]]:
    """Nội suy tuyến tính theo ID chỉ giữa keyframe có nhãn của ID đó."""
    result: list[tuple[int,int,int,int,int,int]]=[]
    ids=sorted({pid for frame in keyframes.values() for pid in frame})
    for pid in ids:
        points=sorted((frame, box) for frame, entries in keyframes.items() if pid in entries for box in [entries[pid]])
        for (f0,b0),(f1,b1) in zip(points,points[1:]):
            for f in range(f0,f1+1):
                t=(f-f0)/(f1-f0)
                result.append((f,pid,*[round(a+(b-a)*t) for a,b in zip(b0,b1)]))
        if len(points)==1: result.append((points[0][0],pid,*points[0][1]))
    return sorted(set(result))

def write_gt(path: str | Path, rows: list[tuple[int,int,int,int,int,int]]) -> None:
    """Ghi CSV đúng schema ground truth tracking."""
    with Path(path).open("w",newline="",encoding="utf-8") as fp:
        writer=csv.writer(fp); writer.writerow(["frame","gt_id","x","y","w","h"]); writer.writerows(rows)

class Annotator:
    """Quản lý thao tác chuột và phím của giao diện gán nhãn."""
    def __init__(self) -> None: self.start=None; self.boxes: list[Box]=[]
    def mouse(self,event: int,x: int,y: int,flags: int,param: object) -> None:
        if event==cv2.EVENT_LBUTTONDOWN: self.start=(x,y)
        elif event==cv2.EVENT_LBUTTONUP and self.start is not None:
            x0,y0=self.start; self.boxes.append((min(x0,x),min(y0,y),abs(x-x0),abs(y-y0))); self.start=None

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--video",required=True); p.add_argument("--out",required=True); p.add_argument("--step",type=int); a=p.parse_args()
    config_path=Path(__file__).parents[1]/"config.yaml"; cfg=load_config(config_path)
    with config_path.open(encoding="utf-8") as fp: configured_step=yaml.safe_load(fp)["scene"].get("annotate_step",5)
    step=a.step if a.step is not None else configured_step
    if step<=0: raise ValueError("--step phải dương")
    frames=list(iter_video(a.video,(cfg.scene.resize_width,cfg.scene.resize_height))); keyframes: dict[int,dict[int,Box]]={}; ended: dict[int,int]={}; frame_ids=list(range(0,len(frames),step)); pos=0; ui=Annotator(); window="Annotate GT"; cv2.namedWindow(window); cv2.setMouseCallback(window,ui.mouse)
    while frame_ids:
        fi=frame_ids[pos]; canvas=frames[fi][1].copy()
        for pid,box in keyframes.get(fi,{}).items():
            x,y,w,h=box; cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,255,0),2); cv2.putText(canvas,str(pid),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
        cv2.putText(canvas,f"frame {fi} | drag box, number=ID, n/p/u/s/q; e=end ID",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
        cv2.imshow(window,canvas); key=cv2.waitKey(20)&0xff
        if key==ord('q'): break
        if key==ord('n'): pos=min(pos+1,len(frame_ids)-1); ui.boxes.clear()
        elif key==ord('p'): pos=max(0,pos-1); ui.boxes.clear()
        elif key==ord('u') and ui.boxes: ui.boxes.pop()
        elif key==ord('s'): write_gt(a.out,interpolate_annotations(keyframes))
        elif key==ord('e'):
            try: pid=int(input("gt_id hết từ frame hiện tại: ")); ended[pid]=fi
            except ValueError: print("gt_id không hợp lệ")
        elif ord('0')<=key<=ord('9') and ui.boxes:
            pid=key-ord('0'); keyframes.setdefault(fi,{})[pid]=ui.boxes.pop()
    cv2.destroyAllWindows(); write_gt(a.out,interpolate_annotations(keyframes))
if __name__=="__main__": main()
