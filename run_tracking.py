"""Chạy detector và optical-flow tracker rồi ghi tracks.csv."""
import argparse
import csv
from pathlib import Path
import cv2
import yaml
from src.config import load_config
from src.detector import Detector
from src.tracker import Tracker
from src.utils import iter_video

def main() -> None:
    """Đọc từng frame một lần, phát hiện, theo dõi và xuất overlay tùy chọn."""
    parser=argparse.ArgumentParser(); parser.add_argument("--video",required=True); parser.add_argument("--out",required=True); parser.add_argument("--debug-video"); parser.add_argument("--config",default="config.yaml"); args=parser.parse_args()
    config_path=Path(args.config); cfg=load_config(config_path)
    with config_path.open(encoding="utf-8") as stream: raw=yaml.safe_load(stream)
    detector=Detector({"detector":cfg.detector,"scene":raw["scene"]}); tracker=Tracker(cfg)
    size=(cfg.scene.resize_width,cfg.scene.resize_height); output=Path(args.out); output.parent.mkdir(parents=True,exist_ok=True)
    debug_writer=None
    try:
        with output.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream); writer.writerow(["frame","track_id","state","x","y","w","h","n_points"])
            for frame_idx,frame in iter_video(args.video,size):
                detections=detector.detect(frame)
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                tracks=tracker.update(gray,detections,frame_bgr=frame)
                for track in tracks:
                    x,y,w,h=track.box; writer.writerow([frame_idx,track.id,track.state,x,y,w,h,len(track.points)])
                if args.debug_video:
                    if debug_writer is None:
                        debug_path=Path(args.debug_video); debug_path.parent.mkdir(parents=True,exist_ok=True)
                        source=cv2.VideoCapture(args.video); fps=source.get(cv2.CAP_PROP_FPS); source.release()
                        if fps<=0: raise ValueError(f"Không đọc được FPS hợp lệ: {args.video}")
                        debug_writer=cv2.VideoWriter(str(debug_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,size)
                        if not debug_writer.isOpened(): raise RuntimeError(f"Không tạo được debug video: {debug_path}")
                    overlay=frame.copy()
                    for track in tracks:
                        x,y,w,h=track.box; color=(0,200,0) if track.state=="confirmed" else (0,190,255) if track.state=="tentative" else (0,0,255)
                        cv2.rectangle(overlay,(x,y),(x+w,y+h),color,2); cv2.putText(overlay,f"ID {track.id} {track.state}",(x,max(15,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
                        if len(track.points):
                            for px,py in track.points.reshape(-1,2): cv2.circle(overlay,(int(round(px)),int(round(py))),2,(255,100,0),-1)
                    debug_writer.write(overlay)
    finally:
        if debug_writer is not None: debug_writer.release()
if __name__=="__main__": main()
