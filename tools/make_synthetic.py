"""Sinh video tổng hợp và ground truth cho thử nghiệm tracker."""
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from src.utils import set_seed

def generate(scenario: str, frames: int, size: tuple[int,int], seed: int, out: Path, write_gt_reid: bool=False) -> None:
    """Tạo hai video tổng hợp, metadata và nhãn hộp theo từng frame."""
    if scenario not in {"cross","occlude","stop"}: raise ValueError("scenario phải là cross, occlude hoặc stop")
    if frames <= 0 or min(size) < 80: raise ValueError("frames phải dương và kích thước tối thiểu là 80")
    set_seed(seed); w,h=size; out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed); base=np.clip(110+rng.normal(0,5,(h,w,3)),0,255).astype(np.uint8)
    specs=[(0,(30,70,220)),(1,(50,210,70)),(2,(220,100,35))]; rows=[]
    writers=[]
    for cam in ("cam_a","cam_b"):
        folder=out/cam; folder.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(folder/"video.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),20.0,size)
        if not writer.isOpened(): raise RuntimeError(f"Không tạo được video tại {folder}")
        writers.append(writer)
    try:
        for f in range(frames):
            for ci in range(2):
                frame=base.copy() if ci==0 else np.clip(base.astype(float)*1.05+rng.normal(0,3,base.shape),0,255).astype(np.uint8)
                for pid,color in specs:
                    bw,bh=(34,85) if ci==0 else (40,100); t=f/max(1,frames-1)
                    if scenario=="cross": cx=int((.1+.8*t)*w) if pid==0 else int((.9-.8*t)*w) if pid==1 else int(w*(.5+.25*np.sin(t*6.28)))
                    else: cx=int((.12+.76*t)*w) if pid==0 else int(w*.5) if pid==1 else int(w*.78)
                    if scenario=="stop" and pid==0 and .4<t<.65: cx=int(w*.42)
                    cy=int(h*(.55 if pid!=2 else .38)); x=max(0,min(w-bw,cx-bw//2)); y=max(0,min(h-bh,cy-bh//2))
                    if scenario=="occlude" and pid==1 and .42<t<.55: continue
                    texture=rng.integers(0,55,(bh,bw,3),dtype=np.uint8)
                    patch=np.clip(np.array(color,dtype=np.int16)[None,None,:]+texture.astype(np.int16)-27,0,255).astype(np.uint8)
                    frame[y:y+bh,x:x+bw]=patch
                    if ci==0: rows.append((f,pid,x,y,bw,bh))
                writers[ci].write(frame)
    finally:
        for wr in writers: wr.release()
    with (out/"gt_tracking.csv").open("w",newline="",encoding="utf-8") as fp:
        wr=csv.writer(fp); wr.writerow(["frame","gt_id","x","y","w","h"]); wr.writerows(rows)
    if write_gt_reid:
        with (out/"gt_reid.csv").open("w",newline="",encoding="utf-8") as fp:
            wr=csv.writer(fp); wr.writerow(["person","cam_a_id","cam_b_id"]); wr.writerows((p,p,p) for p in range(3))
    for cam in ("cam_a","cam_b"):
        (out/cam/"metadata.json").write_text(json.dumps({"synthetic":True,"scenario":scenario,"seed":seed},indent=2),encoding="utf-8")

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--scenario",choices=["cross","occlude","stop"],default="cross"); parser.add_argument("--frames",type=int,default=300); parser.add_argument("--size",nargs=2,type=int,default=(640,480),metavar=("WIDTH","HEIGHT")); parser.add_argument("--seed",type=int,default=0); parser.add_argument("--out",type=Path,required=True); parser.add_argument("--write-gt-reid",action="store_true"); a=parser.parse_args(); generate(a.scenario,a.frames,tuple(a.size),a.seed,a.out,a.write_gt_reid)
if __name__=="__main__": main()
