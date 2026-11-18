"""Kiểm thử LK tracker bằng ảnh tổng hợp có texture xác định."""
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pytest
from src.tracker import Tracker

def config(**changes):
    values={"winSize":[21,21],"maxLevel":2,"criteria":[3,30,0.01],"use_fb_check":True,"fb_thresh":1.0,"min_points":3,"max_corners":80,"quality_level":0.01,"min_distance":3.0,"block_size":5,"refresh_every":3,"matcher":"hungarian","iou_gate":0.05,"confirm_hits":2,"max_misses":2,"box_update_alpha":0.0}
    values.update(changes); return {"tracker":values}

def textured_frame(objects,seed=23):
    rng=np.random.default_rng(seed); frame=np.zeros((120,180),np.uint8)
    for x,y,w,h in objects:
        patch=rng.integers(35,240,(h,w),dtype=np.uint8); frame[y:y+h,x:x+w]=patch
    return frame

def test_single_textured_box_moves_and_keeps_id_with_real_lk():
    tracker=Tracker(config()); first=textured_frame([(20,30,35,45)]); second=textured_frame([(24,32,35,45)]); third=textured_frame([(28,34,35,45)])
    a=tracker.update(first,[(20,30,35,45)]); b=tracker.update(second,[(24,32,35,45)]); c=tracker.update(third,[(28,34,35,45)])
    assert a[0].id==b[0].id==c[0].id==1
    assert b[0].box[0]>=23 and c[0].box[0]>=27
    assert b[0].state=="confirmed"
    assert c[0].points.dtype==np.float32 and c[0].points.ndim==3 and c[0].points.shape[1:]==(1,2)

def test_two_separated_objects_keep_distinct_ids():
    tracker=Tracker(config()); objects=[(15,20,30,40),(120,60,30,40)]
    detections=[(x,y,w,h) for x,y,w,h in objects]
    first=textured_frame(objects); second=textured_frame([(x+2,y,w,h) for x,y,w,h in objects])
    tracks1=tracker.update(first,detections); tracks2=tracker.update(second,[(x+2,y,w,h) for x,y,w,h in objects])
    assert {t.id for t in tracks1}=={1,2} and {t.id for t in tracks2}=={1,2}
    assert len({t.id for t in tracks2})==2

def test_consecutive_hit_lifecycle_and_miss_resets_tentative():
    tr=Tracker(config(confirm_hits=3,max_misses=2)); frame=textured_frame([(20,20,30,40)])
    assert tr.update(frame,[(20,20,30,40)])[0].state=="tentative"
    assert tr.update(frame,[(20,20,30,40)])[0].state=="tentative"
    assert tr.update(frame,[(20,20,30,40)])[0].state=="confirmed"
    tr=Tracker(config(confirm_hits=3,max_misses=2)); tr.update(frame,[(20,20,30,40)]); assert tr.update(frame,[])[0].state=="tentative"
    track=tr.update(frame,[(20,20,30,40)])[0]; assert track.state=="tentative" and track.consecutive_hits==1

def test_lost_track_is_reported_then_id_is_never_reused():
    tr=Tracker(config(max_misses=1)); frame=textured_frame([(20,20,30,40)])
    old=tr.update(frame,[(20,20,30,40)])[0]; tr.update(frame,[]); result=tr.update(frame,[])
    assert any(t.state=="lost" and t.id==old.id for t in result)
    new=tr.update(frame,[(100,20,30,40)])[0]; assert new.id>old.id

def test_fb_check_filters_bad_backward_return(monkeypatch):
    tr=Tracker(config(min_points=1)); previous=np.zeros((40,40),np.uint8); current=previous.copy(); tr._previous_gray=previous
    from src.tracker import Track
    pts=np.array([[[10.,10.]],[[20.,20.]]],np.float32); track=Track(1,"confirmed",(5,5,20,20),pts)
    calls=[(np.array([[[11.,10.]],[[21.,20.]]],np.float32),np.array([[1],[1]],np.uint8)),(np.array([[[10.2,10.]],[[25.,20.]]],np.float32),np.array([[1],[1]],np.uint8))]
    def fake(*args,**kwargs): return (*calls.pop(0),None)
    monkeypatch.setattr(cv2,"calcOpticalFlowPyrLK",fake)
    box,good,sufficient=tr._predict_track(track,current,40,40)
    assert sufficient and len(good)==1 and box[0]==6

def test_failed_lk_and_no_points_do_not_predict_motion(monkeypatch):
    tr=Tracker(config(min_points=2)); tr._previous_gray=np.zeros((40,40),np.uint8)
    from src.tracker import Track
    track=Track(1,"confirmed",(5,5,20,20),np.array([[[10.,10.]]],np.float32))
    monkeypatch.setattr(cv2,"calcOpticalFlowPyrLK",lambda *a,**k:(np.array([[[30.,30.]]],np.float32),np.array([[0]],np.uint8),None))
    box,points,sufficient=tr._predict_track(track,np.zeros((40,40),np.uint8),40,40)
    assert box==(5,5,20,20) and points.shape==(0,1,2) and not sufficient
    tr2=Tracker(config()); first=tr2.update(np.zeros((40,40),np.uint8),[(2,2,15,15)])[0]
    assert first.points.shape==(0,1,2)

def test_greedy_non_square_matching_and_invalid_boxes():
    tr=Tracker(config(matcher="greedy",iou_gate=.2))
    assert tr._assign([(0,0,10,10),(30,0,10,10)],[(0,0,10,10)])==[(0,0)]
    with pytest.raises(ValueError): tr.update(np.zeros((20,20),np.uint8),[(1,2,3)])
    with pytest.raises(ValueError): tr.update(np.zeros((20,20),np.uint8),[(float("nan"),2,3,4)])

def test_border_box_is_clipped_and_fully_outside_detection_ignored():
    tr=Tracker(config()); frame=np.zeros((30,30),np.uint8)
    tracks=tr.update(frame,[(-4,-3,12,12),(40,40,4,4)])
    assert len(tracks)==1 and tracks[0].box==(0,0,8,9)

def test_confirmed_track_already_outside_frame_is_retired():
    tr=Tracker(config(confirm_hits=1)); frame=np.zeros((40,40),np.uint8)
    track=tr.update(frame,[(5,5,20,20)])[0]
    track.box=(40,5,20,20)
    result=tr.update(frame,[])
    assert any(t.state=="lost" and t.id==track.id for t in result)

def test_hungarian_handles_non_square_assignment():
    tr=Tracker(config(matcher="hungarian",iou_gate=.1))
    pairs=tr._assign([(0,0,10,10),(30,0,10,10)],[(30,0,10,10)])
    assert pairs==[(1,0)]

def _edge_sequence(start, step, axis, frame_size=80, box_size=24):
    """Tạo frame và detection đồng nhất cho texture trượt tới một cạnh."""
    rng=np.random.default_rng(91); patch=rng.integers(20,245,(box_size,box_size),dtype=np.uint8)
    position=list(start); sequence=[]; expected=[]
    for _ in range(20):
        frame=np.zeros((frame_size,frame_size),np.uint8); x,y=position
        left=max(0,x); top=max(0,y); right=min(frame_size,x+box_size); bottom=min(frame_size,y+box_size)
        if right>left and bottom>top: frame[top:bottom,left:right]=patch[top-y:bottom-y,left-x:right-x]
        detection=(x,y,box_size,box_size) if right>left and bottom>top else None
        sequence.append((frame,[detection] if detection else [])); expected.append((x,y))
        if x+box_size<=0 or y+box_size<=0 or x>=frame_size or y>=frame_size: break
        position[axis]+=step
    return sequence,expected

def _run_to_exit(start,step,axis):
    """Chạy LK thật cho texture đang tịnh tiến và trả chuỗi kết quả."""
    tr=Tracker(config(min_points=3,max_misses=8,confirm_hits=1,fb_thresh=1.5))
    sequence,positions=_edge_sequence(start,step,axis); observed=[]
    for index,(frame,detections) in enumerate(sequence):
        tracks=tr.update(frame,detections)
        observed.append((positions[index],[(t.id,t.state,t.box,t.misses) for t in tracks]))
        if any(t.state=="lost" for t in tracks): break
    return observed

def test_real_lk_object_exits_right_edge():
    observed=_run_to_exit((12,28),8,0)
    assert observed[-2][0][0]+24>0 and not any(t[1]=="lost" for t in observed[-2][1])
    lost=[t for t in observed[-1][1] if t[1]=="lost"]
    assert lost and lost[0][0]==1 and lost[0][3]==1  # kết thúc ngay, không đợi 8 misses

def test_real_lk_object_exits_left_edge():
    observed=_run_to_exit((44,28),-8,0)
    assert observed[-2][0][0]+24>0 and not any(t[1]=="lost" for t in observed[-2][1])
    lost=[t for t in observed[-1][1] if t[1]=="lost"]
    assert lost and lost[0][0]==1 and lost[0][3]==1  # kết thúc ngay, không đợi 8 misses

def test_real_lk_object_exits_top_and_bottom_edges():
    for start,step,axis in [((28,44),-8,1),((28,12),8,1)]:
        observed=_run_to_exit(start,step,axis)
        previous=observed[-2][0][1]
        assert previous+24>0 and not any(t[1]=="lost" for t in observed[-2][1])
        lost=[t for t in observed[-1][1] if t[1]=="lost"]
        assert lost and lost[0][0]==1 and lost[0][3]==1  # kết thúc ngay, không đợi 8 misses

def test_real_lk_partial_box_at_edge_is_not_lost_early():
    observed=_run_to_exit((12,28),8,0)
    partial=[entry for entry in observed if entry[0][0]<80 and entry[0][0]+24>80]
    assert partial
    assert all(not any(track[1]=="lost" for track in tracks) for _,tracks in partial)

def test_midframe_lk_point_loss_keeps_box_until_existing_timeout(monkeypatch):
    tr=Tracker(config(min_points=3,max_misses=2,confirm_hits=1))
    first=textured_frame([(45,35,24,24)]); moved=textured_frame([(47,35,24,24)])
    initial=tr.update(first,[(45,35,24,24)])[0]
    moved_track=tr.update(moved,[(47,35,24,24)])[0]
    assert moved_track.last_displacement is not None and abs(moved_track.last_displacement[0])>0
    points=moved_track.points
    monkeypatch.setattr(cv2,"calcOpticalFlowPyrLK",lambda *a,**k:(points.copy(),np.zeros((len(points),1),np.uint8),None))
    first_miss=tr.update(moved,[])[0]; second_miss=tr.update(moved,[])[0]
    assert first_miss.box==second_miss.box==moved_track.box
    assert first_miss.state==second_miss.state=="confirmed" and second_miss.misses==2
    final=tr.update(moved,[])
    assert any(track.id==initial.id and track.state=="lost" for track in final)
