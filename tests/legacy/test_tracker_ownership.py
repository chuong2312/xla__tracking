"""Kiểm thử ownership LK và replenishment bằng correspondence tổng hợp."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pytest

from legacy.tracker import Track, Tracker


def _cfg(**changes):
    values={
        "winSize":[21,21],"maxLevel":2,"criteria":[3,30,0.01],
        "use_fb_check":True,"fb_thresh":1.0,"min_points":3,
        "max_corners":80,"quality_level":0.01,"min_distance":3.0,
        "block_size":5,"refresh_every":3,"matcher":"hungarian",
        "iou_gate":0.05,"confirm_hits":2,"max_misses":3,
        "box_update_alpha":0.0,"blob_iou":0.2,"blob_track_coverage":0.75,
        "blob_center_separation":0.35,"blob_grace_frames":35,
        "hist_bins":[16,16],"hist_corr_min":0.7,
    }
    values.update(changes)
    return {"tracker":values}


def _safe_candidate_count(tracker: Tracker, frame: np.ndarray, own: tuple[int,int,int,int], neighbors: list[tuple[int,int,int,int]]) -> int:
    """Đếm độc lập góc trong own box trừ union các neighbor boxes."""
    height,width=frame.shape
    x,y,w,h=own
    mask=np.zeros((height,width),np.uint8)
    mask[max(0,y):min(height,y+h),max(0,x):min(width,x+w)]=255
    for nx,ny,nw,nh in neighbors:
        mask[max(0,ny):min(height,ny+nh),max(0,nx):min(width,nx+nw)]=0
    points=cv2.goodFeaturesToTrack(
        frame,maxCorners=tracker.max_corners,qualityLevel=tracker.quality_level,
        minDistance=tracker.min_distance,mask=mask,blockSize=tracker.block_size,
    )
    return 0 if points is None else len(points)


def _predict(monkeypatch,box,points,neighbors=(),displacements=None,min_points=3):
    tracker=Tracker(_cfg(min_points=min_points))
    previous=np.zeros((80,80),np.uint8)
    current=np.zeros_like(previous)
    tracker._previous_gray=previous
    track=Track(1,"confirmed",box,np.asarray(points,dtype=np.float32).reshape(-1,1,2))
    tracker._ownership_snapshot=((1,"confirmed",box),*neighbors)
    source_cache={"old":None}
    shifts=displacements or {}

    def optical_flow(previous_image,current_image,values,*args,**kwargs):
        xy=values.reshape(-1,2).astype(np.float32)
        if previous_image is previous:
            source_cache["old"]=xy.copy()
            offsets=np.asarray([shifts.get(int(round(x)),(0.0,0.0)) for x,_ in xy],np.float32)
            result=xy+offsets
        else:
            result=source_cache["old"].copy()
        return result.reshape(-1,1,2),np.ones((len(xy),1),np.uint8),None

    monkeypatch.setattr(cv2,"calcOpticalFlowPyrLK",optical_flow)
    predicted,kept,sufficient=tracker._predict_track(track,current,80,80)
    return tracker,track,predicted,kept,sufficient


def test_under_min_safe_candidates_skip_refresh_and_preserve_only_valid_old_points():
    """Candidate thiếu không thay điểm cũ hợp lệ và không nạp candidate unsafe."""
    tracker=Tracker(_cfg())
    own=(10,10,20,20)
    neighbor=(12,10,18,20)
    tracker._predicted_snapshot=(
        (1,"confirmed",own),(2,"tentative",neighbor),
    )
    frame=np.zeros((40,40),np.uint8)
    rng=np.random.default_rng(2026)
    frame[10:30,10:30]=rng.integers(0,255,(20,20),dtype=np.uint8)
    safe_count=_safe_candidate_count(tracker,frame,own,[neighbor])
    assert safe_count<tracker.min_points

    safe_old=np.asarray([[[10.5,14.0]],[[11.0,18.0]]],np.float32)
    track=Track(1,"confirmed",own,safe_old.copy())
    tracker._refresh_track_points(frame,track,"normal_matched")
    assert np.array_equal(track.points,safe_old)

    already_filtered=Track(1,"confirmed",own,np.empty((0,1,2),np.float32))
    tracker._refresh_track_points(frame,already_filtered,"normal_matched")
    assert already_filtered.points.shape==(0,1,2)


def test_source_points_outside_own_box_are_rejected_before_median(monkeypatch):
    points=[(12,12),(18,16),(25,20),(35,20)]
    offsets={12:(2,0),18:(2,0),25:(2,0),35:(20,0)}
    _,_,predicted,kept,sufficient=_predict(
        monkeypatch,(10,10,20,20),points,displacements=offsets,min_points=3,
    )
    assert sufficient and predicted==(12,10,20,20)
    assert len(kept)==3


def test_neighbor_points_are_removed_before_median_and_prevent_drift(monkeypatch):
    points=[(12,12),(18,16),(25,20),(32,12),(35,16),(38,20)]
    neighbors=((2,"confirmed",(30,10,20,25)),)
    offsets={12:(2,0),18:(2,0),25:(2,0),32:(24,0),35:(24,0),38:(24,0)}
    _,_,predicted,kept,sufficient=_predict(
        monkeypatch,(10,10,20,20),points,neighbors,offsets,min_points=3,
    )
    assert sufficient and predicted==(12,10,20,20)
    assert len(kept)==3
    assert np.median([2,2,2,24,24,24])==13.0


@pytest.mark.parametrize("state",["tentative","confirmed"])
def test_active_neighbor_state_filters_shared_ownership_region(monkeypatch,state):
    points=[(22,15),(25,15),(28,15),(12,12)]
    _,_,_,kept,sufficient=_predict(
        monkeypatch,(10,10,20,20),points,((2,state,(20,10,20,20)),),min_points=1,
    )
    assert sufficient and np.array_equal(kept.reshape(-1,2),[[12,12]])


def test_lost_neighbor_is_ignored(monkeypatch):
    points=[(22,15),(25,15),(28,15)]
    _,_,_,kept,sufficient=_predict(
        monkeypatch,(10,10,20,20),points,((2,"lost",(20,10,20,20)),),min_points=3,
    )
    assert sufficient and len(kept)==3


def test_closed_box_edges_are_own_inclusive_and_neighbor_exclusive(monkeypatch):
    points=[(30,15),(12,12),(18,18)]
    _,_,_,kept,sufficient=_predict(
        monkeypatch,(10,10,20,20),points,displacements={30:(1,0),12:(1,0),18:(1,0)},min_points=3,
    )
    assert sufficient and len(kept)==3
    _,_,_,kept,sufficient=_predict(
        monkeypatch,(10,10,20,20),points,
        ((2,"confirmed",(30,10,20,20)),),
        {30:(1,0),12:(1,0),18:(1,0)},min_points=2,
    )
    assert sufficient and len(kept)==2
    assert not np.any(np.all(kept.reshape(-1,2)==[31,15],axis=1))


def test_insufficient_owned_points_use_existing_fallback_and_keep_only_safe_flow(monkeypatch):
    points=[(12,12),(18,16),(32,12),(35,16)]
    _,track,predicted,kept,sufficient=_predict(
        monkeypatch,(10,10,20,20),points,
        ((2,"confirmed",(30,10,20,20)),),
        {12:(2,0),18:(2,0),32:(20,0),35:(20,0)},min_points=3,
    )
    assert not sufficient and predicted==(10,10,20,20)
    assert track.last_displacement is None
    assert np.array_equal(kept.reshape(-1,2),[[14,12],[20,16]])


def test_replenishment_mask_subtracts_union_of_multiple_neighbors():
    tracker=Tracker(_cfg(min_points=3))
    own=(10,10,35,35)
    first_neighbor=(30,10,20,35)
    second_neighbor=(10,30,35,20)
    tracker._predicted_snapshot=(
        (1,"confirmed",own),(2,"tentative",first_neighbor),
        (3,"confirmed",second_neighbor),
    )
    frame=np.zeros((60,60),np.uint8)
    rng=np.random.default_rng(44)
    frame[10:45,10:45]=rng.integers(0,255,(35,35),dtype=np.uint8)
    track=Track(1,"confirmed",own,np.empty((0,1,2),np.float32))
    tracker._refresh_track_points(frame,track,"normal_matched")
    assert len(track.points)>=tracker.min_points
    for px,py in track.points.reshape(-1,2):
        assert not (30<=px<50 and 10<=py<45)
        assert not (10<=px<45 and 30<=py<50)


def test_no_neighbor_preserves_normal_replenishment_and_clips_border_boxes():
    tracker=Tracker(_cfg(min_points=3))
    tracker._predicted_snapshot=((1,"confirmed",(-4,-4,24,24)),)
    frame=np.random.default_rng(77).integers(0,255,(30,30),dtype=np.uint8)
    track=Track(1,"confirmed",(-4,-4,24,24),np.empty((0,1,2),np.float32))
    tracker._refresh_track_points(frame,track,"normal_matched")
    assert len(track.points)>=tracker.min_points
    assert np.isfinite(track.points).all()
    assert np.all(track.points[:,:,0]>=0) and np.all(track.points[:,:,1]>=0)
    assert np.all(track.points[:,:,0]<30) and np.all(track.points[:,:,1]<30)


def test_prediction_results_do_not_depend_on_track_iteration_order(monkeypatch):
    """Immutable frame-start snapshot yields identical results under reversed order."""
    snapshot=((1,"confirmed",(10,10,20,20)),(2,"tentative",(25,10,20,20)))
    sources={1:[(12,12),(18,16),(22,20)],2:[(26,12),(32,16),(38,20)]}
    shifts={12:(2,0),18:(2,0),22:(2,0),26:(-2,0),32:(-2,0),38:(-2,0)}

    def run(order):
        results={}
        for track_id in order:
            tracker=Tracker(_cfg(min_points=1))
            previous=np.zeros((80,80),np.uint8); current=np.zeros_like(previous)
            tracker._previous_gray=previous; tracker._ownership_snapshot=snapshot
            track=Track(track_id,snapshot[track_id-1][1],snapshot[track_id-1][2],np.asarray(sources[track_id],np.float32).reshape(-1,1,2))
            cache={"old":None}
            def flow(previous_image,current_image,values,*args,**kwargs):
                xy=values.reshape(-1,2).astype(np.float32)
                if previous_image is previous:
                    cache["old"]=xy.copy()
                    offset=np.asarray([shifts[int(round(x))] for x,_ in xy],np.float32)
                    moved=xy+offset
                else:
                    moved=cache["old"].copy()
                return moved.reshape(-1,1,2),np.ones((len(xy),1),np.uint8),None
            monkeypatch.setattr(cv2,"calcOpticalFlowPyrLK",flow)
            box,points,sufficient=tracker._predict_track(track,current,80,80)
            results[track_id]=(box,points.copy(),sufficient)
        return results

    forward=run([1,2]); reverse=run([2,1])
    for track_id in forward:
        assert forward[track_id][0]==reverse[track_id][0]
        assert np.array_equal(forward[track_id][1],reverse[track_id][1])
        assert forward[track_id][2]==reverse[track_id][2]


def test_prediction_diagnostics_count_lk_ownership_filtering(caplog,monkeypatch):
    """Trace phân biệt điểm ngoài own box và điểm thuộc neighbor trước median."""
    import json
    import logging
    points=[(12,12),(18,16),(35,20),(60,60)]
    neighbors=((2,"confirmed",(30,10,20,25)),)
    with caplog.at_level(logging.INFO,logger="legacy.tracker"):
        tracker,_,_,_,_=_predict(
            monkeypatch,(10,10,20,20),points,neighbors,
            {12:(2,0),18:(2,0),35:(20,0),60:(1,0)},min_points=2,
        )
    events=[json.loads(record.getMessage()) for record in caplog.records if record.name=="legacy.tracker"]
    prediction=next(event for event in events if event["event"]=="prediction")
    assert prediction["points_after_fb_check"]==4
    assert prediction["points_rejected_neighbor"]==1
    assert prediction["points_rejected_own_box"]==1
    assert prediction["points_after_ownership"]==2
    assert prediction["points_used_for_median"]==2
