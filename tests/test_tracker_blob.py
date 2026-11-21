"""Kiểm thử blob và phục hồi appearance trên frame BGR tổng hợp."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from src.config import load_config
from src.metrics import default_iou_threshold
from src.tracker import Tracker, _box_overlap_metrics
from src.utils import iou

BOXES=[(10,20,20,24),(30,20,20,24)]
COLORS=[(0,0,230),(0,220,0)]
def cfg(**overrides):
    section={"winSize":[21,21],"maxLevel":2,"criteria":[3,30,0.01],"use_fb_check":True,"fb_thresh":1.0,"min_points":3,"max_corners":80,"quality_level":0.01,"min_distance":3.0,"block_size":5,"refresh_every":3,"matcher":"hungarian","iou_gate":0.05,"confirm_hits":1,"max_misses":4,"box_update_alpha":0.0,"blob_iou":0.2,"blob_track_coverage":0.75,"blob_center_separation":0.35,"blob_grace_frames":35,"hist_bins":[16,16],"hist_corr_min":0.7}
    section.update(overrides); return {"tracker":section}

def paint(boxes=BOXES,colors=COLORS,background=(20,20,20)):
    """Tạo BGR frame có texture nhẹ trong từng hộp màu."""
    frame=np.full((80,80,3),background,np.uint8)
    for (x,y,w,h),color in zip(boxes,colors):
        yy,xx=np.indices((h,w)); texture=((xx//3+yy//3)%2)*8
        patch=np.empty((h,w,3),np.uint8)
        for channel,value in enumerate(color): patch[:,:,channel]=np.clip(value+texture,0,255)
        frame[y:y+h,x:x+w]=patch
    return frame

def set_static_predictions(tracker):
    """Giữ box dự đoán cố định để cô lập logic ghép blob/appearance."""
    tracker._predict_track=lambda track,gray,width,height:(track.box,track.points,False)

def seeded_tracker(**options):
    tracker=Tracker(cfg(**options)); frame=paint(); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    tracks=tracker.update(gray,BOXES,frame_bgr=frame); set_static_predictions(tracker)
    return tracker,tracks

def test_detect_merged_blob_marks_tracks_and_never_updates_or_spawns():
    tracker,initial=seeded_tracker(); original=[t.box for t in initial]; ids=[t.id for t in initial]; hits=[t.hits for t in initial]
    merged_box=(10,20,40,24); bgr=paint([merged_box],[(80,80,80)])
    result=tracker.update(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),[merged_box],frame_bgr=bgr)
    assert len(tracker._tracks)==2 and {t.id for t in result}==set(ids)
    assert all(t.occluded for t in result)
    assert [t.box for t in tracker._tracks]==original and [t.hits for t in tracker._tracks]==hits
    assert all(t.misses==0 for t in tracker._tracks)

def test_single_track_detection_is_not_classified_as_merged():
    tracker,initial=seeded_tracker(); bgr=paint()
    result=tracker.update(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),[BOXES[0]],frame_bgr=bgr)
    first=next(t for t in result if t.id==initial[0].id); second=next(t for t in result if t.id==initial[1].id)
    assert not first.occluded and first.hits==2 and first.misses==0
    assert not second.occluded and second.misses==1
    assert len(tracker._tracks)==2

def test_histogram_only_updates_on_clean_observation():
    tracker,tracks=seeded_tracker(); before=[t.appearance_hist.copy() for t in tracks]
    clean_frame=paint(colors=[(230,0,0),(0,220,220)])
    tracker.update(cv2.cvtColor(clean_frame,cv2.COLOR_BGR2GRAY),BOXES,frame_bgr=clean_frame)
    clean_hist=[t.appearance_hist.copy() for t in tracker._tracks]
    assert any(not np.array_equal(current,old) for current,old in zip(clean_hist,before))
    merged=(10,20,40,24); blob_frame=paint([merged],[(100,100,100)])
    tracker.update(cv2.cvtColor(blob_frame,cv2.COLOR_BGR2GRAY),[merged],frame_bgr=blob_frame)
    assert all(np.array_equal(t.appearance_hist,h) for t,h in zip(tracker._tracks,clean_hist))
    hidden_frame=paint([],[],background=(100,100,100))
    tracker.update(cv2.cvtColor(hidden_frame,cv2.COLOR_BGR2GRAY),[],frame_bgr=hidden_frame)
    assert all(np.array_equal(t.appearance_hist,h) for t,h in zip(tracker._tracks,clean_hist))

def test_histogram_comes_from_real_bgr_channels():
    tracker=Tracker(cfg()); frame=paint(); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    track=tracker.update(gray,[BOXES[0]],frame_bgr=frame)[0]
    actual=tracker._appearance_histogram(frame,BOXES[0])
    fake_gray_bgr=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    fabricated=tracker._appearance_histogram(fake_gray_bgr,BOXES[0])
    assert actual is not None and fabricated is not None
    assert np.array_equal(track.appearance_hist,actual)
    assert not np.array_equal(actual,fabricated)

def test_split_blob_recovers_original_ids_with_histogram_and_iou():
    tracker,initial=seeded_tracker(); expected={t.id:box for t,box in zip(initial,BOXES)}
    blob=(10,20,40,24); merged_frame=paint([blob],[(120,120,120)])
    tracker.update(cv2.cvtColor(merged_frame,cv2.COLOR_BGR2GRAY),[blob],frame_bgr=merged_frame)
    split_frame=paint(colors=COLORS)
    result=tracker.update(cv2.cvtColor(split_frame,cv2.COLOR_BGR2GRAY),BOXES,frame_bgr=split_frame)
    assert len(tracker._tracks)==2
    assert {t.id:t.box for t in result}==expected
    assert all(not t.occluded and t.misses==0 for t in result)

def test_histogram_below_threshold_does_not_recover_by_iou_alone():
    tracker,initial=seeded_tracker(hist_corr_min=0.95)
    blob=(10,20,40,24); merged_frame=paint([blob],[(120,120,120)])
    tracker.update(cv2.cvtColor(merged_frame,cv2.COLOR_BGR2GRAY),[blob],frame_bgr=merged_frame)
    wrong_colors=[(255,0,0),(255,255,0)]; wrong_frame=paint(colors=wrong_colors)
    result=tracker.update(cv2.cvtColor(wrong_frame,cv2.COLOR_BGR2GRAY),BOXES,frame_bgr=wrong_frame)
    old=[t for t in tracker._tracks if t.id in {v.id for v in initial}]
    new=[t for t in result if t.id not in {v.id for v in initial}]
    assert len(old)==2 and all(t.occluded and t.misses==1 for t in old)
    assert len(new)==2 and all(t.id>max(v.id for v in initial) for t in new)

def test_recovery_assignment_is_one_to_one():
    tracker,tracks=seeded_tracker(); tracks[0].occluded=True; tracks[1].occluded=True
    predicted=[tracks[0].box,tracks[1].box]; detections=[(10,20,40,24)]
    frame=paint([detections[0]],[COLORS[0]])
    pairs=tracker._recover_assignments([0,1],[0],predicted,detections,frame)
    assert len(pairs)==1 and pairs[0][1]==0

def test_update_requires_real_bgr_shape_and_preserves_two_argument_api():
    tracker=Tracker(cfg()); frame=paint(); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    track=tracker.update(gray,[BOXES[0]])[0]
    assert track.appearance_hist is None
    with pytest.raises(ValueError,match="cùng kích thước"):
        tracker.update(gray,[BOXES[0]],frame_bgr=frame[:40])
    other=Tracker(cfg()); got=other.update(gray,[BOXES[0]],frame_bgr=frame)[0]
    assert got.appearance_hist is not None and got.appearance_hist.sum()==pytest.approx(1.0)
    assert other.hist_bins==(16,16)

AUDIT_BOXES={1:(342,138,38,90),2:(276,219,40,91),3:(322,220,39,90)}

def audit_frame(boxes=AUDIT_BOXES,colors=None):
    """Sinh BGR có texture theo đúng bố cục box được audit."""
    frame=np.full((360,640,3),(18,18,18),np.uint8)
    palette=colors or {1:(20,60,220),2:(20,220,60),3:(220,60,20)}
    for track_id,(x,y,w,h) in boxes.items():
        yy,xx=np.indices((h,w)); texture=((xx//4+yy//4+track_id)%2)*10
        patch=np.empty((h,w,3),np.uint8)
        for channel,value in enumerate(palette[track_id]):
            patch[:,:,channel]=np.clip(value+texture,0,255)
        frame[y:y+h,x:x+w]=patch
    return frame

def audit_tracker(**options):
    """Khởi tạo track thật rồi điều khiển prediction cho bài hình học xác định."""
    tracker=Tracker(cfg(**options)); frame=audit_frame(); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    tracks=tracker.update(gray,list(AUDIT_BOXES.values()),frame_bgr=frame)
    predictions={track_id:box for track_id,box in AUDIT_BOXES.items()}
    tracker._predict_track=lambda track,current_gray,width,height:(predictions[track.id],track.points,True)
    return tracker,tracks,predictions

def test_audit_frame_136_separates_clean_and_merged_detections():
    tracker,_,predictions=audit_tracker()
    separate=(276,219,40,91); merged=(324,138,61,172)
    classifications=tracker._classify_detections(list(predictions.values()),[separate,merged])
    assert [value.kind for value in classifications]==["clean","merged"]
    assert classifications[0].track_indices==frozenset({1})
    assert classifications[1].track_indices==frozenset({0,2})

def test_audit_frame_137_large_detection_is_merged_by_coverage():
    tracker,_,predictions=audit_tracker()
    large=(278,138,102,172)
    classifications=tracker._classify_detections(list(predictions.values()),[large])
    assert classifications[0].kind=="merged"
    assert classifications[0].track_indices==frozenset({0,1,2})

def test_single_track_detection_is_clean_and_near_duplicate_tracks_ambiguous():
    tracker,_,predictions=audit_tracker()
    one=tracker._classify_detections(list(predictions.values()),[(276,219,40,91)])
    assert one[0].kind=="clean"
    duplicate=tracker._classify_detections([(10,10,20,20),(11,10,20,20)],[(10,10,21,20)])
    assert duplicate[0].kind=="ambiguous"

def test_coverage_is_safe_for_zero_area_and_non_overlapping_boxes():
    assert _box_overlap_metrics((0,0,0,10),(0,0,10,10))==(0.0,0.0,0.0)
    assert _box_overlap_metrics((100,100,10,10),(0,0,10,10))==(0.0,0.0,0.0)

def test_large_stale_prediction_with_center_outside_does_not_merge_clean_detection():
    tracker=Tracker(cfg())
    detections=[(246,220,40,90)]
    predictions=[(247,220,39,90),(203,138,78,172)]
    assert _box_overlap_metrics(detections[0],predictions[1])[0] > tracker.blob_iou
    assert tracker._classify_detections(predictions,detections)[0].kind=="clean"

def test_ambiguous_detection_cannot_recover_update_histogram_or_create_id():
    tracker,tracks=seeded_tracker(); ids={track.id for track in tracks}
    for track in tracks: track.occluded=True
    tracker._predict_track=lambda track,gray,width,height:(BOXES[0],track.points,False)
    frame=paint(); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    before={track.id:track.appearance_hist.copy() for track in tracks}
    detections=[BOXES[0]]
    assert tracker._classify_detections([BOXES[0],BOXES[0]],detections)[0].kind=="ambiguous"
    result=tracker.update(gray,detections,frame_bgr=frame)
    assert {track.id for track in result}==ids and len(tracker._tracks)==2
    assert all(track.occluded for track in tracker._tracks)
    assert all(np.array_equal(track.appearance_hist,before[track.id]) for track in tracker._tracks)

def test_ambiguous_detection_is_excluded_from_normal_association():
    """Detection ambiguous không được tăng hit cho bất kỳ track nào."""
    boxes=[(10,20,20,24),(20,20,20,24)]
    initial_frame=paint(boxes,[(0,0,230),(0,220,0)])
    gray=cv2.cvtColor(initial_frame,cv2.COLOR_BGR2GRAY)
    tracker=Tracker(cfg(confirm_hits=1,box_update_alpha=1.0))
    initial=tracker.update(gray,boxes,frame_bgr=initial_frame)
    before={track.id:(track.box,track.hits,track.appearance_hist.copy()) for track in initial}

    ambiguous_box=(10,20,30,24)
    ambiguous_frame=initial_frame.copy()
    ambiguous_gray=cv2.cvtColor(ambiguous_frame,cv2.COLOR_BGR2GRAY)
    result=tracker.update(ambiguous_gray,[ambiguous_box],frame_bgr=ambiguous_frame)

    assert len(tracker._tracks)==2
    assert {track.id for track in result}==set(before)
    assert all(track.hits==before[track.id][1] for track in tracker._tracks)
    assert all(track.misses==1 for track in tracker._tracks)
    assert all(track.box==before[track.id][0] for track in tracker._tracks)
    assert all(np.array_equal(track.appearance_hist,before[track.id][2]) for track in tracker._tracks)

def test_ambiguous_detection_cannot_steal_assignment_from_unrelated_track():
    """Detection ambiguous không được chiếm candidate của track thứ ba."""
    boxes=[(10,10,100,100),(11,10,100,100),(60,60,20,20)]
    frame=np.full((140,140,3),(20,20,20),np.uint8)
    for box,color in zip(boxes,[(0,0,220),(0,220,0),(220,0,0)]):
        x,y,w,h=box
        yy,xx=np.indices((h,w)); texture=((xx//4+yy//4)%2)*10
        patch=np.empty((h,w,3),np.uint8)
        for channel,value in enumerate(color): patch[:,:,channel]=np.clip(value+texture,0,255)
        frame[y:y+h,x:x+w]=patch
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    tracker=Tracker(cfg(confirm_hits=1,box_update_alpha=1.0))
    initial=tracker.update(gray,boxes,frame_bgr=frame)
    before={track.id:(track.hits,track.box) for track in initial}

    result=tracker.update(gray.copy(),[boxes[2]],frame_bgr=frame.copy())

    assert len(tracker._tracks)==3
    assert {track.id for track in result}==set(before)
    assert all(track.hits==before[track.id][0] for track in tracker._tracks)
    assert all(track.misses==1 for track in tracker._tracks)
    assert all(track.box==before[track.id][1] for track in tracker._tracks)

def test_ambiguous_tracks_do_not_replenish_points():
    """Track nghi liên quan ambiguous không được vào nhánh refresh unmatched."""
    boxes=[(10,20,20,24),(20,20,20,24)]
    frame=paint(boxes,[(0,0,230),(0,220,0)])
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    tracker=Tracker(cfg(confirm_hits=1,box_update_alpha=1.0))
    tracks=tracker.update(gray,boxes,frame_bgr=frame)
    for track in tracks: track.points=np.empty((0,1,2),np.float32)
    calls=[]
    original=tracker._refresh_track_points
    def spy(current,track,branch):
        calls.append((track.id,branch))
        return original(current,track,branch)
    tracker._refresh_track_points=spy

    tracker.update(gray.copy(),[(10,20,30,24)],frame_bgr=frame.copy())

    assert calls==[]

def test_recovered_prior_occluded_track_does_not_refresh_implicitly():
    """Recovery dùng nhánh matched chung nhưng không tự cấp quyền replenish."""
    tracker,tracks=seeded_tracker()
    merged=(10,20,40,24)
    blob=paint([merged],[(120,120,120)])
    tracker.update(cv2.cvtColor(blob,cv2.COLOR_BGR2GRAY),[merged],frame_bgr=blob)
    calls=[]
    original=tracker._refresh_track_points
    def spy(current,track,branch):
        calls.append((track.id,branch))
        return original(current,track,branch)
    tracker._refresh_track_points=spy
    split=paint()

    result=tracker.update(cv2.cvtColor(split,cv2.COLOR_BGR2GRAY),BOXES,frame_bgr=split)

    assert {track.id for track in result}=={track.id for track in tracks}
    assert calls==[]

def test_tentative_track_does_not_receive_blob_grace_or_recovery(caplog):
    """Tentative bị che đi theo normal miss; confirmed neighbor vẫn phục hồi."""
    import json
    import logging
    tracker=Tracker(cfg(confirm_hits=3,max_misses=4))
    frame=paint(); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    tentative,confirmed=tracker.update(gray,BOXES,frame_bgr=frame)
    tentative.occluded=True; tentative.blob_grace_age=4
    confirmed.state="confirmed"; confirmed.occluded=True
    set_static_predictions(tracker)
    with caplog.at_level(logging.INFO,logger="src.tracker"):
        result=tracker.update(gray,[BOXES[1]],frame_bgr=frame)
    events=[json.loads(record.getMessage()) for record in caplog.records if record.name=="src.tracker"]
    recovery=next(event for event in events if event["event"]=="recovery_matrix")
    tentative_after=next(track for track in result if track.id==tentative.id)
    confirmed_after=next(track for track in result if track.id==confirmed.id)
    assert recovery["track_ids"]==[confirmed.id]
    assert tentative_after.state=="tentative" and tentative_after.misses==1
    assert tentative_after.blob_grace_age==0 and not tentative_after.occluded
    assert confirmed_after.state=="confirmed" and not confirmed_after.occluded

def test_tentative_merged_this_frame_gets_no_grace_hits_or_refresh():
    """Tentative gặp merged ngay frame này miss; confirmed track vẫn có grace."""
    tracker=Tracker(cfg(confirm_hits=3,max_misses=4))
    frame=paint(); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    tentative,confirmed=tracker.update(gray,BOXES,frame_bgr=frame)
    tentative.consecutive_hits=2; confirmed.state="confirmed"
    set_static_predictions(tracker)
    calls=[]
    original=tracker._refresh_track_points
    def spy(current,track,branch):
        calls.append((track.id,branch))
        return original(current,track,branch)
    tracker._refresh_track_points=spy
    merged=(10,20,40,24); blob=paint([merged],[(120,120,120)])
    result=tracker.update(cv2.cvtColor(blob,cv2.COLOR_BGR2GRAY),[merged],frame_bgr=blob)
    current=next(track for track in result if track.id==tentative.id)
    protected=next(track for track in result if track.id==confirmed.id)
    assert current.state=="tentative" and current.hits==1 and current.consecutive_hits==0
    assert current.misses==1 and current.blob_grace_age==0 and not current.occluded
    assert protected.state=="confirmed" and protected.occluded
    assert calls==[]

def test_replenished_points_do_not_belong_to_neighboring_person():
    """Điểm được bổ sung trong box mở rộng không được thuộc người lân cận."""
    boxes=[(10,20,30,40),(40,20,20,40)]
    initial_frame=np.full((80,80,3),(20,20,20),np.uint8)
    tracker=Tracker(cfg(confirm_hits=1,box_update_alpha=0.0))
    initial_gray=cv2.cvtColor(initial_frame,cv2.COLOR_BGR2GRAY)
    tracker.update(initial_gray,boxes,frame_bgr=initial_frame)

    current=np.full_like(initial_frame,(20,20,20))
    rng=np.random.default_rng(2026)
    x,y,w,h=boxes[0]
    person_one_rng=np.random.default_rng(2025)
    person_one_texture=person_one_rng.integers(30,230,(h,w),dtype=np.uint8)
    current[y:y+h,x:x+w]=person_one_texture[:,:,None]
    x,y,w,h=boxes[1]
    texture=rng.integers(0,256,(h,w),dtype=np.uint8)
    current[y:y+h,x:x+w]=texture[:,:,None]
    # Detection của người 1 hơi rộng, chạm vào vùng texture nhìn thấy của người 2.
    detection=(10,20,39,40)
    current_gray=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
    tracks=tracker.update(current_gray,[detection],frame_bgr=current)

    target=next(track for track in tracks if track.id==1)
    owner_mask=np.zeros(current_gray.shape,np.int8)
    for owner,box in enumerate(boxes,start=1):
        x,y,w,h=box; owner_mask[y:y+h,x:x+w]=owner
    point_owners=[]
    for px,py in target.points.reshape(-1,2):
        ix,iy=int(px),int(py)
        point_owners.append(int(owner_mask[iy,ix]) if 0<=iy<80 and 0<=ix<80 else 0)

    owner_counts={
        "person_1":point_owners.count(1),
        "person_2":point_owners.count(2),
        "background":point_owners.count(0),
        "unknown":sum(not 0<=int(px)<80 or not 0<=int(py)<80 for px,py in target.points.reshape(-1,2)),
    }
    print(f"T1-E owner counts: {owner_counts}")

    assert len(target.points)>=tracker.min_points
    assert point_owners, "refresh phải tạo feature points để kiểm tra ownership"
    assert 2 not in point_owners, f"track 1 nhận nhầm feature points của người 2: {point_owners}"

def test_audit_merged_detection_never_updates_box_histogram_or_id():
    tracker,initial,predictions=audit_tracker(); initial_ids={track.id for track in initial}
    before_boxes={track.id:track.box for track in initial}
    before_hist={track.id:track.appearance_hist.copy() for track in initial}
    bgr=audit_frame(); gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    result=tracker.update(gray,[(278,138,102,172)],frame_bgr=bgr)
    assert len(tracker._tracks)==3 and {track.id for track in result}==initial_ids
    assert {track.id:track.box for track in tracker._tracks}==before_boxes
    assert all(track.occluded for track in tracker._tracks)
    assert all(np.array_equal(track.appearance_hist,before_hist[track.id]) for track in tracker._tracks)

def test_frame_163_blob_is_never_used_for_recovery_even_with_high_hist_correlation():
    predictions={1:(300,138,38,90),2:(233,154,90,156),3:(277,220,39,90)}
    tracker,tracks,_=audit_tracker()
    hist_before={track.id:track.appearance_hist.copy() for track in tracks}
    frame137=audit_frame(); gray137=cv2.cvtColor(frame137,cv2.COLOR_BGR2GRAY)
    tracker.update(gray137,[(278,138,102,172)],frame_bgr=frame137)
    assert all(track.occluded for track in tracks)
    assert all(np.array_equal(track.appearance_hist,hist_before[track.id]) for track in tracks)
    tracker._predict_track=lambda track,gray,width,height:(predictions[track.id],track.points,True)
    merged=(256,138,106,172); bgr=audit_frame(predictions); gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    contaminated=tracker._appearance_histogram(bgr,merged)
    assert contaminated is not None
    # Mô phỏng trạng thái cũ đã lưu crop blob; gate phải chặn trước recovery.
    tracks[1].appearance_hist=contaminated.copy()
    before={track.id:(track.box,track.appearance_hist.copy()) for track in tracks}
    calls=[]; original_recovery=tracker._recover_assignments
    def spy(track_indices,det_indices,predicted_boxes,detections,frame_bgr):
        calls.append((list(track_indices),list(det_indices)))
        return original_recovery(track_indices,det_indices,predicted_boxes,detections,frame_bgr)
    tracker._recover_assignments=spy
    result=tracker.update(gray,[merged],frame_bgr=bgr)
    assert calls==[([],[])]
    assert len(tracker._tracks)==3 and {track.id for track in result}=={1,2,3}
    assert all(track.occluded for track in tracker._tracks)
    assert all(track.box==predictions[track.id] for track in tracker._tracks)
    assert all(np.array_equal(track.appearance_hist,before[track.id][1]) for track in tracker._tracks)

def test_appearance_match_without_geometry_is_rejected():
    tracker,tracks=seeded_tracker(); track=tracks[0]; track.occluded=True
    bgr=paint(); far_prediction=(55,50,20,24)
    assert cv2.compareHist(track.appearance_hist,tracker._appearance_histogram(bgr,BOXES[0]),cv2.HISTCMP_CORREL)>=tracker.hist_corr_min
    assert tracker._recover_assignments([0],[0],[far_prediction],[BOXES[0]],bgr)==[]

def test_appearance_only_candidate_does_not_spawn_duplicate_during_blob_grace():
    tracker=Tracker(cfg()); frame=paint(); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    track=tracker.update(gray,[BOXES[0]],frame_bgr=frame)[0]
    track.occluded=True; far_prediction=(55,50,20,24)
    tracker._predict_track=lambda active,current_gray,width,height:(far_prediction,active.points,False)
    result=tracker.update(gray,[BOXES[0]],frame_bgr=frame)
    assert len(tracker._tracks)==1 and len(result)==1 and result[0].id==track.id
    assert result[0].occluded and result[0].blob_grace_age==1
    assert result[0].box==far_prediction

def test_blob_grace_is_bounded_and_only_applies_to_confirmed_blob_tracks():
    boxes=[(10,20,20,24),(30,20,20,24),(65,20,10,24)]
    frame=paint(boxes,[(0,0,230),(0,220,0),(220,0,0)]); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    tracker=Tracker(cfg(max_misses=1,blob_grace_frames=3)); tracks=tracker.update(gray,boxes,frame_bgr=frame)
    tracker._predict_track=lambda track,current_gray,width,height:(track.box,track.points,False)
    merged=(10,20,40,24); merge_frame=paint([merged,boxes[2]],[(100,100,100),(220,0,0)])
    tracker.update(cv2.cvtColor(merge_frame,cv2.COLOR_BGR2GRAY),[merged,boxes[2]],frame_bgr=merge_frame)
    blob_ids={tracks[0].id,tracks[1].id}; independent_id=tracks[2].id
    first=tracker.update(gray,[]); assert all(track.state!="lost" for track in first if track.id in blob_ids)
    second=tracker.update(gray,[])
    assert all(track.state!="lost" for track in second if track.id in blob_ids)
    assert any(track.id==independent_id and track.state=="lost" for track in second)
    third=tracker.update(gray,[]); assert all(track.id in blob_ids for track in tracker._tracks)
    expired=tracker.update(gray,[])
    assert all(any(track.id==track_id and track.state=="lost" for track in expired) for track_id in blob_ids)
    assert not tracker._tracks

def test_blob_track_can_reacquire_after_waiting_within_grace():
    tracker,initial=seeded_tracker(max_misses=1,blob_grace_frames=35)
    merged=(10,20,40,24); blob=paint([merged],[(120,120,120)])
    tracker.update(cv2.cvtColor(blob,cv2.COLOR_BGR2GRAY),[merged],frame_bgr=blob)
    blank=paint([],[],background=(20,20,20)); gray=cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
    for _ in range(28): tracker.update(gray,[])
    assert all(track.blob_grace_age==28 for track in tracker._tracks)
    split=paint(colors=COLORS)
    recovered=tracker.update(cv2.cvtColor(split,cv2.COLOR_BGR2GRAY),BOXES,frame_bgr=split)
    assert {track.id for track in recovered}=={track.id for track in initial}
    assert all(not track.occluded and track.misses==0 for track in recovered)

def test_real_lk_synthetic_sequence_keeps_ids_through_merge_and_split():
    """Chạy ảnh BGR chuyển động thật, ghép detection đã biết rồi tách lại."""
    rng=np.random.default_rng(45); patches=[]
    for color in ((0,0,230),(0,220,0)):
        texture=rng.integers(0,35,(40,30,1),dtype=np.uint8)
        checker=((np.indices((40,30)).sum(axis=0)//3)%2*45).astype(np.uint8)
        patch=np.empty((40,30,3),dtype=np.uint8); patch[:]=color
        patch=np.clip(patch.astype(np.int16)+checker[:,:,None]+texture.astype(np.int16),0,255).astype(np.uint8)
        patches.append(patch)
    tracker=Tracker(cfg(confirm_hits=1,max_misses=3,blob_grace_frames=20,fb_thresh=1.2))
    saved_ids:set[int]=set(); history={}
    for frame_index in range(56):
        if frame_index<=20:
            x_a,x_b=20+2*frame_index,130-2*frame_index
        elif frame_index<=26:
            x_a,x_b=60,90
        else:
            x_a,x_b=60-2*(frame_index-26),90+2*(frame_index-26)
        boxes=[(x_a,45,30,40),(x_b,45,30,40)]
        frame=np.full((130,180,3),30,dtype=np.uint8)
        frame[45:85,x_a:x_a+30]=patches[0]
        frame[45:85,x_b:x_b+30]=patches[1]
        detections=[(min(x_a,x_b),45,abs(x_a-x_b)+30,40)] if 21<=frame_index<=26 else boxes
        tracks=tracker.update(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),detections,frame_bgr=frame)
        if frame_index==0:
            saved_ids={track.id for track in tracks}
        history[frame_index]=[(track.id,track.box,track.occluded,track.state) for track in tracks]
    assert len(saved_ids)==2
    assert {track_id for track_id,_,_,_ in history[26]}==saved_ids
    assert all(occluded for _,_,occluded,_ in history[26])
    assert {track_id for track_id,_,_,_ in history[27]}==saved_ids
    assert {track_id for track_id,_,_,_ in history[55]}==saved_ids
    final_by_id={track_id:box for track_id,box,_,_ in history[55]}
    assert final_by_id[min(saved_ids)][0]<final_by_id[max(saved_ids)][0]
    assert all(not occluded for _,_,occluded,_ in history[55])


# Các scenario phía dưới dùng ảnh và detection sinh trong bộ nhớ, không chạy Detector/MOG2.
CONTROL_WIDTH = 220
CONTROL_HEIGHT = 150
CONTROL_BOX_SIZE = (30, 48)
CONTROL_Y = 48


def controlled_cfg():
    """Đọc nguyên ngưỡng tracking hiện có, không nới cấu hình cho test."""
    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")


def stable_textures() -> dict[int, np.ndarray]:
    """Tạo một texture cố định cho mỗi ID trong toàn bộ sequence."""
    width, height = CONTROL_BOX_SIZE
    colors = {1: (25, 55, 220), 2: (35, 220, 55), 3: (220, 55, 35)}
    textures: dict[int, np.ndarray] = {}
    for object_id, color in colors.items():
        rng = np.random.default_rng(7100 + object_id)
        noise = rng.integers(-42, 43, size=(height, width, 1), dtype=np.int16)
        yy, xx = np.indices((height, width))
        checker = (((xx // 3 + yy // 4) % 2) * 18).astype(np.int16)[..., None]
        base = np.asarray(color, dtype=np.int16)[None, None, :]
        textures[object_id] = np.clip(base + noise + checker, 0, 255).astype(np.uint8)
    return textures


_CONTROL_TEXTURES = stable_textures()


def controlled_frame(positions: dict[int, tuple[int, int]]) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    """Vẽ rectangle có texture ổn định tại tọa độ đã chỉ định."""
    frame = np.full((CONTROL_HEIGHT, CONTROL_WIDTH, 3), (18, 18, 18), dtype=np.uint8)
    width, height = CONTROL_BOX_SIZE
    boxes = {object_id: (x, CONTROL_Y, width, height) for object_id, x in positions.items()}
    for object_id in sorted(boxes):
        x, y, box_width, box_height = boxes[object_id]
        frame[y:y + box_height, x:x + box_width] = _CONTROL_TEXTURES[object_id]
    return frame, boxes


def progressively_occluded_frame(
    positions: dict[int, tuple[int, int]],
    covered_fraction: float,
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    """Che dần mỗi texture bằng nền để LK mất dần các điểm ảnh bám."""
    frame, boxes = controlled_frame(positions)
    for x, y, width, height in boxes.values():
        covered_width = int(round(width * covered_fraction))
        if covered_width:
            frame[y:y + height, x + width - covered_width:x + width] = (18, 18, 18)
    return frame, boxes


def controlled_detections(boxes: dict[int, tuple[int, int, int, int]], merged: bool) -> list[tuple[int, int, int, int]]:
    """Trả box GT riêng lẻ hoặc một box union được kiểm soát cho merge."""
    if not merged:
        return list(boxes.values())
    x0 = min(box[0] for box in boxes.values())
    y0 = min(box[1] for box in boxes.values())
    x1 = max(box[0] + box[2] for box in boxes.values())
    y1 = max(box[1] + box[3] for box in boxes.values())
    return [(x0, y0, x1 - x0, y1 - y0)]


def controlled_step(
    tracker: Tracker,
    positions: dict[int, tuple[int, int]],
    merged: bool,
) -> dict[str, object]:
    """Nạp một frame/detection biết trước và lưu trạng thái quan sát được."""
    previously_occluded = {track.id for track in tracker._tracks if track.occluded}
    frame, boxes = controlled_frame(positions)
    detections = controlled_detections(boxes, merged)
    tracks = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detections, frame_bgr=frame)
    return {
        "gt": boxes,
        "detections": detections,
        "merged": merged,
        "recovered": sorted(
            track.id for track in tracks
            if track.id in previously_occluded and not track.occluded and track.misses == 0
        ),
        "tracks": [
            (track.id, track.state, track.box, len(track.points), track.occluded, track.misses, track.blob_grace_age)
            for track in tracks
        ],
    }


def controlled_matches(
    ground_truth: dict[int, tuple[int, int, int, int]],
    tracks: list[tuple[int, str, tuple[int, int, int, int], int, bool, int, int]],
) -> dict[int, tuple[int, float]]:
    """Ghép GT/track bằng Hungarian như Metrics, chỉ dùng confirmed và IoU config."""
    confirmed = [track for track in tracks if track[1] == "confirmed"]
    gt_items = list(ground_truth.items())
    if not gt_items or not confirmed:
        return {}
    overlaps = np.asarray([[iou(gt_box, track[2]) for track in confirmed] for _, gt_box in gt_items])
    rows, cols = linear_sum_assignment(1.0 - overlaps)
    threshold = default_iou_threshold()
    return {
        gt_items[row][0]: (confirmed[col][0], float(overlaps[row, col]))
        for row, col in zip(rows, cols)
        if overlaps[row, col] >= threshold
    }


def sequence_id_metrics(records: list[dict[str, object]]) -> tuple[int, int]:
    """Đếm ID switch và fragmentation trên các sequence in-memory."""
    previous: dict[int, int] = {}
    states: dict[int, list[bool]] = {}
    switches = 0
    for record in records:
        matches = controlled_matches(record["gt"], record["tracks"])
        matched_ids = set(matches)
        for object_id, (track_id, _) in matches.items():
            if object_id in previous and previous[object_id] != track_id:
                switches += 1
            previous[object_id] = track_id
        for object_id in record["gt"]:
            states.setdefault(object_id, []).append(object_id in matched_ids)
    fragments = 0
    for visible_states in states.values():
        was_matched = False
        had_gap = False
        for visible in visible_states:
            if visible:
                fragments += int(was_matched and had_gap)
                was_matched, had_gap = True, False
            elif was_matched:
                had_gap = True
    return switches, fragments


def summarize_record(record: dict[str, object]) -> str:
    """Tạo thông báo lỗi có frame, detection và trạng thái track."""
    return f"GT={record['gt']}, detections={record['detections']}, tracks={record['tracks']}"


def test_controlled_normal_motion_keeps_identity_and_lk_points():
    """Hai đối tượng độc lập đi qua ảnh với texture cố định và detection oracle."""
    tracker = Tracker(controlled_cfg())
    records = [
        controlled_step(tracker, {1: 18 + 2 * frame, 2: 164 - 2 * frame}, merged=False)
        for frame in range(32)
    ]
    initial = controlled_matches(records[2]["gt"], records[2]["tracks"])
    final = controlled_matches(records[-1]["gt"], records[-1]["tracks"])
    assert set(initial) == {1, 2} and set(final) == {1, 2}, summarize_record(records[-1])
    assert {object_id: track_id for object_id, (track_id, _) in initial.items()} == {
        object_id: track_id for object_id, (track_id, _) in final.items()
    }, summarize_record(records[-1])
    assert all(points >= 4 for _, state, _, points, _, _, _ in records[-1]["tracks"] if state == "confirmed")
    assert len({track_id for track_id, *_ in records[-1]["tracks"]}) == 2
    assert sequence_id_metrics(records) == (0, 0)


def test_controlled_merge_split_preserves_old_ids_without_blob_updates():
    """Hai đối tượng ghép thành một detection rồi tách về vị trí riêng."""
    tracker = Tracker(controlled_cfg())
    records: list[dict[str, object]] = []
    initial_ids: dict[int, int] = {}
    for frame_index in range(37):
        if frame_index <= 10:
            positions = {1: 20 + 5 * frame_index, 2: 160 - 6 * frame_index}
            merged = False
        elif frame_index <= 18:
            positions = {1: 70, 2: 100}
            merged = True
        else:
            distance = frame_index - 18
            positions = {1: max(20, 70 - 5 * distance), 2: min(160, 100 + 6 * distance)}
            merged = False
        record = controlled_step(tracker, positions, merged)
        records.append(record)
        if frame_index == 2:
            initial_ids = {object_id: pair[0] for object_id, pair in controlled_matches(record["gt"], record["tracks"]).items()}
        if merged:
            assert len(record["detections"]) == 1
            assert record["merged"] is True
            assert {row[0] for row in record["tracks"]} == set(initial_ids.values()), summarize_record(record)
            assert all(row[4] for row in record["tracks"]), summarize_record(record)
    final_matches = controlled_matches(records[-1]["gt"], records[-1]["tracks"])
    assert {object_id: pair[0] for object_id, pair in final_matches.items()} == initial_ids, summarize_record(records[-1])
    assert set(records[19]["recovered"]) == set(initial_ids.values()), summarize_record(records[19])
    assert len({row[0] for row in records[-1]["tracks"]}) == 2
    assert sequence_id_metrics(records) == (0, 0)


def test_controlled_three_track_simultaneous_recovery_is_one_to_one():
    """Ba track cùng merge, sau đó ba detection sạch xuất hiện trong một frame."""
    tracker = Tracker(controlled_cfg())
    positions = {1: 18, 2: 86, 3: 154}
    records: list[dict[str, object]] = []
    initial_ids: dict[int, int] = {}
    for frame_index in range(12):
        merged = 3 <= frame_index <= 7
        record = controlled_step(tracker, positions, merged)
        records.append(record)
        if frame_index == 2:
            initial_ids = {object_id: pair[0] for object_id, pair in controlled_matches(record["gt"], record["tracks"]).items()}
        if merged:
            assert len(record["detections"]) == 1
            assert record["merged"] is True
            assert {row[0] for row in record["tracks"]} == set(initial_ids.values()), summarize_record(record)
        if frame_index == 8:
            recovered = {object_id: pair[0] for object_id, pair in controlled_matches(record["gt"], record["tracks"]).items()}
            assert recovered == initial_ids, summarize_record(record)
            assert set(record["recovered"]) == set(initial_ids.values()), summarize_record(record)
            assert len({track_id for track_id, *_ in record["tracks"]}) == 3
    assert sequence_id_metrics(records) == (0, 0)


def test_same_color_distractor_cannot_recover_occluded_track_by_histogram_alone():
    """Detection HSV gần giống nhưng xa box prediction không được nhận làm track cũ."""
    tracker = Tracker(controlled_cfg())
    records = [controlled_step(tracker, {1: 20, 2: 75}, merged=False) for _ in range(3)]
    old_ids = {row[0] for row in records[-1]["tracks"]}
    records.append(controlled_step(tracker, {1: 20, 2: 75}, merged=True))
    # Dùng cùng texture/HSV của ID 1 ở vị trí xa; hai vật cũ khuất khỏi ảnh.
    distractor_frame, distractor_box = controlled_frame({1: 170})
    distractor_gray = cv2.cvtColor(distractor_frame, cv2.COLOR_BGR2GRAY)
    result = tracker.update(distractor_gray, list(distractor_box.values()), frame_bgr=distractor_frame)
    records.append({
        "gt": {1: (20, CONTROL_Y, *CONTROL_BOX_SIZE), 2: (75, CONTROL_Y, *CONTROL_BOX_SIZE)},
        "detections": list(distractor_box.values()),
        "merged": False,
        "recovered": [],
        "tracks": [
            (track.id, track.state, track.box, len(track.points), track.occluded, track.misses, track.blob_grace_age)
            for track in result
        ],
    })
    assert {track.id for track in tracker._tracks} == old_ids
    assert all(track.occluded and track.blob_grace_age == 1 for track in tracker._tracks)
    assert all(track.box[0] < 120 for track in tracker._tracks)
    assert {track.id for track in result} == old_ids
    assert sequence_id_metrics(records) == (0, 0)


def test_lk_point_loss_records_frozen_box_and_bounded_grace_expiry():
    """Khi texture biến mất dưới merged detection, ghi nhận box hold và ID reinit."""
    tracker = Tracker(controlled_cfg())
    positions = {1: 20, 2: 150}
    records: list[dict[str, object]] = []
    for frame_index in range(3):
        records.append(controlled_step(tracker, positions, merged=False))
    original = {track.id: track.box for track in tracker._tracks}
    merged_box = (20, CONTROL_Y, 160, CONTROL_BOX_SIZE[1])
    point_history: dict[int, list[int]] = {track.id: [len(track.points)] for track in tracker._tracks}
    box_history: dict[int, list[tuple[int, int, int, int]]] = {track.id: [track.box] for track in tracker._tracks}
    cover_fractions = (0.15, 0.30, 0.45, 0.60, 0.80, 1.0)
    for cover_fraction in cover_fractions:
        covered_frame, gt_boxes = progressively_occluded_frame(positions, cover_fraction)
        gray = cv2.cvtColor(covered_frame, cv2.COLOR_BGR2GRAY)
        tracks = tracker.update(gray, [merged_box], frame_bgr=covered_frame)
        for track in tracker._tracks:
            point_history[track.id].append(len(track.points))
            box_history[track.id].append(track.box)
        records.append({
            "gt": gt_boxes,
            "detections": [merged_box],
            "merged": True,
            "recovered": [],
            "tracks": [(t.id, t.state, t.box, len(t.points), t.occluded, t.misses, t.blob_grace_age) for t in tracks],
        })
    lost_points = [row[3] for row in records[-1]["tracks"]]
    assert all(points < controlled_cfg().tracker["min_points"] for points in lost_points), summarize_record(records[-1])
    assert all(values[-1] < values[0] and len(set(values)) > 1 for values in point_history.values())
    min_points = controlled_cfg().tracker["min_points"]
    frozen_boxes: dict[int, tuple[int, int, int, int]] = {}
    for track_id, counts in point_history.items():
        below_min = next(index for index, count in enumerate(counts) if count < min_points)
        frozen_boxes[track_id] = box_history[track_id][below_min]
        assert all(box == frozen_boxes[track_id] for box in box_history[track_id][below_min:])
    assert all(row[2] == frozen_boxes[row[0]] for row in records[-1]["tracks"]), summarize_record(records[-1])
    assert all(row[4] and row[6] == 0 for row in records[-1]["tracks"]), summarize_record(records[-1])

    # Sau blob, cùng HSV nhưng dịch xa: prediction cũ không thể phục hồi bằng IoU.
    moved = {1: 70, 2: 110}
    split_frame, split_boxes = controlled_frame(moved)
    split_gray = cv2.cvtColor(split_frame, cv2.COLOR_BGR2GRAY)
    records.append({"gt": split_boxes, "detections": list(split_boxes.values()), "merged": False, "recovered": [], "tracks": []})
    split_tracks = tracker.update(split_gray, list(split_boxes.values()), frame_bgr=split_frame)
    records[-1]["tracks"] = [(t.id, t.state, t.box, len(t.points), t.occluded, t.misses, t.blob_grace_age) for t in split_tracks]
    assert all(row[2] == frozen_boxes[row[0]] for row in records[-1]["tracks"] if row[0] in original), summarize_record(records[-1])
    assert all(row[4] and row[6] == 1 for row in records[-1]["tracks"] if row[0] in original), summarize_record(records[-1])

    # Grace hiện tại được cấu hình 35 frame; không kéo dài để cứu track cũ.
    lost_ids: set[int] = set()
    lost_frame: int | None = None
    blank = np.full((CONTROL_HEIGHT, CONTROL_WIDTH, 3), (18, 18, 18), dtype=np.uint8)
    gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    for offset in range(controlled_cfg().tracker["blob_grace_frames"] + 2):
        expired = tracker.update(gray, [], frame_bgr=blank)
        records.append({
            "gt": split_boxes,
            "detections": [],
            "merged": False,
            "recovered": [],
            "tracks": [
                (t.id, t.state, t.box, len(t.points), t.occluded, t.misses, t.blob_grace_age)
                for t in expired
            ],
        })
        newly_lost = {track.id for track in expired if track.state == "lost"}
        if newly_lost and lost_frame is None:
            lost_frame = 10 + offset
        lost_ids.update(newly_lost)
    assert lost_ids == set(original)
    assert not tracker._tracks
    assert lost_frame == 44

    reappeared = controlled_step(tracker, moved, merged=False)
    records.append(reappeared)
    replacement_ids = {row[0] for row in reappeared["tracks"]}
    assert len(replacement_ids) == 2 and replacement_ids.isdisjoint(original)
    assert min(replacement_ids) > max(original)
    for _ in range(2):
        records.append(controlled_step(tracker, moved, merged=False))
    final_matches = controlled_matches(records[-1]["gt"], records[-1]["tracks"])
    assert {object_id: pair[0] for object_id, pair in final_matches.items()} == {
        1: min(replacement_ids), 2: max(replacement_ids),
    }
    assert sequence_id_metrics(records) == (2, 2)
