"""Kiểm thử metrics trên bộ CSV nhỏ, dựng riêng cho từng trường hợp."""
import csv
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import detection_pr, tracking_metrics, reid_metrics, rank1, confusion_matrix

def make_csv(path: Path, columns: list[str], rows: list[tuple]) -> Path:
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream); writer.writerow(columns); writer.writerows(rows)
    return path

def track_csv(path: Path, rows: list[tuple]) -> Path:
    return make_csv(path,["frame","track_id","state","x","y","w","h"],rows)
def gt_csv(path: Path, rows: list[tuple]) -> Path:
    return make_csv(path,["frame","gt_id","x","y","w","h"],rows)

def test_perfect_tracking_has_no_switches(tmp_path):
    gt=gt_csv(tmp_path/"gt.csv",[(f,1,0,0,10,10) for f in range(3)])
    tracks=track_csv(tmp_path/"tracks.csv",[(f,8,"confirmed",0,0,10,10) for f in range(3)])
    result=tracking_metrics(tracks,gt,0.5)
    assert result["id_switches"]==0 and result["fragmentations"]==0

def test_id_swap_mid_sequence_counts_one_switch(tmp_path):
    gt=gt_csv(tmp_path/"gt.csv",[(f,1,0,0,10,10) for f in range(3)])
    tracks=track_csv(tmp_path/"tracks.csv",[(0,10,"confirmed",0,0,10,10),(1,10,"confirmed",0,0,10,10),(2,20,"confirmed",0,0,10,10)])
    assert tracking_metrics(tracks,gt,0.5)["id_switches"]==1

def test_gap_then_reacquisition_is_one_fragmentation(tmp_path):
    gt=gt_csv(tmp_path/"gt.csv",[(f,1,0,0,10,10) for f in range(4)])
    tracks=track_csv(tmp_path/"tracks.csv",[(0,4,"confirmed",0,0,10,10),(1,4,"confirmed",0,0,10,10),(3,4,"confirmed",0,0,10,10)])
    assert tracking_metrics(tracks,gt,0.5)["fragmentations"]==1

def test_detection_extra_box_reduces_precision_and_non_square_assignment(tmp_path):
    det=make_csv(tmp_path/"det.csv",["frame","x","y","w","h","area"],[(0,0,0,10,10,100),(0,20,20,10,10,100),(1,99,99,2,2,4)])
    gt=gt_csv(tmp_path/"gt.csv",[(0,1,0,0,10,10)])
    result=detection_pr(det,gt,0.5)
    assert result=={"precision":0.5,"recall":1.0,"tp":1,"fp":1,"fn":0}

def test_threshold_boundary_and_empty_detections_gt_frames(tmp_path):
    det=make_csv(tmp_path/"det.csv",["frame","x","y","w","h","area"],[(0,5,0,10,10,100)])
    gt=gt_csv(tmp_path/"gt.csv",[(0,1,0,0,10,10),(1,2,0,0,10,10)])
    assert detection_pr(det,gt,1/3)["tp"]==1  # IoU bằng ngưỡng được nhận
    empty=make_csv(tmp_path/"empty_det.csv",["frame","x","y","w","h","area"],[])
    result=detection_pr(empty,gt,0.5)
    assert (result["tp"],result["fp"],result["fn"],result["precision"],result["recall"])==(0,0,2,0.0,0.0)

def test_empty_ground_truth_and_header_only_tracks(tmp_path):
    det=make_csv(tmp_path/"det.csv",["frame","x","y","w","h","area"],[(0,0,0,10,10,100)])
    gt=gt_csv(tmp_path/"gt.csv",[])
    assert detection_pr(det,gt,0.5)["tp"]==0
    tracks=track_csv(tmp_path/"tracks.csv",[])
    result=tracking_metrics(tracks,gt,0.5)
    assert result["n_gt_ids"]==result["n_track_ids"]==0

def test_reid_seven_of_nine_and_unmatched_is_wrong(tmp_path):
    gt=make_csv(tmp_path/"gt_reid.csv",["person","cam_a_id","cam_b_id"],[(f"p{i}",i,i+10) for i in range(9)])
    rows=[]
    for i in range(9):
        rows.extend([("cam_a",i,100+i),("cam_b",i+10,100+i if i<7 else -1)])
    ids=make_csv(tmp_path/"ids.csv",["cam","track_id","global_id"],rows)
    assert reid_metrics(ids,gt)["accuracy_str"]=="7/9"
    cm=confusion_matrix(ids,gt)
    assert cm["p8"]["-1"]==1

def test_rank1_uses_best_candidate_and_method_filter(tmp_path):
    gt=make_csv(tmp_path/"gt_reid.csv",["person","cam_a_id","cam_b_id"],[("p1",1,11),("p2",2,12)])
    scores=make_csv(tmp_path/"scores.csv",["cam_a_track","cam_b_track","method","score"],[(1,11,"ORB",0.9),(1,12,"ORB",0.1),(2,11,"ORB",0.8),(2,12,"SIFT",0.95)])
    result=rank1(scores,gt,"ORB")
    assert result["accuracy_str"]=="1/2"

def test_empty_file_and_missing_required_column_raise(tmp_path):
    empty=tmp_path/"empty.csv"; empty.write_text("")
    with pytest.raises(ValueError,match="thiếu header"):
        detection_pr(empty,gt_csv(tmp_path/"gt.csv",[]),0.5)
    bad=make_csv(tmp_path/"bad.csv",["frame","x"],[(0,1)])
    with pytest.raises(ValueError,match="thiếu cột"):
        detection_pr(bad,gt_csv(tmp_path/"gt2.csv",[]),0.5)

def test_count_accuracy_counts_frames_with_zero_detections(tmp_path):
    det=make_csv(tmp_path/"det.csv",["frame","x","y","w","h","area"],[(0,0,0,10,10,100)])
    manual=make_csv(tmp_path/"manual.csv",["frame_start","frame_end","count"],[(0,1,1),(2,2,0)])
    from src.metrics import count_accuracy
    result=count_accuracy(det,manual)
    assert result=={"accuracy":2/3,"correct_frames":2,"total_frames":3}


def test_detection_pr_scores_only_hand_labelled_frames_without_heavy_occlusion(tmp_path):
    """XLA 3.3: chỉ chấm keyframe gán tay; bỏ frame nội suy và frame có box che nặng."""
    columns = ["frame", "gt_id", "x", "y", "w", "h", "occluded", "keyframe"]
    gt = make_csv(tmp_path / "gt.csv", columns, [
        (0, 1, 0, 0, 10, 10, 0, 1),
        (1, 1, 0, 0, 10, 10, 0, 0),
        (2, 1, 0, 0, 10, 10, 2, 1),
    ])
    det = make_csv(tmp_path / "det.csv", ["frame", "x", "y", "w", "h", "area"], [
        (0, 0, 0, 10, 10, 100),
        (1, 50, 50, 10, 10, 100),
        (2, 50, 50, 10, 10, 100),
    ])
    result = detection_pr(det, gt, 0.5)
    assert result == {"precision": 1.0, "recall": 1.0, "tp": 1, "fp": 0, "fn": 0}
