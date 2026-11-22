import pytest
from src.common.utils import iou, iou_matrix, scale_rect

def test_iou_edges():
    assert iou((0,0,2,2),(3,0,2,2))==0
    assert iou((0,0,2,2),(0,0,2,2))==1
    assert iou((0,0,0,2),(0,0,2,2))==0
    assert iou_matrix([],[]).shape==(0,0)

def test_scale_rect(): assert scale_rect((1,2,3,4),2,0.5)==(2,1,6,2)
