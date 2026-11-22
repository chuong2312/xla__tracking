"""Kiểm thử Kalman vận tốc hằng (src/common/kalman.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.kalman import ConstantVelocityKalman

CFG = {"acceleration_std_px": 0.5, "detection_noise_px": 4.0, "lk_noise_px": 3.0, "initial_velocity_std_px": 5.0}


def _truth(t):
    return (100.0 + 3.0 * t, 50.0 + 1.0 * t)


def _converged_filter(frames=30):
    kalman = ConstantVelocityKalman(*_truth(0), CFG)
    for t in range(1, frames + 1):
        kalman.predict()
        kalman.correct(*_truth(t))
    return kalman


def test_velocity_converges_on_straight_line():
    vx, vy = _converged_filter().velocity
    assert vx == pytest.approx(3.0, abs=0.2)
    assert vy == pytest.approx(1.0, abs=0.2)


def test_prediction_keeps_moving_while_person_is_hidden():
    kalman = _converged_filter(30)
    for t in range(31, 41):
        predicted = kalman.predict()
    error = np.hypot(predicted[0] - _truth(40)[0], predicted[1] - _truth(40)[1])
    assert error < 3.0


def test_filter_smooths_noisy_measurements():
    rng = np.random.default_rng(0)
    kalman = ConstantVelocityKalman(*_truth(0), CFG)
    raw_errors, filtered_errors = [], []
    for t in range(1, 101):
        truth = np.array(_truth(t))
        measured = truth + rng.normal(0.0, 4.0, 2)
        kalman.predict()
        estimate = np.array(kalman.correct(*measured))
        if t > 10:
            raw_errors.append(np.sum((measured - truth) ** 2))
            filtered_errors.append(np.sum((estimate - truth) ** 2))
    assert np.sqrt(np.mean(filtered_errors)) < np.sqrt(np.mean(raw_errors))


def test_less_noisy_source_pulls_estimate_closer():
    by_lk = ConstantVelocityKalman(0.0, 0.0, CFG)
    by_detection = ConstantVelocityKalman(0.0, 0.0, CFG)
    by_lk.predict()
    by_detection.predict()
    lk_x, _ = by_lk.correct(10.0, 0.0, source="lk")
    det_x, _ = by_detection.correct(10.0, 0.0, source="detection")
    assert lk_x > det_x


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        ConstantVelocityKalman(0.0, 0.0, CFG).correct(1.0, 1.0, source="mau")
