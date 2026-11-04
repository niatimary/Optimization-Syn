#!/usr/bin/env python
"""
Lightweight, fast checks for optuna_search.py's pure helper functions -- deliberately
NOT an end-to-end test of the Optuna study itself (that's slow and requires the real
train/validation .npz files plus a live ammonyte install). Run with:

    python -m unittest test_optuna_search.py -v

covers:
  - transition direction assignment (transitions_from_states)
  - empty detected-transition arrays (_matched_timing_errors, evaluate_detection_by_direction)
  - empty warming/cooling subsets (evaluate_detection_by_direction)
  - ambiguous detections (_classify_detected_directions, evaluate_detection_by_direction)
  - load_split's new validation (mismatched/invalid dataset shapes)
  - _best_full_params reconstruction for Dynp vs non-Dynp
  - _validate_args (CLI argument validation)
  - _fmt (safe formatting of possibly-missing/NaN trial attributes)
"""
import argparse
import json
import os
import tempfile
import unittest

import numpy as np

from optuna_search import (
    DEFAULT_TRAIN_PATH, DEFAULT_VAL_PATH, JUMP,
    _best_full_params, _classify_detected_directions, _current_search_config, _fmt,
    _matched_timing_errors, _validate_args, evaluate_detection_by_direction, load_split,
    transitions_from_states,
)


class FakeTrial:
    """Minimal stand-in for an optuna.trial.FrozenTrial -- _best_full_params only
    ever reads .params."""
    def __init__(self, params):
        self.params = params


def _default_args(**overrides):
    ns = argparse.Namespace(
        n_trials=300, timeout=None, patience=75, min_trials=150, min_delta=0.01,
        train_path=DEFAULT_TRAIN_PATH, val_path=DEFAULT_VAL_PATH, objective='warming',
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class TestTransitionsFromStates(unittest.TestCase):
    def test_direction_assignment(self):
        # stadial(1), stadial, interstadial(0), interstadial, stadial(1)
        states = np.array([1, 1, 0, 0, 1])
        time_arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        times, directions = transitions_from_states(time_arr, states)
        np.testing.assert_array_equal(times, [2.0, 4.0])
        np.testing.assert_array_equal(directions, ['warming', 'cooling'])

    def test_no_transitions(self):
        states = np.array([1, 1, 1])
        time_arr = np.array([0.0, 1.0, 2.0])
        times, directions = transitions_from_states(time_arr, states)
        self.assertEqual(len(times), 0)
        self.assertEqual(len(directions), 0)


class TestMatchedTimingErrors(unittest.TestCase):
    def test_empty_detected(self):
        errs = _matched_timing_errors(np.array([]), np.array([1.0, 2.0]), tolerance=0.5)
        self.assertEqual(len(errs), 0)

    def test_empty_ground_truth(self):
        errs = _matched_timing_errors(np.array([1.0]), np.array([]), tolerance=0.5)
        self.assertEqual(len(errs), 0)

    def test_one_to_one_match(self):
        errs = _matched_timing_errors(np.array([10.1]), np.array([10.0]), tolerance=0.5)
        np.testing.assert_allclose(errs, [0.1])


class TestClassifyDetectedDirections(unittest.TestCase):
    def setUp(self):
        # flat-low then flat-high, jump at index 50 -> a clean 'warming' detection
        self.time_arr = np.arange(100, dtype=float)
        self.value = np.concatenate([np.zeros(50), np.ones(50)])

    def test_warming_direction(self):
        directions = _classify_detected_directions(self.value, self.time_arr,
                                                     detected_times=[50.0], min_size=5)
        self.assertEqual(list(directions), ['warming'])

    def test_cooling_direction(self):
        value = np.concatenate([np.ones(50), np.zeros(50)])
        directions = _classify_detected_directions(value, self.time_arr,
                                                     detected_times=[50.0], min_size=5)
        self.assertEqual(list(directions), ['cooling'])

    def test_empty_detections(self):
        directions = _classify_detected_directions(self.value, self.time_arr,
                                                     detected_times=[], min_size=5)
        self.assertEqual(len(directions), 0)

    def test_duplicate_time_is_ambiguous(self):
        # two detections resolving to the identical index collapse each other's
        # before/after window to nothing -- see _classify_detected_directions docstring
        directions = _classify_detected_directions(self.value, self.time_arr,
                                                     detected_times=[50.0, 50.0], min_size=5)
        self.assertEqual(list(directions), ['ambiguous', 'ambiguous'])


class TestEvaluateDetectionByDirection(unittest.TestCase):
    def setUp(self):
        self.time_arr = np.arange(100, dtype=float)
        self.value = np.concatenate([np.zeros(50), np.ones(50)])

    def test_single_warming_match_has_empty_cooling_subset(self):
        metrics, timing_errors, n_ambiguous = evaluate_detection_by_direction(
            self.time_arr, self.value, detected=[50.0],
            gt_times=[50.0], gt_directions=['warming'],
            tolerance=0.5, min_size=5)
        self.assertEqual(n_ambiguous, 0)
        self.assertEqual(metrics['warming'].precision, 1.0)
        self.assertEqual(metrics['warming'].recall, 1.0)
        # cooling subset is empty on both sides -> ammonyte's own empty/empty edge case
        self.assertEqual(metrics['cooling'].precision, 0)
        self.assertEqual(metrics['cooling'].recall, 0)
        self.assertEqual(metrics['cooling'].n_detected, 0)
        self.assertEqual(metrics['cooling'].n_ground_truth, 0)

    def test_empty_detected_array(self):
        metrics, timing_errors, n_ambiguous = evaluate_detection_by_direction(
            self.time_arr, self.value, detected=[],
            gt_times=[50.0], gt_directions=['warming'],
            tolerance=0.5, min_size=5)
        self.assertEqual(n_ambiguous, 0)
        self.assertEqual(metrics['all'].n_detected, 0)
        self.assertEqual(metrics['all'].false_negatives, 1)

    def test_ambiguous_detections_excluded_from_directional_subsets(self):
        metrics, timing_errors, n_ambiguous = evaluate_detection_by_direction(
            self.time_arr, self.value, detected=[50.0, 50.0],
            gt_times=[50.0], gt_directions=['warming'],
            tolerance=0.5, min_size=5)
        self.assertEqual(n_ambiguous, 2)
        # ambiguous detections still count in f1_all (2 detected, 1 ground truth)
        self.assertEqual(metrics['all'].n_detected, 2)
        # but neither shows up in the warming- or cooling-filtered detected set
        self.assertEqual(metrics['warming'].n_detected, 0)
        self.assertEqual(metrics['cooling'].n_detected, 0)


class TestLoadSplitValidation(unittest.TestCase):
    def _write_npz(self, tmpdir, **arrays):
        path = os.path.join(tmpdir, "split.npz")
        np.savez(path, **arrays)
        return path

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_split("/nonexistent/path/does_not_exist.npz")

    def test_missing_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_npz(tmp, time=np.arange(10, dtype=float),
                                    db_noisy=np.zeros((2, 10)))
            with self.assertRaisesRegex(ValueError, "missing required array"):
                load_split(path)

    def test_mismatched_state_true_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_npz(
                tmp, time=np.arange(10, dtype=float),
                db_noisy=np.zeros((2, 10)),
                state_true=np.zeros((2, 9)),
            )
            with self.assertRaisesRegex(ValueError, "'state_true' shape"):
                load_split(path)

    def test_series_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_npz(
                tmp, time=np.arange(5, dtype=float),
                db_noisy=np.zeros((2, 10)),
                state_true=np.zeros((2, 10)),
            )
            with self.assertRaisesRegex(ValueError, "series length"):
                load_split(path)

    def test_non_increasing_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_npz(
                tmp, time=np.array([0.0, 1.0, 0.5, 2.0]),
                db_noisy=np.zeros((1, 4)),
                state_true=np.zeros((1, 4)),
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_split(path)

    def test_non_binary_state_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = np.zeros((1, 4))
            state[0, 2] = 2  # invalid label
            path = self._write_npz(
                tmp, time=np.arange(4, dtype=float),
                db_noisy=np.zeros((1, 4)),
                state_true=state,
            )
            with self.assertRaisesRegex(ValueError, "binary"):
                load_split(path)

    def test_valid_split_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = np.zeros((1, 4))
            state[0, 2:] = 1
            path = self._write_npz(
                tmp, time=np.arange(4, dtype=float),
                db_noisy=np.random.randn(1, 4),
                state_true=state,
            )
            time_arr, db_noisy, gt_times, gt_directions, n = load_split(path)
            self.assertEqual(n, 1)
            self.assertEqual(len(gt_times[0]), 1)


class TestBestFullParams(unittest.TestCase):
    def test_dynp_reconstruction(self):
        trial = FakeTrial(dict(method='Dynp', cost_Dynp='l1', min_size=20, n_bkps=45))
        params = _best_full_params(trial)
        self.assertEqual(params, dict(method='Dynp', cost='l1', pen=None, n_bkps=45,
                                       min_size=20, jump=JUMP))

    def test_non_dynp_reconstruction(self):
        trial = FakeTrial(dict(method='Pelt', cost_Pelt='l2', min_size=5, pen_Pelt_l2=3.2))
        params = _best_full_params(trial)
        self.assertEqual(params, dict(method='Pelt', cost='l2', pen=3.2, n_bkps=None,
                                       min_size=5, jump=JUMP))


class TestValidateArgs(unittest.TestCase):
    def test_valid_args_pass(self):
        _validate_args(_default_args())  # should not raise

    def test_n_trials_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "--n-trials"):
            _validate_args(_default_args(n_trials=0))

    def test_timeout_must_be_positive_when_supplied(self):
        with self.assertRaisesRegex(ValueError, "--timeout"):
            _validate_args(_default_args(timeout=0))

    def test_timeout_none_is_fine(self):
        _validate_args(_default_args(timeout=None))

    def test_patience_cannot_be_negative(self):
        with self.assertRaisesRegex(ValueError, "--patience"):
            _validate_args(_default_args(patience=-1))

    def test_min_trials_cannot_be_negative(self):
        with self.assertRaisesRegex(ValueError, "--min-trials"):
            _validate_args(_default_args(min_trials=-1))

    def test_min_delta_cannot_be_negative(self):
        with self.assertRaisesRegex(ValueError, "--min-delta"):
            _validate_args(_default_args(min_delta=-0.01))

    def test_train_path_must_exist(self):
        with self.assertRaisesRegex(ValueError, "--train-path"):
            _validate_args(_default_args(train_path="/nonexistent/train.npz"))

    def test_val_path_must_exist(self):
        with self.assertRaisesRegex(ValueError, "--val-path"):
            _validate_args(_default_args(val_path="/nonexistent/validation.npz"))


class TestCurrentSearchConfig(unittest.TestCase):
    def test_stable_across_calls(self):
        args = _default_args(objective='warming')
        self.assertEqual(_current_search_config(args), _current_search_config(args))

    def test_survives_json_round_trip(self):
        # Optuna's storage round-trips user_attrs through JSON -- tuples silently become
        # lists, so the freshly-built dict must already be all-lists or a resumed run
        # would spuriously look "different" from the one that created the study.
        args = _default_args(objective='warming')
        config = _current_search_config(args)
        round_tripped = json.loads(json.dumps(config))
        self.assertEqual(config, round_tripped)

    def test_differs_when_objective_changes(self):
        args_a = _default_args(objective='warming')
        args_b = _default_args(objective='all')
        self.assertNotEqual(_current_search_config(args_a), _current_search_config(args_b))

    def test_differs_when_train_path_changes(self):
        args_a = _default_args(train_path=DEFAULT_TRAIN_PATH)
        args_b = _default_args(train_path=DEFAULT_VAL_PATH)
        self.assertNotEqual(_current_search_config(args_a), _current_search_config(args_b))


class TestFmt(unittest.TestCase):
    def test_none_is_na(self):
        self.assertEqual(_fmt(None), "N/A")

    def test_nan_is_na(self):
        self.assertEqual(_fmt(float('nan')), "N/A")

    def test_finite_float_formats(self):
        self.assertEqual(_fmt(0.123456), "0.123")

    def test_int_formats(self):
        self.assertEqual(_fmt(1), "1.000")


if __name__ == "__main__":
    unittest.main()
