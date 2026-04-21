"""
pytest test suite for Hand Gesture Control System
Run: pytest tests/test_suite.py -v
"""

import pytest
import numpy as np
import torch
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import VolumeController, extract_keypoints, check_pinch
from src.model import GestureLSTM
from src.mouse_controller import AirMouse
from src.config import GESTURES, SEQ_LENGTH, INPUT_SIZE, THRESHOLD


# ---------------------------------------------------------------------------
# Helpers — fake landmark-s замість MediaPipe
# ---------------------------------------------------------------------------

def make_landmark(x: float, y: float, z: float = 0.0) -> MagicMock:
    lm = MagicMock()
    lm.x, lm.y, lm.z = x, y, z
    return lm


def make_hand_landmarks(positions: dict) -> MagicMock:
    """positions: {index: (x, y, z)}. Відсутні заповнюються (0.5, 0.5, 0)."""
    hand = MagicMock()
    lms = []
    for i in range(21):
        if i in positions:
            x, y, z = positions[i] if len(positions[i]) == 3 else (*positions[i], 0.0)
            lms.append(make_landmark(x, y, z))
        else:
            lms.append(make_landmark(0.5, 0.5, 0.0))
    hand.landmark = lms
    return hand


def make_results(hands_data: list) -> MagicMock:
    """hands_data: [{'label': 'Left'|'Right', 'landmarks': {idx: (x,y,z)}}]"""
    results = MagicMock()
    if not hands_data:
        results.multi_hand_landmarks = None
        results.multi_handedness = None
        return results

    results.multi_hand_landmarks = []
    results.multi_handedness = []

    for hand in hands_data:
        results.multi_hand_landmarks.append(make_hand_landmarks(hand['landmarks']))
        handedness = MagicMock()
        handedness.classification = [MagicMock()]
        handedness.classification[0].label = hand['label']
        results.multi_handedness.append(handedness)

    return results


# ===========================================================================
# 1. VolumeController
# ===========================================================================

class TestVolumeController:

    def setup_method(self):
        self.vc = VolumeController(smoothing=0.15)

    # --- calculate_level ---

    def test_no_hand_returns_last_vol(self):
        self.vc._last_vol = 42
        assert self.vc.calculate_level(None) == 42

    def test_snap_to_max_when_fingers_far(self):
        hand = make_hand_landmarks({4: (0.0, 0.5), 8: (0.25, 0.5)})
        result = self.vc.calculate_level(hand)
        assert result == 100

    def test_snap_to_zero_when_fingers_close(self):
        hand = make_hand_landmarks({4: (0.5, 0.5), 8: (0.505, 0.5)})
        result = self.vc.calculate_level(hand)
        assert result == 0

    def test_dead_zone_no_change_on_micro_movement(self):
        self.vc._last_vol = 50
        hand = make_hand_landmarks({4: (0.4, 0.5), 8: (0.5, 0.5)})
        result = self.vc.calculate_level(hand)
        assert result == self.vc._last_vol

    def test_smoothing_applied(self):
        self.vc._last_vol = 0
        hand = make_hand_landmarks({4: (0.3, 0.5), 8: (0.5, 0.5)})
        result = self.vc.calculate_level(hand)
        assert 0 < result < 100

    def test_state_persists_across_calls(self):
        hand = make_hand_landmarks({4: (0.3, 0.5), 8: (0.5, 0.5)})
        first = self.vc.calculate_level(hand)
        second = self.vc.calculate_level(hand)
        assert isinstance(second, int)


# ===========================================================================
# 2. extract_keypoints
# ===========================================================================

class TestExtractKeypoints:

    def test_no_hands_returns_zeros(self):
        results = make_results([])
        kp = extract_keypoints(results)
        assert kp.shape == (INPUT_SIZE,)
        assert np.all(kp == 0)

    def test_right_hand_fills_second_half(self):
        landmarks = {i: (float(i) * 0.01, float(i) * 0.01, 0.0) for i in range(21)}
        results = make_results([{'label': 'Right', 'landmarks': landmarks}])
        kp = extract_keypoints(results)
        assert kp.shape == (INPUT_SIZE,)
        assert np.all(kp[:63] == 0)       
        assert not np.all(kp[63:] == 0)   

    def test_left_hand_fills_first_half(self):
        landmarks = {i: (float(i) * 0.01, float(i) * 0.01, 0.0) for i in range(21)}
        results = make_results([{'label': 'Left', 'landmarks': landmarks}])
        kp = extract_keypoints(results)
        assert not np.all(kp[:63] == 0)   
        assert np.all(kp[63:] == 0)       

    def test_wrist_normalization(self):
        landmarks = {i: (0.5, 0.5, 0.0) for i in range(21)}
        results = make_results([{'label': 'Right', 'landmarks': landmarks}])
        kp = extract_keypoints(results)
        assert np.allclose(kp[63:66], 0)

    def test_two_hands_both_filled(self):
        lm = {i: (float(i) * 0.01, 0.0, 0.0) for i in range(21)}
        results = make_results([
            {'label': 'Left',  'landmarks': lm},
            {'label': 'Right', 'landmarks': lm},
        ])
        kp = extract_keypoints(results)
        assert not np.all(kp[:63] == 0)
        assert not np.all(kp[63:] == 0)

    def test_output_dtype_is_float(self):
        results = make_results([])
        kp = extract_keypoints(results)
        assert kp.dtype in [np.float32, np.float64]


# ===========================================================================
# 3. check_pinch
# ===========================================================================

class TestCheckPinch:

    def test_pinch_detected_when_close(self):
        hand = make_hand_landmarks({4: (0.5, 0.5), 12: (0.51, 0.5)})
        assert check_pinch(hand) is True

    def test_no_pinch_when_far(self):
        hand = make_hand_landmarks({4: (0.3, 0.5), 12: (0.7, 0.5)})
        assert check_pinch(hand) is False

    def test_pinch_boundary(self):
        hand = make_hand_landmarks({4: (0.5, 0.5), 12: (0.5 + 0.03, 0.5)})
        assert check_pinch(hand) is False  


# ===========================================================================
# 4. GestureLSTM
# ===========================================================================

class TestGestureLSTM:

    def test_output_shape(self):
        model = GestureLSTM(num_classes=len(GESTURES))
        x = torch.randn(4, SEQ_LENGTH, INPUT_SIZE)  # batch=4
        out = model(x)
        assert out.shape == (4, len(GESTURES))

    def test_num_classes_from_config(self):
        model = GestureLSTM()
        x = torch.randn(1, SEQ_LENGTH, INPUT_SIZE)
        out = model(x)
        assert out.shape[1] == len(GESTURES)

    def test_explicit_num_classes(self):
        model = GestureLSTM(num_classes=3)
        x = torch.randn(2, SEQ_LENGTH, INPUT_SIZE)
        out = model(x)
        assert out.shape == (2, 3)

    def test_single_sample_inference(self):
        model = GestureLSTM(num_classes=len(GESTURES))
        model.eval()
        x = torch.randn(1, SEQ_LENGTH, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, len(GESTURES))

    def test_softmax_sums_to_one(self):
        model = GestureLSTM(num_classes=len(GESTURES))
        model.eval()
        x = torch.randn(1, SEQ_LENGTH, INPUT_SIZE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_gradients_flow(self):
        model = GestureLSTM(num_classes=len(GESTURES))
        x = torch.randn(2, SEQ_LENGTH, INPUT_SIZE)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None


# ===========================================================================
# 5. AirMouse
# ===========================================================================

class TestAirMouse:

    def setup_method(self):
        with patch('pyautogui.size', return_value=(1920, 1080)):
            self.mouse = AirMouse(smoothing=1.0)  

    def _make_hand(self, idx_x=0.5, idx_y=0.45,
                   thumb_x=0.5, thumb_y=0.5,
                   mid_x=0.5, mid_y=0.5):
        positions = {
            4:  (thumb_x, thumb_y),   
            6:  (0.5, 0.55),          
            8:  (idx_x, idx_y),       
            10: (0.5, 0.55),          
            12: (mid_x, mid_y),       
        }
        return make_hand_landmarks(positions)

    # --- move ---

    @patch('pyautogui.moveTo')
    def test_move_calls_moveto(self, mock_move):
        hand = self._make_hand(idx_x=0.5, idx_y=0.45)
        self.mouse.move(hand)
        mock_move.assert_called_once()

    @patch('pyautogui.moveTo')
    def test_move_interpolates_to_screen(self, mock_move):
        hand = self._make_hand(idx_x=0.5, idx_y=0.45)
        self.mouse.move(hand)
        args = mock_move.call_args[0]
        assert 800 < args[0] < 1100  

    # --- handle_actions: лівий клік ---

    @patch('pyautogui.mouseDown')
    def test_left_click_press(self, mock_down):
        hand = self._make_hand(thumb_x=0.5, thumb_y=0.5, idx_x=0.51, idx_y=0.5)
        result = self.mouse.handle_actions(hand)
        assert result == "L_DOWN"
        mock_down.assert_called_once()

    @patch('pyautogui.mouseUp')
    def test_left_click_release(self, mock_up):
        self.mouse.is_pressed = True
        hand = self._make_hand(thumb_x=0.3, thumb_y=0.5, idx_x=0.7, idx_y=0.5)
        result = self.mouse.handle_actions(hand)
        assert result == "L_UP"
        mock_up.assert_called_once()
        assert self.mouse.is_pressed is False

    # --- handle_actions: правий клік ---

    @patch('pyautogui.rightClick')
    def test_right_click(self, mock_right):
        hand = self._make_hand(thumb_x=0.5, thumb_y=0.5, mid_x=0.51, mid_y=0.5)
        result = self.mouse.handle_actions(hand)
        assert result == "RIGHT_CLICK"
        mock_right.assert_called_once()

    # --- handle_actions: скролінг ---

    @patch('pyautogui.scroll')
    def test_scroll_up(self, mock_scroll):
        hand = self._make_hand(idx_x=0.5, idx_y=0.40, mid_x=0.52, mid_y=0.40,
                               thumb_x=0.1, thumb_y=0.9)
        self.mouse.handle_actions(hand)
        
        hand2 = self._make_hand(idx_x=0.5, idx_y=0.30, mid_x=0.52, mid_y=0.30,
                                thumb_x=0.1, thumb_y=0.9)
        result = self.mouse.handle_actions(hand2)
        assert result == "SCROLLING"
        mock_scroll.assert_called_once()
        assert mock_scroll.call_args[0][0] > 0  

    @patch('pyautogui.scroll')
    def test_scroll_resets_on_idle(self, mock_scroll):
        hand_scroll = self._make_hand(idx_x=0.5, idx_y=0.40, mid_x=0.52, mid_y=0.40,
                                      thumb_x=0.1, thumb_y=0.9)
        self.mouse.handle_actions(hand_scroll)
        hand_idle = self._make_hand(idx_x=0.5, idx_y=0.60, mid_x=0.7, mid_y=0.60,
                                    thumb_x=0.1, thumb_y=0.9)
        self.mouse.handle_actions(hand_idle)
        assert self.mouse._prev_scroll_y is None

    # --- idle ---

    def test_idle_returns_idle(self):
        hand = self._make_hand(thumb_x=0.1, thumb_y=0.9,
                               idx_x=0.5, idx_y=0.8,
                               mid_x=0.6, mid_y=0.8)
        result = self.mouse.handle_actions(hand)
        assert result == "IDLE"


# ===========================================================================
# 6. Config validation
# ===========================================================================

class TestConfig:

    def test_gestures_not_empty(self):
        assert len(GESTURES) > 0

    def test_gestures_are_strings(self):
        assert all(isinstance(g, str) for g in GESTURES)

    def test_static_gesture_present(self):
        assert 'static' in GESTURES

    def test_seq_length_positive(self):
        assert SEQ_LENGTH > 0

    def test_input_size_matches_landmarks(self):
        assert INPUT_SIZE == 21 * 3 * 2

    def test_threshold_in_valid_range(self):
        assert 0.0 < THRESHOLD < 1.0

    def test_no_duplicate_gestures(self):
        assert len(GESTURES) == len(set(GESTURES))