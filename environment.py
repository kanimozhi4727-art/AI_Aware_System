"""
environment.py  ─  Project Chitti  (v2 — Interactive Voice Assistant)
======================================================================
OpenEnv-compatible RL environment.

What's new in v2
────────────────
• Real speech input pipeline (SpeechRecognition / Whisper)
• MediaPipe hand-gesture observation
• Emergency keyword + alarming-audio detection
• SOS multi-step workflow with gesture confirmation
• All inputs gracefully fall back to simulation when hardware
  is unavailable (no mic / no camera needed for demo)

Observation vector  (8 dims)
────────────────────────────
  [0] wake_word_detected      0 or 1
  [1] speech_type             0=silence  0.5=normal  1.0=emergency
  [2] gesture_sos_strength    0–1  (normalised SOS gesture count)
  [3] alarming_audio          0 or 1
  [4] assistant_active        0 or 1
  [5] sos_stage               0 / 0.33 / 0.66 / 1.0
  [6] permission_ok           0 or 1
  [7] gesture_confirm_count   0–1  (normalised; for SOS confirmation)

Action space  (6 discrete)
──────────────────────────
  0  Stay Idle
  1  Activate Assistant
  2  Respond in Conversation
  3  Show SOS Button
  4  Trigger Emergency Call
  5  Notify Trusted Contact
"""

import random
import threading
import time
import numpy as np

# ── Optional real-hardware imports (graceful fallback) ─────────────
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    import mediapipe as mp
    import cv2
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────
#  ACTION CONSTANTS
# ──────────────────────────────────────────────────────────────────
ACTION_IDLE            = 0
ACTION_ACTIVATE        = 1
ACTION_RESPOND         = 2
ACTION_SHOW_SOS_BTN    = 3
ACTION_TRIGGER_CALL    = 4
ACTION_NOTIFY_CONTACT  = 5

ACTION_NAMES = {
    ACTION_IDLE:           "Stay Idle",
    ACTION_ACTIVATE:       "Activate Assistant",
    ACTION_RESPOND:        "Respond in Conversation",
    ACTION_SHOW_SOS_BTN:   "Show SOS Button",
    ACTION_TRIGGER_CALL:   "Trigger Emergency Call",
    ACTION_NOTIFY_CONTACT: "Notify Trusted Contact",
}

# ──────────────────────────────────────────────────────────────────
#  KEYWORD LISTS
# ──────────────────────────────────────────────────────────────────
WAKE_WORDS        = ["hey chitti", "ok chitti", "chitti"]
EMERGENCY_WORDS   = ["help", "danger", "scary", "stop", "emergency",
                     "sos", "save me", "call police", "fire", "attack"]
ALARMING_SOUNDS   = ["glass breaking", "scream", "fire alarm",
                     "crash", "explosion", "siren"]

# ──────────────────────────────────────────────────────────────────
#  SPEECH INPUT HANDLER
# ──────────────────────────────────────────────────────────────────
class SpeechHandler:
    """
    Handles real microphone input via SpeechRecognition.
    Falls back to keyboard / simulated input when no mic is present.

    Parameters
    ----------
    use_real_mic : bool   – attempt to use real microphone
    language     : str    – BCP-47 language tag (e.g. "en-IN")
    """

    def __init__(self, use_real_mic: bool = False, language: str = "en-IN"):
        self.use_real_mic = use_real_mic and SR_AVAILABLE
        self.language     = language
        self.last_text    = ""

        if self.use_real_mic:
            self.recognizer = sr.Recognizer()
            self.mic        = sr.Microphone()
            # Calibrate for ambient noise once at startup
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("[SpeechHandler] 🎙️  Real microphone initialised.")
        else:
            print("[SpeechHandler] 🔇  Simulation mode (no real microphone).")

    def listen_once(self, timeout: float = 3.0) -> str:
        """
        Capture one phrase.

        Returns the recognised text (lower-case) or "" on failure.
        """
        if not self.use_real_mic:
            return self._simulate_speech()

        try:
            with self.mic as source:
                audio = self.recognizer.listen(source, timeout=timeout,
                                               phrase_time_limit=5)
            text = self.recognizer.recognize_google(
                audio, language=self.language
            ).lower()
            self.last_text = text
            return text
        except (sr.WaitTimeoutError, sr.UnknownValueError,
                sr.RequestError):
            return ""

    def _simulate_speech(self) -> str:
        """Return a random simulated utterance."""
        roll = random.random()
        if roll < 0.10:
            return random.choice(WAKE_WORDS)
        elif roll < 0.20:
            return random.choice(EMERGENCY_WORDS)
        elif roll < 0.50:
            return random.choice([
                "what's the weather",
                "play some music",
                "set a reminder",
                "tell me a joke",
                "what time is it",
            ])
        return ""  # silence


# ──────────────────────────────────────────────────────────────────
#  GESTURE HANDLER
# ──────────────────────────────────────────────────────────────────
class GestureHandler:
    """
    Detects SOS / stop / open-palm hand gestures via MediaPipe.
    Falls back to random simulation when no camera is available.

    Recognised as SOS gesture: open palm (all 5 fingers extended).

    Parameters
    ----------
    use_real_camera : bool  – attempt to open webcam
    """

    # Finger tip landmark IDs in MediaPipe Hands
    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [3, 6, 10, 14, 18]

    def __init__(self, use_real_camera: bool = False):
        self.use_real_camera = use_real_camera and MP_AVAILABLE
        self.gesture_history : list[str] = []
        self.confirm_count   = 0   # consecutive SOS confirmations

        if self.use_real_camera:
            mp_hands       = mp.solutions.hands
            self.hands     = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
            )
            self.cap = cv2.VideoCapture(0)
            print("[GestureHandler] ✋  Real camera initialised.")
        else:
            print("[GestureHandler] 🎭  Simulation mode (no real camera).")

    def detect_once(self) -> str:
        """
        Capture one frame and classify gesture.

        Returns one of: "sos_palm" | "idle" | "none"
        """
        if not self.use_real_camera:
            return self._simulate_gesture()

        ret, frame = self.cap.read()
        if not ret:
            return "none"

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return "none"

        landmarks = result.multi_hand_landmarks[0].landmark
        return self._classify(landmarks)

    def _classify(self, landmarks) -> str:
        """
        Return 'sos_palm' if all 5 fingers are extended, else 'idle'.
        """
        extended = 0
        for tip_id, pip_id in zip(self.FINGER_TIPS, self.FINGER_PIPS):
            # Thumb: compare x; other fingers: compare y
            if tip_id == 4:
                if landmarks[tip_id].x < landmarks[pip_id].x:
                    extended += 1
            else:
                if landmarks[tip_id].y < landmarks[pip_id].y:
                    extended += 1
        return "sos_palm" if extended >= 4 else "idle"

    def _simulate_gesture(self) -> str:
        """Random simulated gesture (12 % chance of SOS palm)."""
        return "sos_palm" if random.random() < 0.12 else "idle"

    def update_confirm_count(self, gesture: str):
        """Track consecutive SOS gestures for confirmation logic."""
        if gesture == "sos_palm":
            self.confirm_count = min(self.confirm_count + 1, 3)
        else:
            self.confirm_count = max(self.confirm_count - 1, 0)

    def release(self):
        if self.use_real_camera:
            self.cap.release()


# ──────────────────────────────────────────────────────────────────
#  MAIN ENVIRONMENT
# ──────────────────────────────────────────────────────────────────
class ChittiEnv:
    """
    OpenEnv-compatible environment for Project Chitti v2.

    Parameters
    ----------
    permissions     : dict   – which input channels are active
    max_steps       : int    – episode length
    use_real_mic    : bool   – use real microphone via SpeechRecognition
    use_real_camera : bool   – use real webcam via MediaPipe
    """

    OBS_DIM = 8

    def __init__(
        self,
        permissions      : dict = None,
        max_steps        : int  = 50,
        use_real_mic     : bool = False,
        use_real_camera  : bool = False,
    ):
        self.permissions = permissions or {
            "listening_enabled": True,
            "gesture_enabled":   True,
            "sos_enabled":       True,
        }
        self.max_steps = max_steps

        # Input handlers
        self.speech_handler  = SpeechHandler(use_real_mic=use_real_mic)
        self.gesture_handler = GestureHandler(use_real_camera=use_real_camera)

        # Step-level observations (set by _simulate_inputs)
        self._raw_text          = ""
        self._current_wake_word = False
        self._current_speech_type     = 0.0
        self._current_gesture         = "none"
        self._current_audio_alarm     = 0.0
        self._current_gesture_sos     = 0.0

        self.reset()

    # ── OpenEnv API ────────────────────────────────────────────────

    def reset(self):
        self.step_count              = 0
        self.assistant_active        = False
        self.sos_stage               = 0       # 0–3
        self.sos_gesture_count       = 0
        self.emergency_keyword_count = 0
        self.last_action             = None
        self.info                    = {}
        self._raw_text               = ""
        self._current_wake_word      = False
        self._current_speech_type    = 0.0
        self._current_gesture        = "none"
        self._current_audio_alarm    = 0.0
        self._current_gesture_sos    = 0.0
        self.gesture_handler.confirm_count = 0
        return self._get_obs()

    def step(self, action: int, injected_text: str = ""):
        """
        Advance environment one step.

        Parameters
        ----------
        action        : int  – agent action index
        injected_text : str  – override speech input (used by Gradio UI)

        Returns obs, reward, done, info
        """
        self.step_count += 1
        self._gather_inputs(injected_text=injected_text)
        reward, info = self._compute_reward(action)
        self._apply_action(action)
        obs  = self._get_obs()
        done = self.step_count >= self.max_steps
        self.last_action = action
        self.info        = info
        return obs, reward, done, info

    @property
    def action_space_n(self):
        return 6

    @property
    def observation_space_shape(self):
        return (self.OBS_DIM,)

    # ── Input gathering ────────────────────────────────────────────

    def _gather_inputs(self, injected_text: str = ""):
        """Collect observations from speech and gesture handlers."""
        p = self.permissions

        # ── Speech ──────────────────────────────────────────────
        if p["listening_enabled"]:
            text = injected_text if injected_text else \
                   self.speech_handler.listen_once()
            self._raw_text = text.lower().strip()
        else:
            self._raw_text = ""

        # Classify speech
        self._current_wake_word   = any(w in self._raw_text
                                         for w in WAKE_WORDS)
        is_emergency              = any(w in self._raw_text
                                         for w in EMERGENCY_WORDS)
        has_normal_speech         = bool(self._raw_text) and \
                                    not self._current_wake_word

        if is_emergency:
            self._current_speech_type = 1.0
            self.emergency_keyword_count = min(
                self.emergency_keyword_count + 1, 3
            )
        elif has_normal_speech:
            self._current_speech_type = 0.5
            self.emergency_keyword_count = max(
                self.emergency_keyword_count - 1, 0
            )
        else:
            self._current_speech_type = 0.0
            self.emergency_keyword_count = max(
                self.emergency_keyword_count - 1, 0
            )

        # ── Gesture ──────────────────────────────────────────────
        if p["gesture_enabled"]:
            self._current_gesture = self.gesture_handler.detect_once()
            self.gesture_handler.update_confirm_count(self._current_gesture)
            if self._current_gesture == "sos_palm":
                self.sos_gesture_count = min(self.sos_gesture_count + 1, 3)
            else:
                self.sos_gesture_count = max(self.sos_gesture_count - 1, 0)
            self._current_gesture_sos = self.sos_gesture_count / 3.0
        else:
            self._current_gesture     = "none"
            self._current_gesture_sos = 0.0

        # ── Alarming audio (simulated) ────────────────────────────
        # In a real system: run an audio classifier here
        self._current_audio_alarm = (
            1.0 if any(s in self._raw_text for s in ALARMING_SOUNDS)
            else (1.0 if random.random() < 0.05 else 0.0)
        )

    # ── Observation vector ─────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        perm_ok = float(
            self.permissions["listening_enabled"] or
            self.permissions["gesture_enabled"]
        )
        return np.array([
            float(self._current_wake_word),
            self._current_speech_type,
            self._current_gesture_sos,
            self._current_audio_alarm,
            float(self.assistant_active),
            self.sos_stage / 3.0,
            perm_ok,
            self.gesture_handler.confirm_count / 3.0,
        ], dtype=np.float32)

    # ── Reward ─────────────────────────────────────────────────────

    def _compute_reward(self, action: int):
        reward = 0.0
        info   = {"reason": [], "action_name": ACTION_NAMES[action]}

        wake          = self._current_wake_word
        etype         = self._current_speech_type == 1.0
        gsos          = self._current_gesture_sos >= 0.66
        alarm         = self._current_audio_alarm == 1.0
        emergency_cue = etype or gsos or alarm

        # Permission check
        if not self.permissions["listening_enabled"] and \
                action in (ACTION_ACTIVATE, ACTION_RESPOND):
            reward -= 5
            info["reason"].append("violated listening_disabled (−5)")
        elif not self.permissions["sos_enabled"] and \
                action in (ACTION_SHOW_SOS_BTN, ACTION_TRIGGER_CALL,
                           ACTION_NOTIFY_CONTACT):
            reward -= 5
            info["reason"].append("violated sos_disabled (−5)")
        else:
            reward += 10
            info["reason"].append("respected permissions (+10)")

        # Wake-word activation
        if action == ACTION_ACTIVATE:
            if wake:
                reward += 20
                info["reason"].append("correct wake-word (+20)")
            else:
                reward -= 5
                info["reason"].append("activated without wake word (−5)")

        # Idle reward
        if action == ACTION_IDLE and not wake and not emergency_cue:
            reward += 10
            info["reason"].append("correctly stayed idle (+10)")

        # Emergency actions
        if action in (ACTION_SHOW_SOS_BTN, ACTION_TRIGGER_CALL,
                      ACTION_NOTIFY_CONTACT):
            if emergency_cue and self.permissions["sos_enabled"]:
                reward += 20
                info["reason"].append("correct emergency action (+20)")
            else:
                reward -= 10
                info["reason"].append("false alarm (−10)")

        # Missed emergency
        if emergency_cue and action == ACTION_IDLE:
            reward -= 20
            info["reason"].append("missed emergency (−20)")

        info["total_reward"] = reward
        return reward, info

    # ── State transition ───────────────────────────────────────────

    def _apply_action(self, action: int):
        if action == ACTION_ACTIVATE and self._current_wake_word:
            self.assistant_active = True

        elif action == ACTION_SHOW_SOS_BTN:
            if self.sos_stage < 1 and self.permissions["sos_enabled"]:
                self.sos_stage = 1

        elif action == ACTION_TRIGGER_CALL:
            if self.sos_stage >= 1 and self.permissions["sos_enabled"]:
                self.sos_stage = min(self.sos_stage + 1, 3)

        elif action == ACTION_NOTIFY_CONTACT:
            if self.sos_stage >= 2:
                self.sos_stage = 3

        # Auto-sleep assistant after long idle
        if action == ACTION_IDLE and self.assistant_active:
            if random.random() < 0.04:
                self.assistant_active = False

    # ── Describe ───────────────────────────────────────────────────

    def describe_observation(self) -> dict:
        """Human-readable snapshot of the current observation."""
        speech_map = {0.0: "Silence", 0.5: "Normal speech",
                      1.0: "Emergency speech ⚠️"}
        sos_map    = {0: "None", 1: "SOS Button Shown ⚠️",
                      2: "Countdown 🚨", 3: "Emergency Call Triggered 📞"}
        return {
            "raw_text"         : self._raw_text or "(silence)",
            "wake_word"        : self._current_wake_word,
            "speech_type"      : speech_map.get(self._current_speech_type, "?"),
            "gesture"          : self._current_gesture,
            "gesture_confirms" : self.gesture_handler.confirm_count,
            "alarming_audio"   : bool(self._current_audio_alarm),
            "assistant_active" : self.assistant_active,
            "sos_stage"        : sos_map[self.sos_stage],
            "permissions"      : self.permissions,
        }

    def cleanup(self):
        """Release camera/mic resources."""
        self.gesture_handler.release()