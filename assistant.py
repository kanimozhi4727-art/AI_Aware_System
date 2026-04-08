"""
assistant.py  ─  Project Chitti v2
=====================================
Chitti's conversational brain.

What's new in v2
────────────────
• Text-to-Speech via pyttsx3 (offline) with gTTS fallback
• Claude / Siri-style natural responses using keyword matching
• Simulated memory with user name + topic history
• Multi-step SOS workflow messages with TTS output
• Thread-safe TTS so the RL loop never blocks
"""

import os
import random
import threading
import time
from dataclasses import dataclass, field

# ── TTS imports (graceful fallback) ───────────────────────────────
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    import tempfile
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
#  TTS ENGINE
# ──────────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Text-to-Speech wrapper.

    Priority: pyttsx3 (offline) → gTTS (online) → print-only fallback.
    All speech is dispatched on a background thread so it never
    blocks the RL loop.
    """

    def __init__(self, rate: int = 175, volume: float = 0.9):
        self._lock   = threading.Lock()
        self._engine = None
        self.mode    = "none"

        if PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate",   rate)
                engine.setProperty("volume", volume)
                # Prefer a female voice if available
                voices = engine.getProperty("voices")
                for v in voices:
                    if "female" in v.name.lower() or \
                       "zira"   in v.name.lower() or \
                       "hazel"  in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
                self._engine = engine
                self.mode    = "pyttsx3"
                print("[TTS] 🔊  pyttsx3 (offline) ready.")
            except Exception as e:
                print(f"[TTS] pyttsx3 init failed: {e}")

        if self.mode == "none" and GTTS_AVAILABLE:
            self.mode = "gtts"
            print("[TTS] 🌐  gTTS (online) ready.")

        if self.mode == "none":
            print("[TTS] 📝  Print-only mode (no TTS library).")

    def speak(self, text: str, blocking: bool = False):
        """
        Speak `text`.  Set blocking=True to wait for completion.
        """
        if blocking:
            self._speak_now(text)
        else:
            t = threading.Thread(target=self._speak_now,
                                 args=(text,), daemon=True)
            t.start()

    def _speak_now(self, text: str):
        print(f"[Chitti 🗣️ ] {text}")

        if self.mode == "pyttsx3" and self._engine:
            with self._lock:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception:
                    pass  # silent fallback

        elif self.mode == "gtts":
            try:
                tts  = gTTS(text=text, lang="en", slow=False)
                with tempfile.NamedTemporaryFile(
                    suffix=".mp3", delete=False
                ) as f:
                    tts.save(f.name)
                    tmp = f.name
                os.system(f"mpg123 -q {tmp} 2>/dev/null || "
                          f"afplay {tmp} 2>/dev/null || "
                          f"aplay  {tmp} 2>/dev/null")
                os.unlink(tmp)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────
#  SIMULATED MEMORY
# ──────────────────────────────────────────────────────────────────

@dataclass
class UserMemory:
    name              : str  = "Friend"
    preferences       : dict = field(default_factory=dict)
    interaction_count : int  = 0
    last_topic        : str  = ""
    mood              : str  = "neutral"

    def remember(self, key, value):
        self.preferences[key] = value

    def recall(self, key, default=None):
        return self.preferences.get(key, default)


# ──────────────────────────────────────────────────────────────────
#  RESPONSE BANKS
# ──────────────────────────────────────────────────────────────────

WAKE_GREETINGS = [
    "Hey {name}! How can I help you? 😊",
    "Hi {name}! Chitti is here. What do you need?",
    "Hello! I'm listening, {name}.",
    "Yes {name}? I'm all ears!",
]

TOPIC_RESPONSES = {
    "weather"  : ["I'll check the weather for you!", "Let me pull up the forecast."],
    "music"    : ["Playing your favourite tunes!", "Sure, starting some music!"],
    "reminder" : ["Reminder set! I'll alert you.", "Done — reminder added."],
    "joke"     : [
        "Why did the robot go to school? To improve its A.I. grade! 😄",
        "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads! 🍫",
    ],
    "time"     : ["I'll check the current time for you."],
    "default"  : [
        "Sure! Let me take care of that, {name}.",
        "Got it! I'll handle that right away.",
        "Of course! Give me a moment.",
        "No problem at all, {name}!",
    ],
}

SOS_MESSAGES = {
    1: "⚠️  I noticed something might be wrong. I've shown the SOS button. "
       "Show me an open palm to confirm.",
    2: "🚨  SOS confirmed! Starting a 10-second countdown. "
       "Stay calm — I'm here with you.",
    3: "📞  Calling emergency services now! "
       "Your trusted contact has been notified with your location.",
}

IDLE_MESSAGES = [
    "🎧 Monitoring quietly. Say 'Hey Chitti' to wake me.",
    "👁️  I'm watching over you silently.",
    "🔇 Standing by. I'm here if you need me.",
]


# ──────────────────────────────────────────────────────────────────
#  ASSISTANT CLASS
# ──────────────────────────────────────────────────────────────────

class ChittiAssistant:
    """
    Handles personality, TTS, memory, and SOS workflow.

    Parameters
    ----------
    user_name  : str   – the user's name
    tts_rate   : int   – speech rate (words/min) for pyttsx3
    speak_aloud: bool  – actually play TTS audio (False = print only)
    """

    def __init__(
        self,
        user_name   : str  = "Arun",
        tts_rate    : int  = 175,
        speak_aloud : bool = True,
    ):
        self.memory    = UserMemory(name=user_name)
        self.tts       = TTSEngine(rate=tts_rate) if speak_aloud \
                         else TTSEngine.__new__(TTSEngine)
        if not speak_aloud:
            # Mute TTS: only print
            self.tts.mode = "none"
        self.sos_stage = 0
        self.active    = False
        self.logs      : list[str] = []

    # ── Public API ─────────────────────────────────────────────────

    def on_wake_word(self) -> str:
        """Call when wake word detected and ACTION_ACTIVATE chosen."""
        self.active = True
        self.memory.interaction_count += 1
        msg = random.choice(WAKE_GREETINGS).format(
            name=self.memory.name
        )
        self._say(msg)
        return msg

    def on_speech(self, text: str) -> str:
        """
        Generate a context-aware reply to user speech.
        Matches keywords to select the best response category.
        """
        if not self.active:
            return ""
        self.memory.last_topic = text

        # Keyword matching for topic-specific replies
        for keyword, replies in TOPIC_RESPONSES.items():
            if keyword != "default" and keyword in text.lower():
                msg = random.choice(replies)
                self._say(msg)
                return msg

        # Default reply
        msg = random.choice(TOPIC_RESPONSES["default"]).format(
            name=self.memory.name
        )
        self._say(msg)
        return msg

    def on_idle(self) -> str:
        """Passive monitoring message (not spoken, just logged)."""
        msg = random.choice(IDLE_MESSAGES)
        self._log(f"[IDLE] {msg}")
        return msg

    def on_sos_stage(self, stage: int) -> str:
        """Progress the SOS workflow and speak the alert."""
        if stage < 1 or stage > 3:
            return ""
        self.sos_stage = stage
        msg = SOS_MESSAGES[stage]
        self._say(msg)
        return msg

    def cancel_sos(self):
        self.sos_stage = 0
        msg = "SOS cancelled. Everything is okay."
        self._say(msg)
        self._log("[SOS] Cancelled.")

    def get_status(self) -> dict:
        return {
            "active"           : self.active,
            "sos_stage"        : self.sos_stage,
            "user_name"        : self.memory.name,
            "interaction_count": self.memory.interaction_count,
            "last_topic"       : self.memory.last_topic,
        }

    def get_logs(self, last_n: int = 30) -> list[str]:
        return self.logs[-last_n:]

    # ── Private ────────────────────────────────────────────────────

    def _say(self, text: str):
        """Log + speak a message."""
        self._log(text)
        self.tts.speak(text)

    def _log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")