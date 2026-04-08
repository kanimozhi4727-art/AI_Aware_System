# 🤖 Project Chitti v2 — Interactive RL Voice Assistant

> *Voice-activated. Emergency-aware. Gesture-confirmed.*

---

## 📁 File Structure

```
chitti_v2/
├── environment.py    ← OpenEnv RL env + SpeechRecognition + MediaPipe hooks
├── agent.py          ← PyTorch DQN agent (8-dim obs, 6 actions)
├── assistant.py      ← TTS engine + Siri-style responses + memory
├── main.py           ← CLI simulation + full Gradio web demo
├── requirements.txt  ← All dependencies
└── README.md         ← This file
```

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. CLI simulation (simulated inputs — no hardware needed)
```bash
python main.py --episodes 4 --steps 20
```

### 3. Gradio web demo
```bash
python main.py --gradio
# Open http://localhost:7860
```

### 4. Real microphone
```bash
python main.py --gradio --real-mic
```

### 5. Real webcam (MediaPipe hand gestures)
```bash
python main.py --gradio --real-cam
```

### 6. Full hardware mode
```bash
python main.py --gradio --real-mic --real-cam
```

---

## 🏗️ Architecture

### Observation Vector (8 dims)

| Index | Name | Range | Description |
|---|---|---|---|
| 0 | `wake_word_detected` | 0–1 | "Hey Chitti" heard |
| 1 | `speech_type` | 0/0.5/1.0 | silence / normal / emergency |
| 2 | `gesture_sos_strength` | 0–1 | normalised SOS gesture count |
| 3 | `alarming_audio` | 0–1 | glass breaking / scream / alarm |
| 4 | `assistant_active` | 0–1 | assistant is awake |
| 5 | `sos_stage` | 0–1 | 0/0.33/0.66/1.0 (stages 0–3) |
| 6 | `permission_ok` | 0–1 | any permission enabled |
| 7 | `gesture_confirm_count` | 0–1 | consecutive SOS palm detections |

### Actions

| ID | Name | Trigger |
|---|---|---|
| 0 | Stay Idle | No wake word, no emergency |
| 1 | Activate Assistant | Wake word detected |
| 2 | Respond in Conversation | Normal speech while active |
| 3 | Show SOS Button | First emergency cue |
| 4 | Trigger Emergency Call | Second confirmation |
| 5 | Notify Trusted Contact | Third / final step |

### Reward Table

| Event | Reward |
|---|---|
| Correct wake-word activation | +20 |
| Stay idle before wake word | +10 |
| Correct emergency detection | +20 |
| Respecting permissions | +10 |
| Activated without wake word | −5 |
| False alarm | −10 |
| Missed emergency | −20 |
| Permission violation | −5 |

---

## 🎙️ Speech Input (SpeechRecognition)

| Mode | How to enable | What happens |
|---|---|---|
| Simulation | Default | Random utterances from keyword banks |
| Real mic | `--real-mic` | Google Speech API via SpeechRecognition |
| Typed text | Gradio UI | Text field bypasses mic entirely |

Emergency keywords: `help, danger, scary, stop, emergency, sos, save me, call police, fire, attack`

Wake words: `hey chitti, ok chitti, chitti`

---

## ✋ Hand Gesture Detection (MediaPipe)

| Mode | How to enable |
|---|---|
| Simulation | Default (12% chance of SOS palm per step) |
| Real webcam | `--real-cam` |

**SOS gesture = open palm (4+ fingers extended)**

Confirmation logic:
- 1 detection  → gesture progress bar fills
- 2 detections → SOS stage advances
- 3 detections → emergency call triggered automatically

---

## 🔊 Text-to-Speech

| Engine | Availability | Notes |
|---|---|---|
| `pyttsx3` | Offline, preferred | Fast, no internet needed |
| `gTTS` | Online fallback | Requires `mpg123` / `afplay` |
| Print-only | Always available | Logs to console |

---

## 🖥️ Gradio Interface Tabs

| Tab | Content |
|---|---|
| 💬 Conversation | Live chat with Chitti, text input |
| 🚨 Emergency | SOS button (hidden by default), stage display |
| 🧠 Agent | Real-time observation + reward breakdown |
| 📋 Logs | Full assistant activity log |

---

## 🌐 Deploy to Hugging Face Spaces

1. Create a new Space (SDK: **Gradio**)
2. Upload: `environment.py`, `agent.py`, `assistant.py`, `main.py`, `requirements.txt`
3. Create `app.py`:
```python
from main import launch_gradio
launch_gradio()
```
4. Done!  Hugging Face will install requirements automatically.

---

## 💡 Example CLI Output

```
═══════════════════════════════════════════════════════════
  🤖  PROJECT CHITTI v2 — INTERACTIVE VOICE ASSISTANT
═══════════════════════════════════════════════════════════

  Step  1 │ Activate Assistant
         │ Input   : "hey chitti"
         │ Speech  : Normal speech
         │ Gesture : idle  (confirms=0)
         │ SOS     : None
         │ Reward  : +30.0  (respected permissions (+10), correct wake-word (+20))
         │ 🤖       : Hey Arun! How can I help you? 😊

  Step  4 │ Show SOS Button
         │ Input   : "help help please danger"
         │ Speech  : Emergency speech ⚠️
         │ Gesture : sos_palm  (confirms=1)
         │ SOS     : SOS Button Shown ⚠️
         │ Reward  : +30.0  (respected permissions (+10), correct emergency (+20))
         │ 🤖       : ⚠️  I noticed something might be wrong. Show me an open palm.

  Step  6 │ Trigger Emergency Call
         │ Input   : "help"
         │ Gesture : sos_palm  (confirms=3)
         │ SOS     : Countdown 🚨
         │ 🤖       : 🚨  SOS confirmed! Starting 10-second countdown.
```

---

