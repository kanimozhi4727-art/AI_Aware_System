"""
main.py  ─  Project Chitti v2  (Interactive Voice Assistant)
=============================================================
Entry point.  Two modes:

  python main.py              → CLI simulation loop
  python main.py --gradio     → Full Gradio web interface
  python main.py --real-mic   → Use real microphone (SpeechRecognition)
  python main.py --real-cam   → Use real webcam (MediaPipe)

  Combine flags as needed:
    python main.py --gradio --real-mic --real-cam

Gradio interface features
──────────────────────────
  • Type (or speak) to Chitti in normal conversation mode
  • Emergency keywords auto-trigger SOS workflow
  • SOS button appears only when emergency is detected
  • Gesture confirmation panel (simulated or real camera)
  • Live reward / agent decision feedback
  • Tabbed layout: Chat | Emergency | Agent | Logs
"""

import argparse
import sys
import time
import random

import numpy as np

from environment import (
    ChittiEnv,
    ACTION_NAMES,
    ACTION_IDLE, ACTION_ACTIVATE, ACTION_RESPOND,
    ACTION_SHOW_SOS_BTN, ACTION_TRIGGER_CALL, ACTION_NOTIFY_CONTACT,
    WAKE_WORDS, EMERGENCY_WORDS,
)
from agent     import ChittiAgent
from assistant import ChittiAssistant


# ══════════════════════════════════════════════════════════════════
#  CLI SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════

SEPARATOR = "─" * 64

def banner(msg):
    print(f"\n{'═'*64}\n  {msg}\n{'═'*64}")

def run_cli(
    n_episodes   : int  = 4,
    n_steps      : int  = 20,
    use_real_mic : bool = False,
    use_real_cam : bool = False,
):
    env       = ChittiEnv(
        use_real_mic=use_real_mic,
        use_real_camera=use_real_cam,
        max_steps=n_steps,
    )
    agent     = ChittiAgent()
    assistant = ChittiAssistant(user_name="Arun", speak_aloud=True)
    obs       = env.reset()

    banner("🤖  PROJECT CHITTI v2 — INTERACTIVE VOICE ASSISTANT")
    print(f"  Episodes : {n_episodes}  |  Steps : {n_steps}")
    print(f"  Real mic : {use_real_mic}  |  Real cam : {use_real_cam}")

    ep_rewards = []

    for ep in range(1, n_episodes + 1):
        obs          = env.reset()
        total_reward = 0.0
        print(f"\n{SEPARATOR}\n  EPISODE {ep}/{n_episodes}\n{SEPARATOR}")

        for t in range(n_steps):
            action               = agent.select_action(obs)
            next_obs, rew, done, info = env.step(action)
            agent.store(obs, action, rew, next_obs, done)
            loss = agent.learn()
            obs  = next_obs
            total_reward += rew

            desc          = env.describe_observation()
            assistant_msg = _dispatch_assistant(action, desc, assistant)

            _print_step(t + 1, action, desc, rew, info,
                        assistant_msg, loss, agent.epsilon)
            if done:
                break

        ep_rewards.append(total_reward)
        print(f"\n  ✅  Episode {ep}  |  Reward: {total_reward:+.1f}"
              f"  |  ε = {agent.epsilon:.3f}")

    banner("📊  SUMMARY")
    for i, r in enumerate(ep_rewards, 1):
        bar = "█" * max(0, int((r + 200) / 15))
        print(f"  Ep {i}: {r:+7.1f}  {bar}")
    print(f"\n  Mean : {np.mean(ep_rewards):+.2f}  |  "
          f"Best : {max(ep_rewards):+.2f}")

    env.cleanup()


def _dispatch_assistant(action, desc, assistant: ChittiAssistant) -> str:
    """Route action + observation to the correct assistant method."""
    if action == ACTION_ACTIVATE and desc["wake_word"]:
        return assistant.on_wake_word()
    elif action == ACTION_RESPOND:
        return assistant.on_speech(desc["raw_text"])
    elif action == ACTION_IDLE:
        return assistant.on_idle()
    elif action == ACTION_SHOW_SOS_BTN:
        return assistant.on_sos_stage(1)
    elif action == ACTION_TRIGGER_CALL:
        return assistant.on_sos_stage(2)
    elif action == ACTION_NOTIFY_CONTACT:
        return assistant.on_sos_stage(3)
    return ""


def _print_step(t, action, desc, reward, info, msg, loss, eps):
    print(f"\n  Step {t:>2} │ {ACTION_NAMES[action]}")
    print(f"         │ Input   : \"{desc['raw_text']}\"")
    print(f"         │ Speech  : {desc['speech_type']}")
    print(f"         │ Gesture : {desc['gesture']}"
          f"  (confirms={desc['gesture_confirms']})")
    print(f"         │ SOS     : {desc['sos_stage']}")
    print(f"         │ Reward  : {reward:+.1f}  "
          f"({', '.join(info['reason'])})")
    if msg:
        print(f"         │ 🤖       : {msg}")
    if loss is not None:
        print(f"         │ Loss    : {loss:.4f}  ε={eps:.3f}")


# ══════════════════════════════════════════════════════════════════
#  GRADIO WEB INTERFACE
# ══════════════════════════════════════════════════════════════════

def launch_gradio(use_real_mic=False, use_real_cam=False):
    try:
        import gradio as gr
    except ImportError:
        print("Install Gradio:  pip install gradio")
        sys.exit(1)

    # ── Shared state ───────────────────────────────────────────────
    env       = ChittiEnv(use_real_mic=False,    # Gradio uses text input
                          use_real_camera=use_real_cam)
    agent     = ChittiAgent()
    assistant = ChittiAssistant(user_name="Arun", speak_aloud=True)
    obs_ref   = [env.reset()]
    stats     = {"total_reward": 0.0, "steps": 0, "episodes": 0}

    # Conversation history for the chatbot panel
    chat_history: list[tuple[str, str]] = []

    # ── Core step function ─────────────────────────────────────────
    def chitti_step(user_text: str,
                    listen_on: bool, gesture_on: bool, sos_on: bool):
        """
        One RL step driven by Gradio inputs.

        Called when user submits text OR clicks 'Step'.
        Returns updated UI components.
        """
        nonlocal obs_ref, chat_history

        # Update permissions
        env.permissions = {
            "listening_enabled": listen_on,
            "gesture_enabled":   gesture_on,
            "sos_enabled":       sos_on,
        }

        # Step the environment (injecting typed text as speech)
        obs                      = obs_ref[0]
        action                   = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action, injected_text=user_text)
        agent.store(obs, action, reward, next_obs, done)
        loss = agent.learn()
        obs_ref[0] = next_obs

        stats["total_reward"] += reward
        stats["steps"]        += 1
        if done:
            obs_ref[0] = env.reset()
            stats["episodes"] += 1

        desc          = env.describe_observation()
        assistant_msg = _dispatch_assistant(action, desc, assistant)

        # Update chat history
        if user_text.strip():
            chat_history.append((user_text, assistant_msg or "…"))
        elif assistant_msg and action != ACTION_IDLE:
            chat_history.append(("(background detection)", assistant_msg))

        # ── Build UI outputs ───────────────────────────────────────

        # Observation panel
        obs_md = (
            f"**Input received:** `{desc['raw_text']}`\n\n"
            f"| Signal | Value |\n|---|---|\n"
            f"| Wake word | {'✅ YES' if desc['wake_word'] else '—'} |\n"
            f"| Speech | {desc['speech_type']} |\n"
            f"| Gesture | `{desc['gesture']}` |\n"
            f"| SOS confirms | {desc['gesture_confirms']} / 3 |\n"
            f"| Alarming audio | {'🔔 YES' if desc['alarming_audio'] else '—'} |\n"
            f"| Assistant active | {'🟢 YES' if desc['assistant_active'] else '🔴 No'} |\n"
        )

        # Agent decision panel
        agent_md = (
            f"**Action:** `{ACTION_NAMES[action]}`\n\n"
            f"**Reward this step:** `{reward:+.1f}`\n\n"
            f"**Reasons:**\n"
            + "\n".join(f"- {r}" for r in info["reason"])
            + f"\n\n**Cumulative reward:** `{stats['total_reward']:+.1f}`\n"
            f"**Steps:** `{stats['steps']}`  "
            f"**Exploration ε:** `{agent.epsilon:.3f}`\n"
            + (f"**Loss:** `{loss:.4f}`" if loss else "")
        )

        # SOS panel — show stage + gesture confirmation progress
        sos_visible = env.sos_stage >= 1 and sos_on
        sos_md = ""
        if env.sos_stage == 0:
            sos_md = "✅  No emergency detected.  SOS button hidden."
        elif env.sos_stage == 1:
            confirms = desc['gesture_confirms']
            bar      = "🟥" * confirms + "⬜" * (3 - confirms)
            sos_md   = (
                f"## ⚠️  SOS BUTTON ACTIVE\n\n"
                f"Emergency detected!  Show an **open palm** "
                f"gesture to confirm.\n\n"
                f"**Gesture progress:** {bar}  ({confirms}/3)\n\n"
                f"_{assistant_msg}_"
            )
        elif env.sos_stage == 2:
            sos_md = (
                f"## 🚨  COUNTDOWN ACTIVE\n\n"
                f"10-second countdown started.  Calling soon...\n\n"
                f"_{assistant_msg}_"
            )
        elif env.sos_stage == 3:
            sos_md = (
                f"## 📞  EMERGENCY CALL TRIGGERED\n\n"
                f"**Calling 112 / 100 — Trusted contact notified!**\n\n"
                f"_{assistant_msg}_"
            )

        # Logs
        logs_text = "\n".join(assistant.get_logs(last_n=20))

        return (
            chat_history,   # chatbot
            obs_md,         # observation panel
            agent_md,       # agent decision
            sos_md,         # sos panel
            gr.update(visible=sos_visible),   # SOS button visibility
            logs_text,      # logs textbox
            "",             # clear input
        )

    def reset_session():
        nonlocal obs_ref, chat_history
        obs_ref[0]    = env.reset()
        chat_history  = []
        stats.update({"total_reward": 0.0, "steps": 0, "episodes": 0})
        assistant.__init__(user_name="Arun", speak_aloud=True)
        return ([], "Reset ✅", "", "", "", "", "")

    def inject_scenario(scenario: str, listen_on, gesture_on, sos_on):
        """Preset scenarios for demo purposes."""
        text_map = {
            "🎙️  Wake Word"       : "hey chitti",
            "💬  Normal Chat"     : "what's the weather today",
            "🆘  Emergency Voice" : "help help please danger",
            "✋  SOS Gesture"     : "",       # gesture injected by env
            "🔔  Alarming Audio"  : "glass breaking",
            "🎲  Random"          : "",
        }
        text = text_map.get(scenario, "")

        # For gesture scenario, force gesture count
        if scenario == "✋  SOS Gesture":
            env.sos_gesture_count             = 3
            env._current_gesture              = "sos_palm"
            env.gesture_handler.confirm_count = 3
            env._current_gesture_sos          = 1.0

        return chitti_step(text, listen_on, gesture_on, sos_on)

    # ── Build Gradio UI ────────────────────────────────────────────
    CSS = """
    .sos-btn { background: #e53e3e !important; color: white !important;
               font-size: 1.4em !important; font-weight: bold !important;
               border-radius: 12px !important; padding: 1em 2em !important;
               box-shadow: 0 4px 20px rgba(229,62,62,0.5) !important;
               animation: pulse 1.2s infinite; }
    @keyframes pulse {
      0%,100% { box-shadow: 0 0 0 0 rgba(229,62,62,0.6); }
      50%      { box-shadow: 0 0 0 14px rgba(229,62,62,0); }
    }
    .agent-box { font-family: monospace; }
    """

    with gr.Blocks(
        title="🤖 Chitti — Voice Assistant",
    ) as demo:

        # ── Header ─────────────────────────────────────────────
        gr.Markdown(
            """
            # 🤖 Project Chitti  —  Interactive RL Voice Assistant
            ### Wake word · Natural conversation · Emergency detection · Hand-gesture SOS

            > Type in the box below (or enable a real mic) and press **Send**.
            > Use the **Scenario** dropdown to simulate specific events instantly.
            """
        )

        # ── Permission toggles ──────────────────────────────────
        with gr.Row():
            listen_cb  = gr.Checkbox(value=True,  label="🎙️  listening_enabled")
            gesture_cb = gr.Checkbox(value=True,  label="✋  gesture_enabled")
            sos_cb     = gr.Checkbox(value=True,  label="🆘  sos_enabled")

        # ── Scenario selector ───────────────────────────────────
        with gr.Row():
            scenario_dd = gr.Dropdown(
                choices=["🎙️  Wake Word", "💬  Normal Chat",
                         "🆘  Emergency Voice", "✋  SOS Gesture",
                         "🔔  Alarming Audio", "🎲  Random"],
                value="🎲  Random",
                label="📡 Simulate Scenario",
                scale=3,
            )
            scenario_btn = gr.Button("▶ Run Scenario", scale=1,
                                     variant="secondary")

        # ── Tabs ───────────────────────────────────────────────
        with gr.Tabs():

            # Tab 1: Conversation
            with gr.Tab("💬 Conversation"):
                chatbot = gr.Chatbot(
                    label="Chat with Chitti",
                    height=380,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder='Type a message or say "Hey Chitti" ...',
                        label="Your Message",
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("Send 📨", variant="primary", scale=1)

            # Tab 2: Emergency / SOS
            with gr.Tab("🚨 Emergency"):
                sos_panel  = gr.Markdown("✅  No emergency detected.")
                sos_button = gr.Button(
                    "🆘  CONFIRM SOS — CALL 112",
                    visible=False,
                    elem_classes=["sos-btn"],
                )

            # Tab 3: Agent decisions
            with gr.Tab("🧠 Agent"):
                with gr.Row():
                    obs_panel   = gr.Markdown("*(waiting for first step)*")
                    agent_panel = gr.Markdown("*(waiting for first step)*",
                                              elem_classes=["agent-box"])

            # Tab 4: Logs
            with gr.Tab("📋 Logs"):
                logs_box = gr.Textbox(
                    label="Assistant Activity Log",
                    lines=18,
                    interactive=False,
                )

        reset_btn = gr.Button("🔄 Reset Session", variant="stop")

        # ── Wire up events ─────────────────────────────────────

        # All step outputs in one list
        step_outputs = [chatbot, obs_panel, agent_panel,
                        sos_panel, sos_button, logs_box, msg_input]

        send_btn.click(
            fn=chitti_step,
            inputs=[msg_input, listen_cb, gesture_cb, sos_cb],
            outputs=step_outputs,
        )
        msg_input.submit(
            fn=chitti_step,
            inputs=[msg_input, listen_cb, gesture_cb, sos_cb],
            outputs=step_outputs,
        )
        scenario_btn.click(
            fn=inject_scenario,
            inputs=[scenario_dd, listen_cb, gesture_cb, sos_cb],
            outputs=step_outputs,
        )

        # SOS button click → force emergency call action
        def confirm_sos(listen_on, gesture_on, sos_on):
            return chitti_step("call 112 emergency", listen_on,
                               gesture_on, sos_on)

        sos_button.click(
            fn=confirm_sos,
            inputs=[listen_cb, gesture_cb, sos_cb],
            outputs=step_outputs,
        )

        reset_btn.click(
            fn=reset_session,
            inputs=[],
            outputs=step_outputs,
        )

        gr.Markdown(
            """
            ---
            **Tips:**
            - Say `hey chitti` to activate the assistant
            - Say `help`, `danger`, `scary`, or `stop` to trigger emergency detection
            - Use the **Scenario** dropdown for instant demos
            - Toggle permissions to see how the RL agent adapts
            """
        )

    demo.launch(share=True, server_name="127.0.0.1", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="red"), css=CSS)


# ══════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING & ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Project Chitti v2")
    p.add_argument("--gradio",    action="store_true",
                   help="Launch Gradio interface")
    p.add_argument("--real-mic",  action="store_true",
                   help="Use real microphone (SpeechRecognition)")
    p.add_argument("--real-cam",  action="store_true",
                   help="Use real webcam (MediaPipe)")
    p.add_argument("--episodes",  type=int, default=4,
                   help="CLI: number of episodes")
    p.add_argument("--steps",     type=int, default=20,
                   help="CLI: steps per episode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.gradio:
        launch_gradio(
            use_real_mic=args.real_mic,
            use_real_cam=args.real_cam,
        )
    else:
        run_cli(
            n_episodes=args.episodes,
            n_steps=args.steps,
            use_real_mic=args.real_mic,
            use_real_cam=args.real_cam,
        )