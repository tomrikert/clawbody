---
title: ClawBody
emoji: 🦞
colorFrom: red
colorTo: purple
sdk: static
pinned: false
short_description: OpenClaw AI with robot body and face tracking
tags:
 - reachy_mini
 - reachy_mini_python_app
 - openclaw
 - clawson
 - embodied-ai
 - ai-assistant
 - voice-assistant
 - robotics
 - openai-realtime
 - conversational-ai
 - physical-ai
 - robot-body
 - speech-to-speech
 - multimodal
 - vision
 - expressive-robot
 - simulation
 - mujoco
 - face-tracking
 - face-detection
 - eye-contact
 - human-robot-interaction
---

# 🦞🤖 ClawBody

**Give your OpenClaw AI agent an expressive, embodied physical body!**

[繁體中文版 (Traditional Chinese)](README_zh-TW.md)

ClawBody bridges the gap between high-level AI intelligence (OpenClaw) and low-level robotic control (Reachy Mini). By leveraging OpenAI's Realtime API, it creates an ultra-low latency, speech-to-speech interaction loop where your AI assistant, Clawson, can see, hear, and express emotions physically.

![Reachy Mini Dance](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app/resolve/main/docs/assets/reachy_mini_dance.gif)

---

## 🚀 Key Improvements (Feb 2026)

- **Natural Embodiment**: Introduced **Natural Turn-Level Gestures** and **Speech-Synced Body Sway**.
- **Dynamic Capability Discovery**: A new **Capability Registry** automatically scans for recorded expressions and dances.
- **Context-Aware Triggers**: Added support for **Cue Word Gestures** from live transcripts.
- **Enhanced Perception**: Optimized MediaPipe tracking and resolved OpenClaw Gateway CORS issues.

---

## 🚀 Getting Started

### 💡 Critical Usage Note: Virtual Environment
If you installed ClawBody within a virtual environment, you **must** use the environment's python/bin to run it.

**For local/simulator setup:**
```bash
source .venv/bin/activate
clawbody --gradio
```

**For physical robot setup:**
```bash
/venvs/apps_venv/bin/clawbody --gradio
```

---

### Option A: Installation for Simulator

```bash
git clone https://github.com/dAAAb/clawbody
cd clawbody
python -m venv .venv && source .venv/bin/activate
pip install -e ".[mediapipe_vision]"
pip install "reachy-mini[mujoco]"

# Configure your .env
cp .env.example .env

# Terminal 1: Run Simulator
reachy-mini-daemon --sim

# Terminal 2: Start ClawBody
source .venv/bin/activate
clawbody --gradio
```

---

### Option B: Installation on Physical Robot

The Reachy Mini robot comes with a pre-configured application environment at `/venvs/apps_venv/`.

```bash
# Connect to your robot
ssh pollen@reachy-mini.local

# Clone and install into the robot's app environment
git clone https://github.com/dAAAb/clawbody
cd clawbody
/venvs/apps_venv/bin/pip install -e .

# Run directly on hardware (with Gradio enabled)
/venvs/apps_venv/bin/clawbody --gradio
```

---

## 🤖 Automation & Background Service

On a physical Reachy Mini, you can register ClawBody as a managed service using the `reachy-mini-daemon` tool so it starts automatically when the robot boots up. **We recommend enabling `--gradio` for remote management.**

### 1. Register the Application
Run this command from any directory. Note the use of `--args "--gradio"` to enable the web UI:

```bash
/venvs/apps_venv/bin/reachy-mini-daemon app register clawbody --path /home/pollen/clawbody --args "--gradio"
```

### 2. Enable Auto-start on Boot
```bash
/venvs/apps_venv/bin/reachy-mini-daemon app enable clawbody
```

### 3. Management Commands
| Action | Command |
|--------|---------|
| **Start** | `/venvs/apps_venv/bin/reachy-mini-daemon app start clawbody` |
| **Stop** | `/venvs/apps_venv/bin/reachy-mini-daemon app stop clawbody` |
| **Status** | `/venvs/apps_venv/bin/reachy-mini-daemon app list` |
| **Logs** | `/venvs/apps_venv/bin/reachy-mini-daemon app logs clawbody` |

---

## 🛑 Remote Shutdown (No SSH Required)

To use these features, ensure the app was started with the `--gradio` flag (see Automation section).

### 1. Web UI Shutdown (Gradio)
Access the UI at `http://reachy-mini.local:7860`:
- Click the **"🛑 Shutdown App"** button.
- This will completely terminate the background Python process.

### 2. Voice Command Shutdown
You can tell the AI to stop directly:
- **Example**: "Hey Clawbody, please shutdown", "Stop service and rest", "Goodbye".
- The AI will bid you farewell and safely exit the application after a 3-second delay.

---

## ⚙️ Configuration & Remote Deployment

### Connecting to Zeabur / Remote OpenClaw
When connecting to a remote OpenClaw instance (e.g., hosted on Zeabur):

1. **Protocol Matters**: Use `https://` for remote instances.
2. **WebSocket (WSS)**: ClawBody communicates via WebSockets. Ensure your remote deployment handles `wss://` traffic correctly.
3. **CORS/Auth**: Verify `OPENCLAW_TOKEN` and gateway permissions.

Example `.env`:
```bash
OPENCLAW_GATEWAY_URL=https://your-openclaw-on-zeabur.zeabur.app
OPENCLAW_TOKEN=your-secure-token
```

---

## ✨ Features

- **👁️ Intelligent Eye Contact**: Real-time face tracking (MediaPipe/YOLO) at 25Hz.
- **🎭 Expressive Gestures**: Automatic gestures synced to voice output.
- **🧠 OpenClaw Integration**: Full tool-calling capabilities through a physical persona.
- **💃 Emotion Engine**: Dynamic discovery and playback of pre-recorded behaviors.
- **🎤 Low-Latency Voice**: Powered by OpenAI Realtime API.
- **🖥️ Simulator-First**: Full support for MuJoCo simulation.

---

## 🛠️ Robot Capabilities (AI-Accessible)

| Capability | Technical Details |
|------------|-------------------|
| **Natural Gestures** | Turn-level triggers synced to transcript deltas |
| **Emotion Registry** | Dynamic discovery of daemon-recorded expressions |
| **Face Tracking** | PID-controlled head movement using 25Hz vision data |
| **Vision Description** | Captures frames and uses GPT-4o-mini for scene understanding |

---

## 📄 License

This project is licensed under the Apache 2.0 License.

## 🙏 Acknowledgments

Built with ❤️ by the community, leveraging works from [Pollen Robotics](https://www.pollen-robotics.com/), [OpenClaw](https://github.com/openclaw/openclaw), and [OpenAI](https://openai.com/).
