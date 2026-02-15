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

The project has recently undergone significant architectural upgrades to enhance "embodied intelligence":

- **Natural Embodiment**: Introduced **Natural Turn-Level Gestures** and **Speech-Synced Body Sway**. The robot no longer stands still while talking; it moves fluidly to match the rhythm and sentiment of the conversation.
- **Dynamic Capability Discovery**: A new **Capability Registry** automatically scans the Reachy daemon for recorded expressions and dances. This allows the AI to discover and use new robot behaviors without code changes.
- **Context-Aware Triggers**: Added support for **Cue Word Gestures**. The robot can now trigger specific movements based on explicit words or phrases detected in the live transcript.
- **Enhanced Perception**: Optimized MediaPipe tracking data formats and resolved OpenClaw Gateway CORS issues for a more stable real-time vision loop.

---

## ✨ Features

- **👁️ Intelligent Eye Contact**: Real-time face tracking (MediaPipe/YOLO) at 25Hz ensures the robot is always engaged with the user.
- **🎭 Expressive Gestures**: Automatic gestures synced to voice output, including "exaggerated macro gestures" for emphasis.
- **🧠 OpenClaw Integration**: Full access to OpenClaw tools (calendar, smart home, web search) delivered through a physical persona.
- **💃 Emotion Engine**: Play back any recorded emotion or dance from the library, triggered dynamically by the AI.
- **🎤 Low-Latency Voice**: Powered by OpenAI Realtime API for natural, human-like response times.
- **🖥️ Simulator-First**: Full support for MuJoCo simulation, allowing development without physical hardware.

---

## 🏗️ Architecture

ClawBody acts as the orchestrator between three major systems:
1. **OpenAI Realtime API**: Handles the audio stream and generates low-latency responses.
2. **OpenClaw Gateway**: The "brain" that provides personality, memory, and tool-calling capabilities.
3. **Reachy Mini SDK/Daemon**: The "nervous system" controlling the robot's servos and sensors.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
- [OpenClaw Gateway](https://github.com/openclaw/openclaw)
- OpenAI API Key (Realtime API access)

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
clawbody --gradio
```

### Option B: Installation on Physical Robot

```bash
# Connect to your robot
ssh pollen@reachy-mini.local

# Clone and install in the app environment
git clone https://github.com/dAAAb/clawbody
cd clawbody
/venvs/apps_venv/bin/pip install -e .

# Run directly on hardware
clawbody
```

---

## ⚙️ Configuration

Your `.env` file controls the core integration:
- `OPENAI_API_KEY`: Required for voice processing.
- `OPENCLAW_GATEWAY_URL`: Points to your local or remote OpenClaw instance.
- `OPENCLAW_TOKEN`: Authorization for the gateway.
- `ENABLE_FACE_TRACKING`: Set to `true` (default) for eye contact.

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
