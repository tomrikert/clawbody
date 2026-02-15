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

**Give your OpenClaw AI agent a physical robot body!**

[繁體中文版 (Traditional Chinese)](README_zh-TW.md)

ClawBody combines OpenClaw's AI intelligence with Reachy Mini's expressive robot body, using OpenAI's Realtime API for ultra-responsive voice conversation. Your OpenClaw assistant (Clawson) can now see, hear, speak, and move in the physical world.

![Reachy Mini Dance](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app/resolve/main/docs/assets/reachy_mini_dance.gif)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 🌟 Latest Updates (Last 24 Hours)

- **🎭 Dynamic Emotion & Dance Registry**: Integrated all daemon-recorded expressions and dances, making the robot significantly more expressive.
- **👋 Natural Turn-Level Gestures**: Added automatic physical gestures that trigger during conversation to mimic human-like interaction.
- **🗣️ Speech-Synced Movement**: Gestures are now aligned with speech, featuring new `body_sway` effects for enhanced realism.
- **🎯 Cue Word Triggers**: Specific gestures can now be triggered by explicit keywords in the conversation.
- **🔗 Connection Improvements**: Resolved OpenClaw CORS issues and optimized MediaPipe tracking data formats.

---

## 👁️ Core Feature: Face Tracking & Eye Contact

**The robot looks at you when you speak!**

ClawBody now includes real-time face tracking that makes conversations feel natural and engaging:

- **Automatic Face Detection**: Uses MediaPipe or YOLO to detect faces at 25Hz
- **Smooth Head Tracking**: Robot smoothly follows your face as you move
- **Natural Eye Contact**: Maintains engagement during conversation
- **Graceful Fallback**: Smoothly returns to neutral position when you leave

```bash
# Face tracking is enabled by default
clawbody

# Choose your tracker (MediaPipe is lighter, YOLO is more accurate)
clawbody --head-tracker mediapipe
clawbody --head-tracker yolo

# Disable if needed
clawbody --no-face-tracking
```

---

## 🎮 No Robot? No Problem!

**You don't need a physical Reachy Mini robot to use ClawBody!**

ClawBody works with the [Reachy Mini Simulator](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started), a MuJoCo-based physics simulation that runs on your computer. Watch Clawson move and express emotions on screen while you talk to your OpenClaw agent.

```bash
# Install simulator support
pip install "reachy-mini[mujoco]"

# Start the simulator (opens a 3D window)
# Mac Users: mjpython -m reachy_mini.daemon.app.main --sim
reachy-mini-daemon --sim

# In another terminal, run ClawBody
clawbody --gradio
```

---

## ✨ Features

- **👁️ Face Tracking**: Robot tracks your face and maintains eye contact during conversation
- **🎤 Real-time Voice Conversation**: OpenAI Realtime API for sub-second response latency
- **🧠 OpenClaw Intelligence**: Your responses come from OpenClaw with full tool access
- **👀 Vision**: See through the robot's camera and describe the environment
- **💃 Expressive Movements**: Natural head movements, emotions, dances, and audio-driven wobble
- **🎭 Natural Gestures**: Speech-synced body language and turn-level gestures
- **🖥️ Simulator Support**: Works with or without physical hardware

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Your Voice / Microphone                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Reachy Mini Robot (or Simulator)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Microphone  │  │   Camera    │  │   Movement System       │  │
│  │  (input)    │  │  (vision)   │  │ (head, antennas, body)  │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────▲────────────┘  │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          ▼                ▼                      │
┌─────────────────────────────────────────────────┼───────────────┐
│                      ClawBody                   │               │
│  ┌─────────────────────────────────────────────┼────────────┐  │
│  │         OpenAI Realtime API Handler         │            │  │
│  │  • Speech recognition (Whisper)             │            │  │
│  │  • Text-to-speech (voices)                 ─┘            │  │
│  │  • Audio analysis → head wobble                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              OpenClaw Gateway Bridge                     │  │
│  │  • AI responses from Clawson                            │  │
│  │  • Full OpenClaw tool access                            │  │
│  │  • Conversation memory & context                        │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OpenClaw Gateway                              │
│  • Web browsing  • Calendar  • Smart home  • Memory  • Tools    │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

### Option A: With Physical Robot
- [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) robot (Wireless or Lite)

### Option B: With Simulator (No Hardware Required!)
- Any computer with Python 3.11+
- Install: `pip install "reachy-mini[mujoco]"`
- [Simulation Setup Guide](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started)

### Software (Both Options)
- Python 3.11+
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini) installed
- [OpenClaw](https://github.com/openclaw/openclaw) gateway running
- OpenAI API key with Realtime API access

## 🚀 Installation

### Quick Start with Simulator

```bash
# Clone ClawBody
git clone https://github.com/dAAAb/clawbody
cd clawbody

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install ClawBody + simulator support + face tracking
pip install -e ".[mediapipe_vision]"
pip install "reachy-mini[mujoco]"

# Configure
cp .env.example .env
# Edit .env with your keys

# Terminal 1: Start the simulator
reachy-mini-daemon --sim

# Terminal 2: Run ClawBody
clawbody --gradio
```

### On a Physical Reachy Mini Robot

```bash
# SSH into the robot
ssh pollen@reachy-mini.local

# Clone the repository
git clone https://github.com/dAAAb/clawbody
cd clawbody

# Install in the apps virtual environment
/venvs/apps_venv/bin/pip install -e .
```

## ⚙️ Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:
- `OPENAI_API_KEY`: your OpenAI key
- `OPENCLAW_GATEWAY_URL`: e.g., `http://localhost:18789`
- `OPENCLAW_TOKEN`: your gateway token

## 🛠️ Robot Capabilities

| Capability | Description |
|------------|-------------|
| **Face Tracking** | Automatically tracks and looks at people during conversation |
| **Look** | Move head to look in directions (left, right, up, down) |
| **See** | Capture images through the robot's camera |
| **Dance** | Perform expressive dance animations |
| **Emotions** | Express emotions through movement (happy, curious, thinking, etc.) |
| **Gestures** | Natural speech-synced body language and turn-level gestures |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

ClawBody builds on:
- [Pollen Robotics](https://www.pollen-robotics.com/)
- [OpenClaw](https://github.com/openclaw/openclaw)
- [OpenAI Realtime API](https://openai.com/)
