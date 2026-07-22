---
title: ClawBody Local
emoji: 🦞
colorFrom: red
colorTo: purple
sdk: static
pinned: false
short_description: Asmo's local OpenClaw body for Reachy Mini
tags:
 - reachy_mini
 - reachy_mini_python_app
 - openclaw
 - clawson
 - embodied-ai
 - ai-assistant
 - voice-assistant
 - robotics
 - local-speech
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

# 🦞🤖 ClawBody Local

See [EMOTIONS.md](EMOTIONS.md) for the complete 81-move Reachy emotion map,
semantic families, aliases, and selection behavior.

**Give your OpenClaw AI agent a physical robot body!**

ClawBody Local is Asmo-owned infrastructure that combines OpenClaw's AI intelligence with Reachy Mini's expressive robot body. Local STT transcribes the microphone, OpenClaw produces the actual response, and Chatterbox or ElevenLabs gives it a voice.

![Reachy Mini Dance](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app/resolve/main/docs/assets/reachy_mini_dance.gif)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 👁️ NEW: Face Tracking & Eye Contact

**The robot looks at you when you speak!**

ClawBody uses Reachy Mini 1.9's daemon-side face tracking and microphone-array
direction-of-arrival support, avoiding a duplicate detector on the OpenClaw host:

- **Automatic Face Detection**: Runs in Reachy's daemon and blends with expressive movement
- **Smooth Head Tracking**: Robot smoothly follows your face as you move
- **Sound Tracking**: Turns conservatively toward speech when no face is visible
- **Visual Questions**: Attaches a current camera frame to OpenClaw when asked what it sees

```bash
# Face tracking is enabled by default
clawbody

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
reachy-mini-daemon --sim

# In another terminal, run ClawBody
clawbody --gradio
```

> 🍎 **Mac Users**: Use `mjpython -m reachy_mini.daemon.app.main --sim` instead.

---

## ✨ Features

- **👁️ Face Tracking**: Robot tracks your face and maintains eye contact during conversation
- **🎤 Local Voice Pipeline**: Always-hot STT and pluggable Chatterbox/ElevenLabs speech
- **🧠 OpenClaw Intelligence**: Your responses come from OpenClaw with full tool access
- **👀 Vision**: See through the robot's camera and describe the environment
- **💃 Expressive Movements**: Natural head movements, emotions, dances, and audio-driven wobble
- **🦞 Clawson Embodied**: Your friendly space lobster AI assistant, now with a body!
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
│  │       Provider-based Voice Pipeline         │            │  │
│  │  • Local STT + local VAD                    │            │  │
│  │  • Chatterbox / ElevenLabs TTS             ─┘            │  │
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
- An OpenAI-compatible local transcription endpoint
- A local Chatterbox endpoint or ElevenLabs API credentials

## 🚀 Installation

### Quick Start with Simulator

```bash
# Clone ClawBody
git clone https://github.com/tomrikert/clawbody
cd clawbody

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install ClawBody + simulator support + face tracking
pip install -e ".[mediapipe_vision]"
pip install "reachy-mini[mujoco]"

# Or for more accurate face tracking (requires more resources)
# pip install -e ".[yolo_vision]"

# Configure (see Configuration section)
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
git clone https://github.com/tomrikert/clawbody
cd clawbody

# Install in the apps virtual environment
/venvs/apps_venv/bin/pip install -e .
```

### Native Reachy App configuration

The native app reads deployment environment variables at startup. A packaged
`.env` file is optional for development only; environment variables supplied by
the app manager take precedence. Configure at least:

```text
OPENCLAW_GATEWAY_URL=ws://your-openclaw-gateway:18789
OPENCLAW_TOKEN=your-gateway-token
OPENCLAW_AGENT_ID=main
OPENCLAW_SESSION_KEY=reachy
STT_BASE_URL=http://your-speech-host:8890/v1
STT_MODEL=distil-large-v3
CHATTERBOX_URL=http://your-speech-host:8890/v1/audio/speech
CHATTERBOX_VOICE=asmo
ENABLE_WAKE_PHRASE=true
WAKE_PHRASE=Hey Asmo
```

The native app uses the SDK-owned robot connection and targets the local Reachy
daemon for camera tracking. It does not require a laptop-side console or a
second OpenClaw agent. Keep gateway, STT, TTS, and optional ElevenLabs secrets
in the app manager's secret/environment store; never commit them to `.env`, a
Space, or the repository.

## ⚙️ Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:

```bash
# Required local STT
STT_BASE_URL=http://speech-host.local:8890/v1
STT_MODEL=distil-large-v3

# OpenClaw Gateway (required for AI responses)
OPENCLAW_GATEWAY_URL=http://localhost:18789  # or your host IP
OPENCLAW_TOKEN=your-gateway-token
OPENCLAW_AGENT_ID=main

# Local Chatterbox voice profile
TTS_PROVIDER=chatterbox
CHATTERBOX_URL=http://speech-host.local:8890/v1/audio/speech
CHATTERBOX_VOICE=default

# Optional - Face tracking (enabled by default)
ENABLE_FACE_TRACKING=true
HEAD_TRACKER_TYPE=mediapipe  # or "yolo" for more accuracy

# Optional - require commands to begin with "Hey Asmo"
ENABLE_WAKE_PHRASE=true
WAKE_PHRASE=Hey Asmo
```

Wake phrase matching is case-insensitive and tolerates STT punctuation such as
`Hey, Asmo`. It must occur at the start of each utterance. The phrase is removed
before the command is sent to OpenClaw; saying only the phrase is treated as a
greeting. Leave `ENABLE_WAKE_PHRASE=false` for the original always-responsive
behavior.

## 🎮 Usage

### With Simulator

```bash
# Terminal 1: Start simulator
reachy-mini-daemon --sim

# Terminal 2: Run ClawBody with web UI (recommended for simulator)
clawbody --gradio
```

The simulator opens a 3D window where you can watch the robot move. The Gradio web UI at http://localhost:7860 lets you interact via your browser's microphone.

### With Physical Robot

```bash
# Basic usage
clawbody

# With debug logging
clawbody --debug

# With specific robot
clawbody --robot-name my-reachy
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--debug` | Enable debug logging |
| `--gradio` | Launch web UI instead of console mode |
| `--robot-name NAME` | Specify robot name for connection |
| `--gateway-url URL` | OpenClaw gateway URL |
| `--no-camera` | Disable camera functionality |
| `--no-openclaw` | Disable OpenClaw integration |
| `--head-tracker TYPE` | Face tracker: `mediapipe` (lighter) or `yolo` (more accurate) |
| `--no-face-tracking` | Disable face tracking |

## 🛠️ Robot Capabilities

ClawBody gives Clawson these physical abilities:

| Capability | Description |
|------------|-------------|
| **Face Tracking** | Automatically tracks and looks at people during conversation |
| **Look** | Move head to look in directions (left, right, up, down) |
| **See** | Capture images through the robot's camera |
| **Dance** | Perform expressive dance animations |
| **Emotions** | Express emotions through movement (happy, curious, thinking, etc.) |
| **Speak** | Voice output through the robot's speaker |
| **Listen** | Hear through the robot's microphone |

## 🖥️ Simulator Features

When running with the simulator:

- **3D Visualization**: Watch Clawson's movements in real-time
- **Scene Options**: Use `--scene minimal` to add objects (apple, duck, croissant)
- **Full SDK Compatibility**: The simulator behaves exactly like a real robot
- **Dashboard Access**: Visit http://localhost:8000 to see the robot dashboard

```bash
# Start simulator with objects on a table
reachy-mini-daemon --sim --scene minimal
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

ClawBody builds on:

- [Pollen Robotics](https://www.pollen-robotics.com/) - Reachy Mini robot, SDK, and simulator
- [OpenClaw](https://github.com/openclaw/openclaw) - AI assistant framework (Clawson!)
- [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) - local speech synthesis
- [MuJoCo](https://mujoco.org/) - Physics simulation engine
- [pollen-robotics/reachy_mini_conversation_app](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app) - Movement and audio systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

- **This project**: [GitHub Issues](https://github.com/tomrikert/clawbody/issues)
- **OpenClaw Skills**: Submit ClawBody as a skill to [ClawHub](https://docs.openclaw.ai/tools/clawhub)
- **Reachy Mini Apps**: Submit to [Pollen Robotics](https://github.com/pollen-robotics)
