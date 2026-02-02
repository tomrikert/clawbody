---
title: ClawBody
emoji: 🦞
colorFrom: red
colorTo: purple
sdk: static
pinned: false
short_description: Give your OpenClaw AI a physical robot body
tags:
 - reachy_mini
 - reachy_mini_python_app
 - openclaw
 - clawson
 - embodied-ai
---

# 🦞🤖 ClawBody

**Give your OpenClaw AI agent a physical robot body!**

ClawBody combines OpenClaw's AI intelligence with Reachy Mini's expressive robot body, using OpenAI's Realtime API for ultra-responsive voice conversation. Your OpenClaw assistant (Clawson) can now see, hear, speak, and move in the physical world.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

- **🎤 Real-time Voice Conversation**: OpenAI Realtime API for sub-second response latency
- **🧠 OpenClaw Intelligence**: Your responses come from OpenClaw with full tool access
- **👀 Vision**: See through the robot's camera and describe the environment
- **💃 Expressive Movements**: Natural head movements, emotions, dances, and audio-driven wobble
- **🦞 Clawson Embodied**: Your friendly space lobster AI assistant, now with a body!

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Your Voice / Microphone                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Reachy Mini Robot                             │
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

### Hardware
- [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) robot (Wireless or Lite)

### Software
- Python 3.11+
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini) installed
- [OpenClaw](https://github.com/openclaw/openclaw) gateway running
- OpenAI API key with Realtime API access

## 🚀 Installation

### On the Reachy Mini Robot

```bash
# SSH into the robot
ssh pollen@reachy-mini.local

# Clone the repository
git clone https://github.com/yourusername/clawbody.git
cd clawbody

# Install in the apps virtual environment
/venvs/apps_venv/bin/pip install -e .
```

### Using pip (Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/clawbody.git
cd clawbody

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .
```

## ⚙️ Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:

```bash
# Required
OPENAI_API_KEY=sk-...your-key...

# OpenClaw Gateway (required for AI responses)
OPENCLAW_GATEWAY_URL=http://your-host-ip:18790
OPENCLAW_TOKEN=your-gateway-token
OPENCLAW_AGENT_ID=main

# Optional - Customize voice
OPENAI_VOICE=cedar
```

## 🎮 Usage

### Console Mode

```bash
# Basic usage
clawbody

# With debug logging
clawbody --debug

# With specific robot
clawbody --robot-name my-reachy
```

### Web UI Mode

```bash
# Launch Gradio interface
clawbody --gradio
```

Then open http://localhost:7860 in your browser.

### As a Reachy Mini App

ClawBody registers as a Reachy Mini App, so you can launch it from the robot's dashboard after installation.

### CLI Options

| Option | Description |
|--------|-------------|
| `--debug` | Enable debug logging |
| `--gradio` | Launch web UI instead of console mode |
| `--robot-name NAME` | Specify robot name for connection |
| `--gateway-url URL` | OpenClaw gateway URL |
| `--no-camera` | Disable camera functionality |
| `--no-openclaw` | Disable OpenClaw integration |

## 🛠️ Robot Capabilities

ClawBody gives Clawson these physical abilities:

| Capability | Description |
|------------|-------------|
| **Look** | Move head to look in directions (left, right, up, down) |
| **See** | Capture images through the robot's camera |
| **Dance** | Perform expressive dance animations |
| **Emotions** | Express emotions through movement (happy, curious, thinking, etc.) |
| **Speak** | Voice output through the robot's speaker |
| **Listen** | Hear through the robot's microphone |

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

ClawBody builds on:

- [Pollen Robotics](https://www.pollen-robotics.com/) - Reachy Mini robot and SDK
- [OpenClaw](https://github.com/openclaw/openclaw) - AI assistant framework (Clawson!)
- [OpenAI](https://openai.com/) - Realtime API for voice I/O
- [pollen-robotics/reachy_mini_conversation_app](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app) - Movement and audio systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

- **This project**: [GitHub Issues](https://github.com/yourusername/clawbody/issues)
- **OpenClaw Skills**: Submit ClawBody as a skill to [ClawHub](https://docs.openclaw.ai/tools/clawhub)
- **Reachy Mini Apps**: Submit to [Pollen Robotics](https://github.com/pollen-robotics)
