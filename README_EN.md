# 🦞🤖 ClawBody (Optimized)

**Give your OpenClaw AI agent a physical robot body!**

ClawBody combines OpenClaw's AI intelligence with Reachy Mini's expressive robot body, using OpenAI's Realtime API for ultra-responsive voice conversation. Your OpenClaw assistant (Clawson) can now see, hear, speak, and move in the physical world.

![Reachy Mini Dance](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app/resolve/main/docs/assets/reachy_mini_dance.gif)

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

ClawBody includes real-time face tracking that makes conversations feel natural and engaging:

- **Automatic Face Detection**: Uses MediaPipe or YOLO at 25Hz.
- **Smooth Head Tracking**: The robot smoothly follows your face movements.
- **Natural Eye Contact**: Maintains engagement throughout the interaction.

```bash
# Face tracking is enabled by default
clawbody

# Choose your tracker (MediaPipe is lighter, YOLO is more accurate)
clawbody --head-tracker mediapipe
```

---

## 🎮 No Hardware? No Problem!

**You don't need a physical Reachy Mini robot to use ClawBody!**

ClawBody works seamlessly with the [Reachy Mini Simulator](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started), a MuJoCo-based physics simulation.

```bash
# Install simulator support
pip install "reachy-mini[mujoco]"

# Start the simulator (opens a 3D window)
# Mac users: mjpython -m reachy_mini.daemon.app.main --sim
reachy-mini-daemon --sim

# In another terminal, run ClawBody
clawbody --gradio
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/dAAAb/clawbody
cd clawbody

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install ClawBody and dependencies
pip install -e ".[mediapipe_vision]"
```

### 2. Configuration

Copy the environment template and add your keys:

```bash
cp .env.example .env
```

Edit `.env`:
- `OPENAI_API_KEY`: Your OpenAI API Key (Realtime API support required)
- `OPENCLAW_GATEWAY_URL`: Your OpenClaw Gateway URL (e.g., `http://localhost:18789`)
- `OPENCLAW_TOKEN`: Your OpenClaw Token

### 3. Execution

```bash
# Recommended for simulator use
clawbody --gradio
```

---

## 🛠️ Robot Capabilities

ClawBody empowers Clawson with the following physical abilities:

| Capability | Description |
|------------|-------------|
| **Face Tracking** | Automatically tracks and looks at humans during conversation. |
| **Vision (See)** | Captures images via the robot's camera and describes the environment. |
| **Emotions** | Performs pre-recorded emotional movements (e.g., curious, happy, thinking). |
| **Gestures** | Natural gestures synced with speech to enhance expression. |
| **Dance** | Executes expressive and fluid dance animations. |

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features or find bugs, please submit a Pull Request or open an Issue.

- **Submit Issues**: Report bugs or suggest new features.
- **Pull Request**: Ensure code style consistency and pass relevant tests.

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built upon these amazing projects:
- [Pollen Robotics](https://www.pollen-robotics.com/) (Reachy Mini)
- [OpenClaw](https://github.com/openclaw/openclaw) (AI Assistant Framework)
- [OpenAI Realtime API](https://openai.com/)
