# Reachy Mini Robot Body

Give your OpenClaw agent a physical presence with Reachy Mini robot.

## Description

This skill enables OpenClaw to embody a Reachy Mini robot, allowing your AI assistant to:

- **See**: View the world through the robot's camera
- **Hear**: Listen to conversations via the robot's microphone  
- **Speak**: Respond with natural voice through the robot's speaker
- **Move**: Express emotions and reactions through expressive head movements

Using OpenAI's Realtime API, the robot responds with sub-second latency for natural conversation flow.

## Requirements

### Hardware
- [Reachy Mini](https://www.hf.co/reachy-mini/) robot (Wireless or Lite version)
- The robot must be powered on and reachable on the network

### Software
- Python 3.11+
- OpenAI API key with Realtime API access
- OpenClaw gateway running (for extended capabilities)

## Installation

```bash
# Install the package
pip install reachy-mini-openclaw

# Or from source
git clone https://github.com/yourusername/reachy_mini_openclaw.git
cd reachy_mini_openclaw
pip install -e .
```

## Configuration

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-key-here
OPENCLAW_GATEWAY_URL=http://localhost:18789
```

## Usage

### Console Mode
```bash
reachy-openclaw
```

### Web UI Mode
```bash
reachy-openclaw --gradio
```

### As Reachy Mini App
Install from the robot's dashboard app store.

## Features

### Voice Conversation
Real-time voice interaction using OpenAI's Realtime API for natural, low-latency conversation.

### Expressive Movements
The robot automatically:
- Nods and moves while speaking (audio-driven wobble)
- Looks toward speakers
- Expresses emotions through head movements
- Performs dances when appropriate

### Vision
Ask the robot to describe what it sees:
> "What do you see in front of you?"

### OpenClaw Integration
Full access to OpenClaw's tool ecosystem:
> "What's the weather like today?"
> "Add a reminder for tomorrow"
> "Turn on the living room lights"

## Links

- [GitHub Repository](https://github.com/yourusername/reachy_mini_openclaw)
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
- [OpenClaw Documentation](https://docs.openclaw.ai)

## Author

Tom

## License

Apache 2.0
