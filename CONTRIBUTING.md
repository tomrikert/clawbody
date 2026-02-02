# Contributing to Reachy Mini OpenClaw

Thank you for your interest in contributing! This project welcomes contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear title and description
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, robot model)

### Suggesting Features

Feature requests are welcome! Please open an issue with:
- A clear description of the feature
- Use cases and motivation
- Any technical considerations

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Run linting: `ruff check . && ruff format .`
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/reachy_mini_openclaw.git
cd reachy_mini_openclaw

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff check --fix .
ruff format .
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for public functions and classes
- Keep functions focused and small

## Where to Submit Contributions

### This Project
Submit PRs directly to this repository for:
- Bug fixes
- New features
- Documentation improvements
- New personality profiles

### Reachy Mini Ecosystem
- **SDK improvements**: [pollen-robotics/reachy_mini](https://github.com/pollen-robotics/reachy_mini)
- **New dances/emotions**: [reachy_mini_dances_library](https://github.com/pollen-robotics/reachy_mini_dances_library)
- **Apps for the app store**: Submit to [Hugging Face Spaces](https://huggingface.co/spaces)

### OpenClaw Ecosystem
- **New skills**: Submit to [MoltDirectory](https://github.com/neonone123/moltdirectory)
- **Core OpenClaw**: [openclaw/openclaw](https://github.com/openclaw/openclaw)

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
