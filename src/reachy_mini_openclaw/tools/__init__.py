"""Tool definitions for Reachy Mini OpenClaw.

These tools are exposed to the OpenAI Realtime API and allow the assistant
to control the robot and interact with the environment.
"""

from reachy_mini_openclaw.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)

__all__ = [
    "ToolDependencies",
    "get_tool_specs",
    "dispatch_tool_call",
]
