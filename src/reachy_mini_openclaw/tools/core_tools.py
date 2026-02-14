"""Core tool definitions for the Clawson robot assistant.

These tools allow Clawson (OpenClaw in a robot body) to control 
robot movements and capture images.

Tool Categories:
1. Movement Tools - Control head position, play emotions/dances
2. Vision Tools - Capture and analyze camera images
"""

import json
import logging
import base64
import asyncio
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reachy_mini_openclaw.moves import MovementManager, HeadLookMove
    from reachy_mini_openclaw.audio.head_wobbler import HeadWobbler
    from reachy_mini_openclaw.openclaw_bridge import OpenClawBridge

logger = logging.getLogger(__name__)


@dataclass
class ToolDependencies:
    """Dependencies required by tools.
    
    This dataclass holds references to robot systems that tools need
    to interact with.
    """
    movement_manager: "MovementManager"
    head_wobbler: "HeadWobbler"
    robot: Any  # ReachyMini instance
    camera_worker: Optional[Any] = None
    openclaw_bridge: Optional["OpenClawBridge"] = None
    vision_manager: Optional[Any] = None  # Local vision processor (SmolVLM2)


# Tool specifications in OpenAI format
TOOL_SPECS = [
    {
        "type": "function",
        "name": "look",
        "description": "Move the robot's head to look in a specific direction. Use this to direct attention or emphasize a point.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down", "front"],
                    "description": "The direction to look. 'front' returns to neutral position."
                }
            },
            "required": ["direction"]
        }
    },
    {
        "type": "function",
        "name": "camera",
        "description": "Capture an image from the robot's camera to see what's in front of you. Use this when asked about your surroundings or to identify objects/people.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "face_tracking",
        "description": "Enable or disable face tracking. When enabled, the robot will automatically look at detected faces.",
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "True to enable face tracking, False to disable"
                }
            },
            "required": ["enabled"]
        }
    },
    {
        "type": "function",
        "name": "dance",
        "description": "Perform a dance animation. Accepts any string; available dances depend on installed libraries. Falls back to macro movements if missing.",
        "parameters": {
            "type": "object",
            "properties": {
                "dance_name": {
                    "type": "string",
                    "description": "Dance name (e.g., happy, excited, wave, nod, shake, bounce)."
                }
            },
            "required": ["dance_name"]
        }
    },
    {
        "type": "function",
        "name": "emotion",
        "description": "Express an emotion through movement. Accepts any string; available emotions depend on installed libraries. Falls back to macro movements if missing.",
        "parameters": {
            "type": "object",
            "properties": {
                "emotion_name": {
                    "type": "string",
                    "description": "Emotion name (e.g., happy, sad, surprised, curious, thinking, confused, excited)."
                }
            },
            "required": ["emotion_name"]
        }
    },
    {
        "type": "function",
        "name": "capabilities",
        "description": "List available dances/emotions detected at runtime (and macro fallbacks). Useful for debugging and UIs.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "body_sway",
        "description": "Sway the robot body/base left-right (body_yaw) then return. Useful for expressive emphasis.",
        "parameters": {
            "type": "object",
            "properties": {
                "amplitude_deg": {
                    "type": "number",
                    "description": "Yaw amplitude in degrees (default 12).",
                    "default": 12
                },
                "repeats": {
                    "type": "integer",
                    "description": "Number of left-right cycles (default 1).",
                    "default": 1
                },
                "duration": {
                    "type": "number",
                    "description": "Seconds per half-sway (default 0.6).",
                    "default": 0.6
                }
            },
            "required": []
        }
    },
    {
        "type": "function",
        "name": "stop_moves",
        "description": "Stop all current movements and clear the movement queue.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "idle",
        "description": "Do nothing and remain idle. Use this when you want to stay still.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
]


def get_tool_specs() -> list[dict]:
    """Get the list of tool specifications for OpenAI.
    
    Returns:
        List of tool specification dictionaries
    """
    return TOOL_SPECS


async def dispatch_tool_call(
    tool_name: str,
    arguments_json: str,
    deps: ToolDependencies,
) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate handler.
    
    Args:
        tool_name: Name of the tool to execute
        arguments_json: JSON string of tool arguments
        deps: Tool dependencies
        
    Returns:
        Dictionary with tool result
    """
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON arguments: {arguments_json}"}
    
    handlers = {
        "look": _handle_look,
        "camera": _handle_camera,
        "face_tracking": _handle_face_tracking,
        "dance": _handle_dance,
        "emotion": _handle_emotion,
        "capabilities": _handle_capabilities,
        "body_sway": _handle_body_sway,
        "stop_moves": _handle_stop_moves,
        "idle": _handle_idle,
    }
    
    handler = handlers.get(tool_name)
    if handler is None:
        return {"error": f"Unknown tool: {tool_name}"}
    
    try:
        return await handler(args, deps)
    except Exception as e:
        logger.error("Tool '%s' failed: %s", tool_name, e, exc_info=True)
        return {"error": str(e)}


async def _handle_look(args: dict, deps: ToolDependencies) -> dict:
    """Handle the look tool."""
    from reachy_mini_openclaw.moves import HeadLookMove
    
    direction = args.get("direction", "front")
    
    try:
        # Get current pose for smooth transition
        _, current_ant = deps.robot.get_current_joint_positions()
        current_head = deps.robot.get_current_head_pose()
        
        move = HeadLookMove(
            direction=direction,
            start_pose=current_head,
            start_antennas=tuple(current_ant),
            duration=1.0,
        )
        deps.movement_manager.queue_move(move)
        
        return {"status": "success", "direction": direction}
    except Exception as e:
        return {"error": str(e)}


async def _handle_camera(args: dict, deps: ToolDependencies) -> dict:
    """Handle the camera tool - capture image and get description.
    
    Uses local vision (SmolVLM2) if available, otherwise falls back to OpenClaw.
    """
    logger.info("Camera tool called, camera_worker=%s, vision_manager=%s", 
                deps.camera_worker is not None, deps.vision_manager is not None)
    
    if deps.camera_worker is None:
        logger.warning("Camera worker is None")
        return {"error": "Camera not available"}
    
    try:
        frame = deps.camera_worker.get_latest_frame()
        logger.info("Got frame from camera_worker: %s", frame is not None)
        
        if frame is None:
            # Try getting frame directly from robot as fallback
            logger.info("Trying direct robot camera access...")
            if deps.robot is not None:
                try:
                    frame = deps.robot.media.get_frame()
                    logger.info("Direct frame capture: %s", frame is not None)
                except Exception as e:
                    logger.error("Direct frame capture failed: %s", e)
        
        if frame is None:
            return {"error": "No frame available from camera"}
        
        logger.info("Got frame, shape=%s", frame.shape)
        
        # Option 1: Use local vision processor (SmolVLM2) if available
        if deps.vision_manager is not None:
            logger.info("Using local vision processor (SmolVLM2)...")
            description = deps.vision_manager.process_now(
                "Describe what you see in this image. Be specific about people, objects, and the environment. Keep it concise (2-3 sentences)."
            )
            if description and not description.startswith(("Vision", "Failed", "Error", "GPU", "No camera")):
                logger.info("Local vision response: %s", description[:100])
                return {
                    "status": "success",
                    "description": description,
                    "source": "local_vision"
                }
            else:
                logger.warning("Local vision failed: %s", description)
        
        # Option 2: Fall back to OpenClaw for vision analysis
        if deps.openclaw_bridge is not None and deps.openclaw_bridge.is_connected:
            logger.info("Using OpenClaw for vision analysis...")
            import cv2
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_image = base64.b64encode(buffer).decode('utf-8')
            
            response = await deps.openclaw_bridge.chat(
                "Describe what you see in this image. Be specific about people, objects, and the environment. Keep it concise (2-3 sentences).",
                image_b64=b64_image,
                system_context="You are looking through your robot camera. Describe what you see naturally, as if you're the one looking.",
            )
            if response.content and not response.error:
                logger.info("OpenClaw vision response: %s", response.content[:100])
                return {
                    "status": "success",
                    "description": response.content,
                    "source": "openclaw"
                }
            else:
                logger.warning("OpenClaw vision failed: %s", response.error)
        
        # Fallback if neither is available
        return {
            "status": "partial",
            "description": "I captured an image but couldn't analyze it. No vision processing available."
        }
    except Exception as e:
        logger.error("Camera tool error: %s", e, exc_info=True)
        return {"error": str(e)}


async def _handle_face_tracking(args: dict, deps: ToolDependencies) -> dict:
    """Handle face tracking toggle."""
    enabled = args.get("enabled", False)
    
    if deps.camera_worker is None:
        return {"error": "Camera not available for face tracking"}
    
    try:
        # Check if head tracker is available
        if deps.camera_worker.head_tracker is None:
            return {"error": "Face tracking not available - no head tracker initialized"}
        
        deps.camera_worker.set_head_tracking_enabled(enabled)
        return {"status": "success", "face_tracking": enabled}
    except Exception as e:
        return {"error": str(e)}


async def _handle_dance(args: dict, deps: ToolDependencies) -> dict:
    """Handle dance tool.

    If reachy_mini_dances_library is installed, use its dances.
    Otherwise fall back to macro movements (emotion handler).
    """
    from reachy_mini_openclaw.capabilities.registry import get_dance_factory

    dance_name = args.get("dance_name", "happy")

    try:
        factory = get_dance_factory(dance_name)
        if factory is not None:
            dance_move = factory()
            deps.movement_manager.queue_move(dance_move)
            return {"status": "success", "dance": dance_name, "source": "dance_library"}

        # Fallback to simple head movement macros
        result = await _handle_emotion({"emotion_name": dance_name}, deps)
        result.setdefault("source", "macro_fallback")
        result.setdefault("dance", dance_name)
        return result
    except Exception as e:
        return {"error": str(e)}


async def _handle_emotion(args: dict, deps: ToolDependencies) -> dict:
    """Handle emotion expression.

    Currently implemented as macro head-movement sequences.
    If a future Reachy Mini SDK exposes emotion primitives, this is
    the place to route to them.
    """
    from reachy_mini_openclaw.moves import HeadLookMove

    emotion_name = args.get("emotion_name", "happy")

    # If Reachy Mini daemon recorded emotions are available, prefer them.
    # This unlocks the full expressions library (e.g., sad2/downcast1/oops1/success1...).
    try:
        from reachy_mini_openclaw.capabilities.registry import (
            DEFAULT_RECORDED_EMOTIONS_DATASET,
            play_recorded_move,
        )

        if isinstance(emotion_name, str) and emotion_name:
            if play_recorded_move(DEFAULT_RECORDED_EMOTIONS_DATASET, emotion_name):
                return {"status": "success", "emotion": emotion_name, "source": "recorded_dataset"}
    except Exception:
        pass


    # Map emotions to simple head movements (macro fallback)
    emotion_sequences: dict[str, list[str]] = {
        "happy": ["up", "front"],
        "sad": ["down"],
        "surprised": ["up", "front"],
        "curious": ["right", "left", "front"],
        "thinking": ["up", "left"],
        "confused": ["left", "right", "front"],
        "excited": ["up", "down", "up", "front"],
        # Common aliases / gestures (exaggerated macros)
        # Note: amplitude depends on Reachy safety/limits; we exaggerate by repetition + snappier timing.
        "wave": ["right", "left", "right", "front"],
        "nod": ["down", "up", "down", "up", "front"],
        "shake": ["left", "right", "left", "right", "left", "front"],
        "bounce": ["down", "up", "down", "front"],
    }

    sequence = emotion_sequences.get(emotion_name, ["front"])

    try:
        for direction in sequence:
            _, current_ant = deps.robot.get_current_joint_positions()
            current_head = deps.robot.get_current_head_pose()

            move = HeadLookMove(
                direction=direction,
                start_pose=current_head,
                start_antennas=tuple(current_ant),
                duration=0.32,
            )
            deps.movement_manager.queue_move(move)

        return {
            "status": "success",
            "emotion": emotion_name,
            "source": "macro",
            "known": emotion_name in emotion_sequences,
        }
    except Exception as e:
        return {"error": str(e)}


async def _handle_capabilities(args: dict, deps: ToolDependencies) -> dict:
    """Return a runtime report of available dances/emotions."""
    from reachy_mini_openclaw.capabilities.registry import capabilities_report

    macro_emotions = [
        "happy",
        "sad",
        "surprised",
        "curious",
        "thinking",
        "confused",
        "excited",
        "wave",
        "nod",
        "shake",
        "bounce",
    ]

    report = capabilities_report(macro_emotions=macro_emotions, macro_dances=["wave", "nod", "shake", "bounce"])
    return {
        "status": "success",
        "dances_available": report.dances_available,
        "dance_names": report.dance_names,
        "emotions_available": report.emotions_available,
        "emotion_names": report.emotion_names,
        "notes": report.notes,
    }


async def _handle_body_sway(args: dict, deps: ToolDependencies) -> dict:
    """Sway the robot base/body yaw left-right then return.

    This uses Reachy Mini SDK `goto_target(body_yaw=...)` if available.
    If unsupported, returns an error.
    """
    import numpy as _np

    amp_deg = float(args.get("amplitude_deg", 12) or 12)
    repeats = int(args.get("repeats", 1) or 1)
    duration = float(args.get("duration", 0.6) or 0.6)

    # Clamp to a conservative safe range
    amp_deg = max(3.0, min(25.0, amp_deg))
    repeats = max(1, min(3, repeats))
    duration = max(0.25, min(2.0, duration))

    robot = getattr(deps, "robot", None)
    if robot is None or not hasattr(robot, "goto_target"):
        return {"error": "body_sway not available (robot.goto_target not found)"}

    async def _runner():
        try:
            # Try to preserve current body yaw if readable
            start_yaw = 0.0
            try:
                # Some SDK versions expose current joint positions.
                # If unavailable, we just return to 0.
                _jp = getattr(robot, "get_current_joint_positions", None)
                if callable(_jp):
                    _, _ant = robot.get_current_joint_positions()
                # No reliable body yaw getter found; keep 0.
            except Exception:
                pass

            amp = float(_np.deg2rad(amp_deg))
            for _ in range(repeats):
                robot.goto_target(body_yaw=+amp, duration=duration, method="minjerk")
                await asyncio.sleep(duration)
                robot.goto_target(body_yaw=-amp, duration=duration, method="minjerk")
                await asyncio.sleep(duration)

            robot.goto_target(body_yaw=float(start_yaw), duration=duration, method="minjerk")
        except Exception:
            return

    asyncio.create_task(_runner())
    return {"status": "success", "amplitude_deg": amp_deg, "repeats": repeats}


async def _handle_stop_moves(args: dict, deps: ToolDependencies) -> dict:
    """Stop all movements."""
    deps.movement_manager.clear_move_queue()
    return {"status": "success", "message": "All movements stopped"}


async def _handle_idle(args: dict, deps: ToolDependencies) -> dict:
    """Do nothing - explicitly stay idle."""
    return {"status": "success", "message": "Staying idle"}
