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
        "description": "Perform a dance animation. Use this to express joy, celebrate, or entertain.",
        "parameters": {
            "type": "object",
            "properties": {
                "dance_name": {
                    "type": "string",
                    "enum": ["happy", "excited", "wave", "nod", "shake", "bounce"],
                    "description": "The dance to perform"
                }
            },
            "required": ["dance_name"]
        }
    },
    {
        "type": "function",
        "name": "emotion",
        "description": "Express an emotion through movement. Use this to show reactions and feelings.",
        "parameters": {
            "type": "object",
            "properties": {
                "emotion_name": {
                    "type": "string",
                    "enum": ["happy", "sad", "surprised", "curious", "thinking", "confused", "excited"],
                    "description": "The emotion to express"
                }
            },
            "required": ["emotion_name"]
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
    """Handle dance tool."""
    dance_name = args.get("dance_name", "happy")
    
    try:
        # Try to use dance library if available
        from reachy_mini_dances_library import dances
        
        if hasattr(dances, dance_name):
            dance_class = getattr(dances, dance_name)
            dance_move = dance_class()
            deps.movement_manager.queue_move(dance_move)
            return {"status": "success", "dance": dance_name}
        else:
            # Fallback to simple head movement
            return await _handle_emotion({"emotion_name": dance_name}, deps)
    except ImportError:
        # No dance library, use emotion as fallback
        return await _handle_emotion({"emotion_name": dance_name}, deps)
    except Exception as e:
        return {"error": str(e)}


async def _handle_emotion(args: dict, deps: ToolDependencies) -> dict:
    """Handle emotion expression."""
    from reachy_mini_openclaw.moves import HeadLookMove
    
    emotion_name = args.get("emotion_name", "happy")
    
    # Map emotions to simple head movements
    emotion_sequences = {
        "happy": ["up", "front"],
        "sad": ["down"],
        "surprised": ["up", "front"],
        "curious": ["right", "left", "front"],
        "thinking": ["up", "left"],
        "confused": ["left", "right", "front"],
        "excited": ["up", "down", "up", "front"],
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
                duration=0.5,
            )
            deps.movement_manager.queue_move(move)
        
        return {"status": "success", "emotion": emotion_name}
    except Exception as e:
        return {"error": str(e)}


async def _handle_stop_moves(args: dict, deps: ToolDependencies) -> dict:
    """Stop all movements."""
    deps.movement_manager.clear_move_queue()
    return {"status": "success", "message": "All movements stopped"}


async def _handle_idle(args: dict, deps: ToolDependencies) -> dict:
    """Do nothing - explicitly stay idle."""
    return {"status": "success", "message": "Staying idle"}
