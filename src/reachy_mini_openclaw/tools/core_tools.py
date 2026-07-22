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


async def _analyze_image_with_openai(frame: np.ndarray, prompt: str) -> Optional[str]:
    """Analyze an image using OpenAI's Chat Completions API (gpt-4o-mini).

    This provides reliable cloud-based vision analysis using the same
    OPENAI_API_KEY already configured for the Realtime API.

    Args:
        frame: BGR numpy array from the camera
        prompt: The question/prompt to ask about the image

    Returns:
        Description string, or None if the call fails
    """
    try:
        import cv2
        from openai import AsyncOpenAI
        from reachy_mini_openclaw.config import config

        api_key = config.OPENAI_API_KEY
        if not api_key:
            logger.warning("No OPENAI_API_KEY for vision analysis")
            return None

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(buffer).decode('utf-8')

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
        )

        text = response.choices[0].message.content
        if text:
            return text.strip()
        return None

    except Exception as e:
        logger.error("OpenAI vision analysis failed: %s", e)
        return None


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
    motion_catalog: Optional[Any] = None


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
        "name": "play_dance",
        "description": "Play an exact move from Reachy's dynamically discovered dance catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "dance_name": {
                    "type": "string",
                    "description": "Exact dance catalog name"
                }
            },
            "required": ["dance_name"]
        }
    },
    {
        "type": "function",
        "name": "play_emotion",
        "description": "Play an emotion by exact move name, semantic family, or common alias.",
        "parameters": {
            "type": "object",
            "properties": {
                "emotion_name": {
                    "type": "string",
                    "description": "Exact move (for example proud3), family (proud), or alias (victory)"
                }
            },
            "required": ["emotion_name"]
        }
    },
    {
        "type": "function",
        "name": "list_moves",
        "description": "List or search the live Reachy emotion and dance catalogs.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["emotion", "dance"]},
                "query": {"type": "string", "description": "Optional case-insensitive name fragment"},
            },
            "required": [],
        },
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
        "play_dance": _handle_dance,
        "play_emotion": _handle_emotion,
        "list_moves": _handle_list_moves,
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
    
    Priority order for vision analysis:
    1. Local SmolVLM2 (on-device, no network latency)
    2. OpenAI Vision API (gpt-4o-mini, reliable cloud vision)
    3. OpenClaw bridge (text-only fallback)
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
        
        vision_prompt = (
            "Describe what you see in this image. Be specific about people, "
            "objects, and the environment. Keep it concise (2-3 sentences). "
            "Speak as if you are looking through your own eyes."
        )
        
        # Option 1: Use local vision processor (SmolVLM2) if available
        if deps.vision_manager is not None:
            logger.info("Using local vision processor (SmolVLM2)...")
            description = deps.vision_manager.process_now(vision_prompt)
            if description and not description.startswith(("Vision", "Failed", "Error", "GPU", "No camera")):
                logger.info("Local vision response: %s", description[:100])
                return {
                    "status": "success",
                    "description": description,
                    "source": "local_vision"
                }
            else:
                logger.warning("Local vision failed: %s", description)
        
        # Option 2: Use OpenAI Vision API (gpt-4o-mini) for image analysis
        logger.info("Using OpenAI Vision API (gpt-4o-mini) for image analysis...")
        openai_description = await _analyze_image_with_openai(frame, vision_prompt)
        if openai_description:
            logger.info("OpenAI vision response: %s", openai_description[:100])
            return {
                "status": "success",
                "description": openai_description,
                "source": "openai_vision"
            }
        
        # Option 3: Fall back to OpenClaw for vision analysis (text-only, limited)
        if deps.openclaw_bridge is not None and deps.openclaw_bridge.is_connected:
            logger.info("Using OpenClaw for vision analysis (text-only fallback)...")
            import cv2
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_image = base64.b64encode(buffer).decode('utf-8')
            
            response = await deps.openclaw_bridge.chat(
                vision_prompt,
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
        
        # Fallback if nothing worked
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
        # A detector can live in this process or in Reachy's daemon.
        if deps.camera_worker.head_tracker is None and not deps.camera_worker.daemon_tracking:
            return {"error": "Face tracking not available - no head tracker initialized"}
        
        deps.camera_worker.set_head_tracking_enabled(enabled)
        return {"status": "success", "face_tracking": enabled}
    except Exception as e:
        return {"error": str(e)}


async def _handle_dance(args: dict, deps: ToolDependencies) -> dict:
    """Handle dance tool."""
    dance_name = args.get("dance_name", "")
    try:
        if deps.motion_catalog is None:
            return {"error": "Motion catalog is unavailable"}
        deps.movement_manager.queue_move(deps.motion_catalog.get("dance", dance_name))
        return {"status": "success", "dance": dance_name}
    except Exception as e:
        return {"error": str(e)}


async def _handle_emotion(args: dict, deps: ToolDependencies) -> dict:
    """Handle emotion expression."""
    emotion_name = args.get("emotion_name", "")
    try:
        if deps.motion_catalog is None:
            return {"error": "Motion catalog is unavailable"}
        resolved = deps.motion_catalog.resolve_emotion(emotion_name)
        if resolved is None:
            return {"error": f"Unknown emotion intent: {emotion_name}"}
        deps.movement_manager.queue_move(deps.motion_catalog.get("emotion", resolved))
        return {"status": "success", "emotion": resolved, "requested": emotion_name}
    except Exception as e:
        return {"error": str(e)}


async def _handle_list_moves(args: dict, deps: ToolDependencies) -> dict:
    if deps.motion_catalog is None:
        return {"error": "Motion catalog is unavailable"}
    kind = args.get("kind")
    query = args.get("query", "")
    if query:
        return {"status": "success", "moves": deps.motion_catalog.search(query, kind)}
    if kind:
        result = {"status": "success", "moves": {kind: deps.motion_catalog.list(kind)}}
        if kind == "emotion":
            result["emotion_families"] = deps.motion_catalog.emotion_map()
            result["unmapped_emotions"] = deps.motion_catalog.unmapped_emotions()
        return result
    return {
        "status": "success",
        "moves": {
            "emotion": deps.motion_catalog.list("emotion"),
            "dance": deps.motion_catalog.list("dance"),
        },
        "emotion_families": deps.motion_catalog.emotion_map(),
        "unmapped_emotions": deps.motion_catalog.unmapped_emotions(),
    }


async def _handle_stop_moves(args: dict, deps: ToolDependencies) -> dict:
    """Stop all movements."""
    deps.movement_manager.clear_move_queue()
    return {"status": "success", "message": "All movements stopped"}


async def _handle_idle(args: dict, deps: ToolDependencies) -> dict:
    """Do nothing - explicitly stay idle."""
    return {"status": "success", "message": "Staying idle"}
