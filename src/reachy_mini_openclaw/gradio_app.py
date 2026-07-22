"""Gradio web UI for Reachy Mini OpenClaw.

This module provides a web interface for:
- Viewing conversation transcripts
- Configuring the assistant personality
- Monitoring robot status
- Manual control options
"""

import asyncio
import logging
import threading
from typing import Optional

import gradio as gr

logger = logging.getLogger(__name__)


def launch_gradio(
    gateway_url: str = "",
    robot_name: Optional[str] = None,
    robot_host: Optional[str] = None,
    robot_port: Optional[int] = None,
    robot_connection_mode: Optional[str] = None,
    enable_camera: bool = True,
    enable_openclaw: bool = True,
    enable_face_tracking: bool = True,
    head_tracker_type: Optional[str] = None,
    share: bool = False,
) -> None:
    """Launch the Gradio web UI.
    
    Args:
        gateway_url: OpenClaw gateway URL
        robot_name: Robot name for connection
        robot_host: Reachy daemon hostname or IP
        robot_port: Reachy daemon TCP port
        robot_connection_mode: Reachy SDK connection mode
        enable_camera: Whether to enable camera
        enable_openclaw: Whether to enable OpenClaw
        enable_face_tracking: Whether to enable face tracking
        head_tracker_type: Head tracker type ('yolo', 'mediapipe', or None)
        share: Whether to create a public URL
    """
    from reachy_mini_openclaw.prompts import get_available_profiles, save_custom_profile
    from reachy_mini_openclaw.config import set_custom_profile, config
    
    # State
    app_instance = None
    app_loop = None
    
    def start_conversation():
        """Start the conversation."""
        nonlocal app_instance, app_loop

        from reachy_mini_openclaw.main import ClawBodyCore
        
        if app_instance is not None:
            return "Already running"
        
        try:
            config.ENABLE_FACE_TRACKING = enable_face_tracking
            if head_tracker_type is not None:
                config.HEAD_TRACKER_TYPE = head_tracker_type
            app_instance = ClawBodyCore(
                gateway_url=gateway_url,
                robot_name=robot_name,
                robot_host=robot_host,
                robot_port=robot_port,
                robot_connection_mode=robot_connection_mode,
                enable_camera=enable_camera,
                enable_openclaw=enable_openclaw,
            )
            
            # Run in background thread
            def run_app(instance):
                nonlocal app_instance, app_loop
                loop = asyncio.new_event_loop()
                app_loop = loop
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(instance.run())
                except asyncio.CancelledError:
                    logger.info("App stopped")
                except Exception as e:
                    logger.error("App error: %s", e)
                finally:
                    loop.run_until_complete(instance.stop())
                    if app_instance is instance:
                        app_instance = None
                    app_loop = None
                    asyncio.set_event_loop(None)
                    loop.close()
            
            thread = threading.Thread(target=run_app, args=(app_instance,), daemon=True)
            thread.start()
            
            return "Started successfully"
        except Exception as e:
            return f"Error: {e}"
    
    def stop_conversation():
        """Stop the conversation."""
        nonlocal app_instance, app_loop
        
        if app_instance is None:
            return "Not running"
        
        try:
            if app_loop is None or not app_loop.is_running():
                return "App is still starting"
            future = asyncio.run_coroutine_threadsafe(app_instance.stop(), app_loop)
            future.result(timeout=10)
            app_instance = None
            return "Stopped"
        except Exception as e:
            return f"Error: {e}"
    
    def apply_profile(profile_name):
        """Apply a personality profile."""
        set_custom_profile(profile_name if profile_name else None)
        return f"Applied profile: {profile_name or 'default'}"
    
    def save_profile(name, instructions):
        """Save a new profile."""
        if save_custom_profile(name, instructions):
            return f"Saved profile: {name}"
        return "Error saving profile"
    
    # Build UI
    with gr.Blocks(title="Reachy Mini OpenClaw") as demo:
        gr.Markdown("""
        # 🤖 Reachy Mini OpenClaw
        
        Give your OpenClaw AI agent a physical presence with Reachy Mini.
        OpenClaw is the brain, with local STT and pluggable speech synthesis.
        """)
        
        with gr.Tab("Conversation"):
            with gr.Row():
                start_btn = gr.Button("▶️ Start", variant="primary")
                stop_btn = gr.Button("⏹️ Stop", variant="secondary")
            
            status_text = gr.Textbox(label="Status", interactive=False)
            
            transcript = gr.Chatbot(label="Conversation", height=400)
            
            start_btn.click(start_conversation, outputs=[status_text])
            stop_btn.click(stop_conversation, outputs=[status_text])
        
        with gr.Tab("Personality"):
            profiles = get_available_profiles()
            profile_dropdown = gr.Dropdown(
                choices=[""] + profiles,
                label="Select Profile",
                value=""
            )
            apply_btn = gr.Button("Apply Profile")
            profile_status = gr.Textbox(label="Status", interactive=False)
            
            apply_btn.click(
                apply_profile,
                inputs=[profile_dropdown],
                outputs=[profile_status]
            )
            
            gr.Markdown("### Create New Profile")
            new_name = gr.Textbox(label="Profile Name")
            new_instructions = gr.Textbox(
                label="Instructions",
                lines=10,
                placeholder="Enter the system prompt for this personality..."
            )
            save_btn = gr.Button("Save Profile")
            save_status = gr.Textbox(label="Save Status", interactive=False)
            
            save_btn.click(
                save_profile,
                inputs=[new_name, new_instructions],
                outputs=[save_status]
            )
        
        with gr.Tab("Settings"):
            gr.Markdown(f"""
            ### Current Configuration
            
            - **OpenClaw Gateway**: {gateway_url}
            - **STT Endpoint**: {config.STT_BASE_URL}
            - **STT Model**: {config.STT_MODEL}
            - **TTS Provider**: {config.TTS_PROVIDER}
            - **TTS Voice**: {config.CHATTERBOX_VOICE if config.TTS_PROVIDER == 'chatterbox' else config.ELEVENLABS_VOICE_ID}
            - **Camera Enabled**: {enable_camera}
            - **OpenClaw Enabled**: {enable_openclaw}
            - **Face Tracking**: {enable_face_tracking}
            - **Head Tracker**: {head_tracker_type or 'auto-detect'}
            
            Edit `.env` file to change these settings.
            """)
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## About Reachy Mini OpenClaw
            
            This application combines:
            
            - **Local speech recognition** through an OpenAI-compatible endpoint
            - **OpenClaw Gateway** as the conversational brain and tool runtime
            - **Chatterbox or ElevenLabs** for speech synthesis
            - **Reachy Mini Robot** for physical embodiment with expressive movements
            
            ### Features
            
            - 🎤 Real-time voice conversation
            - 👀 Camera-based vision
            - 💃 Expressive robot movements
            - 🔧 Tool integration via OpenClaw
            - 🎭 Customizable personalities
            
            ### Links
            
            - [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
            - [OpenClaw](https://github.com/openclaw/openclaw)
            - [Chatterbox](https://github.com/resemble-ai/chatterbox)
            """)
    
    demo.launch(share=share, server_name="0.0.0.0", server_port=7860)
