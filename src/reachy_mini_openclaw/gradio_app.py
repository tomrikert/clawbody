"""Gradio web UI for Reachy Mini OpenClaw.

This module provides a web interface for:
- Viewing conversation transcripts
- Configuring the assistant personality
- Monitoring robot status
- Manual control options
"""

import os
import logging
from typing import Optional

import gradio as gr

logger = logging.getLogger(__name__)


def launch_gradio(
    gateway_url: str = "http://localhost:18789",
    robot_name: Optional[str] = None,
    enable_camera: bool = True,
    enable_openclaw: bool = True,
    share: bool = False,
) -> None:
    """Launch the Gradio web UI.
    
    Args:
        gateway_url: OpenClaw gateway URL
        robot_name: Robot name for connection
        enable_camera: Whether to enable camera
        enable_openclaw: Whether to enable OpenClaw
        share: Whether to create a public URL
    """
    from reachy_mini_openclaw.prompts import get_available_profiles, save_custom_profile
    from reachy_mini_openclaw.config import set_custom_profile, config
    
    # State
    app_instance = None
    
    def start_conversation():
        """Start the conversation."""
        nonlocal app_instance
        
        from reachy_mini_openclaw.main import ReachyOpenClawCore
        import asyncio
        import threading
        
        if app_instance is not None:
            return "Already running"
        
        try:
            app_instance = ReachyOpenClawCore(
                gateway_url=gateway_url,
                robot_name=robot_name,
                enable_camera=enable_camera,
                enable_openclaw=enable_openclaw,
            )
            
            # Run in background thread
            def run_app():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(app_instance.run())
                except Exception as e:
                    logger.error("App error: %s", e)
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_app, daemon=True)
            thread.start()
            
            return "Started successfully"
        except Exception as e:
            return f"Error: {e}"
    
    def stop_conversation():
        """Stop the conversation."""
        nonlocal app_instance
        
        if app_instance is None:
            return "Not running"
        
        try:
            app_instance.stop()
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
        Using OpenAI Realtime API for responsive voice conversation.
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
            - **OpenAI Model**: {config.OPENAI_MODEL}
            - **Voice**: {config.OPENAI_VOICE}
            - **Camera Enabled**: {enable_camera}
            - **OpenClaw Enabled**: {enable_openclaw}
            
            Edit `.env` file to change these settings.
            """)
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## About Reachy Mini OpenClaw
            
            This application combines:
            
            - **OpenAI Realtime API** for ultra-low-latency voice conversation
            - **OpenClaw Gateway** for extended AI capabilities (web, calendar, smart home, etc.)
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
            - [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
            """)
    
    demo.launch(share=share, server_name="0.0.0.0", server_port=7860)
