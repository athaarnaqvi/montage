import os
from mcp.client import mcp_call
from agents.scene_composer import scene_composer_node


def lip_sync_node(state):
    """
    Lip Sync node — delegates to Scene Composer.

    Replaces the old per-character video approach that caused:
      - 'str object has no attribute .get' (audio_files passed as dialogue)
      - Separate videos per character instead of one composed scene
      - No timing alignment between audio and visuals

    Scene Composer:
      1. Builds per-scene timeline from actual WAV durations
      2. Renders ONE video per scene with ALL characters in same frame
      3. Active speaker: brightened + golden border + mouth animation
      4. Idle characters: slightly darkened
      5. Subtitles centered at bottom for each dialogue line
      6. Muxes merged audio → audio duration == video duration
    """
    print("\n[Lip Sync → Scene Composer] Building final timeline-driven scenes...")
    return scene_composer_node(state)