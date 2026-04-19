import os
import json
from mcp.client import mcp_call


def voice_synthesis_node(state):
    print("\n[Voice Synthesis] Generating speech for all scenes...")

    scenes = state.get("scenes", [])
    character_db = state.get("character_db", [])

    if not character_db:
        try:
            with open("outputs/character_db.json", "r", encoding="utf-8") as f:
                character_db = json.load(f)
            state["character_db"] = character_db
        except FileNotFoundError:
            print("[Voice Synthesis] ⚠️  character_db.json not found.")

    # Build emotion/trait map per character
    char_voice_map = {
        c["name"]: {
            "traits":  c.get("traits", []),
            "emotion": c.get("traits", ["neutral"])[0]
        }
        for c in character_db
    }

    os.makedirs("outputs/audio", exist_ok=True)

    audio_tracks = []

    for scene in scenes:
        scene_id = scene["scene_id"]
        dialogue = scene.get("dialogue", [])

        scene_audio_paths = []   # will hold .wav paths (not .mp3)

        print(f"\n[Scene {scene_id}] Voice synthesis...")

        for idx, d in enumerate(dialogue):
            speaker = d.get("speaker", "Narrator")
            line    = d.get("line", "")
            emotion = char_voice_map.get(speaker, {}).get("emotion", "neutral")

            try:
                result = mcp_call("voice_cloning_synthesizer", {
                    "speaker":    speaker,
                    "line":       line,
                    "emotion":    emotion,
                    "scene_id":   scene_id,
                    "line_index": idx
                })

                # audio_path is always a .wav (client converts MP3→WAV internally)
                audio_path = result["audio_path"]
                scene_audio_paths.append(audio_path)
                dur = result.get("duration_ms", 0)
                print(f"  🎙  {speaker}: {audio_path} ({dur}ms)")

            except Exception as e:
                print(f"  ⚠️  Voice synthesis failed for {speaker} (line {idx}): {e}")

        # Merge all per-line WAV files into one scene WAV
        # _merge_scene_audio only accepts .wav files — the client guarantees this
        try:
            merged = mcp_call("merge_scene_audio", {
                "scene_id":   scene_id,
                "audio_files": scene_audio_paths
            })
            merged_path = merged["merged_path"]
            merged_dur  = merged["duration_ms"]
        except Exception as e:
            print(f"  ⚠️  Audio merge failed for scene {scene_id}: {e}")
            merged_path = ""
            merged_dur  = 0

        audio_tracks.append({
            "scene_id":     scene_id,
            "audio_files":  scene_audio_paths,    # list of .wav paths
            "merged_audio": merged_path,
            "duration_ms":  merged_dur
        })

    state["audio_tracks"] = audio_tracks

    mcp_call("commit_memory", {
        "stage":        "voice_synthesis",
        "audio_tracks": audio_tracks
    })

    print("\n[Voice Synthesis] ✅ Complete")
    return state