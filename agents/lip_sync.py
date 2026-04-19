import os
from mcp.client import mcp_call


def lip_sync_node(state):
    print("\n[Lip Sync] Synchronizing audio waveforms with facial movements...")

    audio_tracks         = state.get("audio_tracks", [])
    face_swapped_outputs = state.get("face_swapped_outputs", [])
    video_outputs        = state.get("video_outputs", [])

    audio_map = {a["scene_id"]: a for a in audio_tracks}
    face_map  = {f["scene_id"]: f for f in face_swapped_outputs}
    video_map = {v["scene_id"]: v for v in video_outputs}

    final_scenes = []

    for scene_id in sorted(audio_map.keys()):
        audio_data = audio_map[scene_id]
        face_data  = face_map.get(scene_id, {})
        video_data = video_map.get(scene_id, {})

        # audio_files is already a list of .wav paths (set by voice_synthesis_node)
        audio_files    = audio_data.get("audio_files", [])
        swapped_frames = face_data.get("swapped_frames", [])
        source_video   = face_data.get("source_video",
                         video_data.get("video_path", ""))

        print(f"\n  → Scene {scene_id}: {len(audio_files)} audio files, "
              f"{len(swapped_frames)} swapped characters")

        try:
            result = mcp_call("lip_sync_aligner", {
                "scene_id":      scene_id,
                "source_video":  source_video,
                "swapped_frames":swapped_frames,
                "audio_files":   audio_files,        # list of .wav paths
                "dialogue":      audio_data.get("audio_files", []),
                "output_path":   f"outputs/raw_scenes/scene_{scene_id:02d}.mp4"
            })

            print(f"    ✅ Scene {scene_id}: {result.get('output_path')}")
            print(f"       audio={result.get('audio_duration_s',0):.2f}s "
                  f"video={result.get('video_duration_s',0):.2f}s "
                  f"sync={result.get('sync_score',0):.2f} "
                  f"voiced={result.get('voiced_frames',0)}/{result.get('frame_count',0)}")
            final_scenes.append(result)

        except Exception as e:
            print(f"    ⚠️  Lip sync failed for scene {scene_id}: {e}")
            final_scenes.append({
                "scene_id":   scene_id,
                "output_path": f"outputs/raw_scenes/scene_{scene_id:02d}.mp4",
                "sync_score":  0.0
            })

    print(f"\n[Lip Sync] ✅ Temporal alignment complete for {len(final_scenes)} scenes.")

    state["final_scenes"] = final_scenes

    mcp_call("commit_memory", {
        "stage":        "lip_sync_complete",
        "final_scenes": final_scenes
    })

    return state