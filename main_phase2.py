import json
import os
from workflows.phase2_flow import build_phase2_graph

BANNER = """
╔══════════════════════════════════════╗
║   PROJECT MONTAGE — Phase 2          ║
║   The Studio Floor: AV Synthesis     ║
╚══════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # Load Phase 1 outputs as Phase 2 inputs
    scene_manifest = {}
    character_db = []
    images = []

    scene_path = "outputs/scene_manifest.json"
    char_path = "outputs/character_db.json"
    images_dir = "outputs/images"

    if os.path.exists(scene_path):
        with open(scene_path, "r", encoding="utf-8") as f:
            scene_manifest = json.load(f)
        print(f"[Phase 2] Loaded scene_manifest.json ({len(scene_manifest.get('scenes', []))} scenes)")
    else:
        print(f"[Phase 2] ⚠️  {scene_path} not found. Run Phase 1 first.")
        return

    if os.path.exists(char_path):
        with open(char_path, "r", encoding="utf-8") as f:
            character_db = json.load(f)
        print(f"[Phase 2] Loaded character_db.json ({len(character_db)} characters)")

    if os.path.exists(images_dir):
        images = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.endswith(".png")
        ]
        print(f"[Phase 2] Found {len(images)} character images.")

    # Build initial state from Phase 1 outputs
    state = {
        "scene_manifest": scene_manifest,
        "character_db": character_db,
        "images": images
    }

    print("\n[Pipeline] Starting Phase 2 LangGraph workflow...\n")

    graph = build_phase2_graph()
    result = graph.invoke(state)

    # ── Final summary ─────────────────────────────────────────
    final_scenes = result.get("final_scenes", [])

    print("\n" + "=" * 60)
    print("✅ Phase 2 Completed Successfully!")
    print("=" * 60)
    for scene in final_scenes:
        print(f"  🎬 Scene {scene['scene_id']:02d}: {scene['output_path']}")
    print(f"  🎙  Audio:   outputs/audio/")
    print(f"  🎞  Frames:  outputs/frames/")
    print(f"  🧠  Memory:  memory_db/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()