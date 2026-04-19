import os
import json
from mcp.client import mcp_call


def video_generation_node(state):
    print("\n[Video Generation] Generating scene visuals...")

    scenes = state.get("scenes", [])
    character_db = state.get("character_db", [])

    # Load character_db if missing
    if not character_db:
        try:
            with open("outputs/character_db.json", "r", encoding="utf-8") as f:
                character_db = json.load(f)
            state["character_db"] = character_db
        except FileNotFoundError:
            print("[Video Generation] ⚠️ character_db.json not found.")

    char_appearance_map = {
        char["name"]: char.get("appearance", "")
        for char in character_db
    }

    os.makedirs("outputs/raw_scenes", exist_ok=True)
    os.makedirs("outputs/frames", exist_ok=True)

    video_outputs = []

    for scene in scenes:
        scene_id = scene.get("scene_id")
        location = scene.get("location", "Unknown Location")
        visual_cues = scene.get("visual_cues", "")
        characters = scene.get("characters", [])

        print(f"\n  → Scene {scene_id}: {location}")

        char_refs = []
        for char_name in characters:
            appearance = char_appearance_map.get(char_name, "")
            char_refs.append({"name": char_name, "appearance": appearance})

        frame_dir = f"outputs/frames/scene_{scene_id:02d}/"
        placeholder_video = f"outputs/raw_scenes/scene_{scene_id:02d}_raw.mp4"

        try:
            result = mcp_call("query_stock_footage", {
                "scene_id": scene_id,
                "location": location,
                "visual_cues": visual_cues,
                "characters": char_refs,
                "output_path": placeholder_video
            })

            video_path = result.get("video_path", "")
            frame_dir = result.get("frame_dir", frame_dir)
            frame_count = result.get("frame_count", 48)

            print(f"    ✅ MCP Video OK")

        except Exception as e:
            print(f"    ⚠️ MCP failed → generating dummy frames: {e}")
            _generate_dummy_frames(frame_dir, scene_id)
            video_path = ""
            frame_count = 48

        video_outputs.append({
            "scene_id": scene_id,
            "video_path": video_path,
            "frame_dir": frame_dir,
            "frame_count": frame_count,
            "location": location
        })

    print(f"\n[Video Generation] ✅ Completed for {len(video_outputs)} scenes.")

    state["video_outputs"] = video_outputs

    mcp_call("commit_memory", {
        "stage": "video_generation",
        "video_outputs": video_outputs
    })

    return state


# 🔥 REAL FRAME GENERATION (IMPORTANT)
def _generate_dummy_frames(frame_dir, scene_id, frame_count=48):
    from PIL import Image, ImageDraw
    import os

    os.makedirs(frame_dir, exist_ok=True)

    for i in range(frame_count):
        img = Image.new("RGB", (640, 360), (50, 100, 150))  # visible blue

        draw = ImageDraw.Draw(img)
        draw.text((200, 150), f"Scene {scene_id}", fill=(255, 255, 0))
        draw.text((200, 180), f"Frame {i}", fill=(255, 255, 255))

        img.save(os.path.join(frame_dir, f"frame_{i:04d}.png"))