import os
import json
from mcp.client import mcp_call


def face_swap_node(state):
    print("\n[Face Swap] Mapping character faces onto video frames...")

    video_outputs = state.get("video_outputs", [])
    character_db  = state.get("character_db", [])
    scenes        = state.get("scenes", [])

    if not character_db:
        try:
            with open("outputs/character_db.json", "r", encoding="utf-8") as f:
                character_db = json.load(f)
            state["character_db"] = character_db
        except FileNotFoundError:
            print("[Face Swap] ⚠️  character_db.json not found.")

    # Build character → image path map from state["images"]
    images         = state.get("images", [])
    char_image_map = {}
    for img_path in images:
        basename  = os.path.splitext(os.path.basename(img_path))[0]
        char_name = basename.replace("_", " ")
        char_image_map[char_name] = img_path

    os.makedirs("outputs/face_swapped", exist_ok=True)

    face_swapped_outputs = []

    for video_data in video_outputs:
        scene_id  = video_data.get("scene_id")
        video_path = video_data.get("video_path", "")
        frame_dir  = video_data.get("frame_dir", "")

        # Find which characters appear in this scene
        scene_characters = []
        for scene in scenes:
            if scene.get("scene_id") == scene_id:
                scene_characters = scene.get("characters", [])
                break

        print(f"\n  → Scene {scene_id} | Characters: {scene_characters}")

        swapped_frames = []
        identity_valid = True

        for char_name in scene_characters:
            ref_image = char_image_map.get(char_name, "")

            # Step 1: Identity validation
            try:
                validation = mcp_call("identity_validator", {
                    "character_name": char_name,
                    "reference_image": ref_image,
                    "scene_id":        scene_id
                })
                if not validation.get("valid", True):
                    print(f"    ⚠️  Identity validation failed for {char_name}. Skipping.")
                    identity_valid = False
                    continue
                print(f"    ✅ Identity validated: {char_name}")
            except Exception as e:
                print(f"    ⚠️  Identity validator error for {char_name}: {e}")

            # Step 2: Face swap
            try:
                swap_result = mcp_call("face_swapper", {
                    "character_name":  char_name,
                    "reference_image": ref_image,
                    "source_video":    video_path,
                    "frame_dir":       frame_dir,
                    "scene_id":        scene_id,
                    "output_dir":      f"outputs/face_swapped/scene_{scene_id:02d}/"
                })

                swapped_path = swap_result.get(
                    "output_path",
                    f"outputs/face_swapped/scene_{scene_id:02d}/{char_name.replace(' ','_')}.mp4"
                )
                swapped_frames.append({
                    "character":   char_name,
                    "output_path": swapped_path
                })
                print(f"    ✅ Face swapped: {char_name} → {swapped_path}")

            except Exception as e:
                print(f"    ⚠️  Face swap failed for {char_name}: {e}")
                fallback = (f"outputs/face_swapped/scene_{scene_id:02d}/"
                            f"{char_name.replace(' ','_')}_placeholder.mp4")
                os.makedirs(os.path.dirname(fallback), exist_ok=True)
                swapped_frames.append({"character": char_name, "output_path": fallback})

        face_swapped_outputs.append({
            "scene_id":      scene_id,
            "source_video":  video_path,
            "swapped_frames":swapped_frames,
            "identity_valid":identity_valid
        })

    print(f"\n[Face Swap] ✅ Face mapping completed for {len(face_swapped_outputs)} scenes.")

    state["face_swapped_outputs"] = face_swapped_outputs

    mcp_call("commit_memory", {
        "stage":                "face_swap",
        "face_swapped_outputs": face_swapped_outputs
    })

    return state