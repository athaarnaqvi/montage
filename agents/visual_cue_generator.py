import json
from mcp.client import mcp_call


def visual_cue_generator(state):
    print("\n[Visual Cue Generator] Enhancing scenes with visual details...")

    scenes = state.get("scene_manifest", {}).get("scenes", [])

    for scene in scenes:
        # Generate overall visual cues for the scene
        scene_prompt = f"""
Generate visual cues for this scene in a screenplay. Focus on atmosphere, lighting, camera angles, and visual details that enhance the emotional impact.

Scene: {scene['location']}
Characters: {', '.join(scene['characters'])}
Dialogue summary: {'; '.join([f"{d['speaker']}: {d['line']}" for d in scene['dialogue']])}

Return ONLY a JSON object like {{"visual_cues": "your description here"}}
"""

        scene_result = mcp_call("generate_script_segment", {"prompt": scene_prompt})
        if isinstance(scene_result, str):
            scene_result = json.loads(scene_result)

        # Extract visual cues from the response
        visual_cues_data = scene_result.get("visual_cues", {})
        if isinstance(visual_cues_data, dict):
            visual_cues = f"Atmosphere: {visual_cues_data.get('atmosphere', '')}. Lighting: {visual_cues_data.get('lighting', '')}. Camera angles: {visual_cues_data.get('camera_angles', '')}. Visual details: {visual_cues_data.get('visual_details', '')}."
        else:
            visual_cues = str(visual_cues_data)
        scene["visual_cues"] = visual_cues

        # Generate visual cues for each dialogue line
        for dialogue in scene["dialogue"]:
            if dialogue["speaker"] != "Narrator":
                cue_prompt = f"""
Generate a brief visual cue (1 sentence) for this dialogue line in the scene.

Scene: {scene['location']}
Speaker: {dialogue['speaker']}
Line: {dialogue['line']}

Return ONLY a JSON object like {{"visual_cue": "your cue here"}}
"""

                cue_result = mcp_call("generate_script_segment", {"prompt": cue_prompt})
                if isinstance(cue_result, str):
                    cue_result = json.loads(cue_result)

                visual_cue = cue_result.get("visual_cue", "")
                dialogue["visual_cue"] = visual_cue

    print(f"[Visual Cue Generator] ✅ Enhanced {len(scenes)} scenes with visual details.")

    return state