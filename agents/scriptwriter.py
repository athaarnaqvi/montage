import json
from mcp.client import mcp_call


def scriptwriter(state):
    print("\n[Scriptwriter] Generating script...")

    prompt = f"""
You are a professional screenwriter. Generate a structured screenplay in STRICT JSON format.

RULES:
- Return ONLY valid JSON. No markdown. No explanation.
- Create exactly 3 scenes.
- Each scene must have: scene_id, location, characters (list of names), dialogue (list of objects with speaker/line/visual_cue), visual_cues (string).

FORMAT:
{{
  "scenes": [
    {{
      "scene_id": 1,
      "location": "descriptive location name",
      "characters": ["Character One", "Character Two"],
      "dialogue": [
        {{
          "speaker": "Character One",
          "line": "spoken line here",
          "visual_cue": "camera direction or visual detail"
        }}
      ],
      "visual_cues": "overall scene atmosphere and lighting"
    }}
  ]
}}

Story prompt: {state['input']}
"""

    result = mcp_call("generate_script_segment", {"prompt": prompt})

    # Handle both dict and string responses
    if isinstance(result, str):
        result = json.loads(result)

    scenes = result.get("scenes", [])

    # Flatten if Grok nested scenes inside visual_cues
    if len(scenes) == 1 and isinstance(scenes[0].get("visual_cues"), dict):
        nested = scenes[0]["visual_cues"].get("scenes", [])
        if nested:
            scenes = nested

    print(f"[Scriptwriter] Generated {len(scenes)} scenes.")

    state["scene_manifest"] = {"scenes": scenes}
    return state
