import json
from mcp.client import mcp_call


def human_checkpoint(state):
    """
    HITL checkpoint. On rejection:
    - Asks for feedback
    - If feedback given → regenerates script incorporating feedback, loops back
    - If no feedback → halts pipeline
    """
    while True:
        _print_script(state)

        answer = input("✅ Approve script and continue? (y/n): ").strip().lower()

        if answer == "y":
            print("[HITL] Script approved. Continuing pipeline...\n")
            return state

        elif answer == "n":
            print("\n[HITL] Script rejected.")
            feedback = input("💬 Enter feedback for regeneration (or press Enter to stop): ").strip()

            if not feedback:
                print("[HITL] No feedback provided. Halting pipeline.")
                raise SystemExit("Pipeline halted — script rejected with no feedback.")

            print(f"\n[HITL] Regenerating script with feedback: \"{feedback}\"\n")
            state = _regenerate_with_feedback(state, feedback)
            print("[Scriptwriter] Script regenerated. Returning to review...\n")
            # loop continues → shows new script for approval

        else:
            print("Please enter 'y' or 'n'.")


def _print_script(state):
    print("\n" + "=" * 60)
    print("       HUMAN REVIEW CHECKPOINT")
    print("=" * 60)
    print("\n📄 Generated Script:\n")
    print(json.dumps(state["scene_manifest"], indent=2))
    print("\n" + "=" * 60)


def _regenerate_with_feedback(state, feedback):
    """Call the scriptwriter again with the original prompt + user feedback."""
    import json as _json
    original_input = state.get("input", "")

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

Story prompt: {original_input}

REVISION FEEDBACK FROM REVIEWER: {feedback}
Make sure to fully address this feedback in the regenerated script.
"""

    result = mcp_call("generate_script_segment", {"prompt": prompt})

    if isinstance(result, str):
        result = _json.loads(result)

    scenes = result.get("scenes", [])

    # Flatten nested scenes if needed
    if len(scenes) == 1 and isinstance(scenes[0].get("visual_cues"), dict):
        nested = scenes[0]["visual_cues"].get("scenes", [])
        if nested:
            scenes = nested

    state["scene_manifest"] = {"scenes": scenes}
    state["hitl_feedback"] = feedback
    return state