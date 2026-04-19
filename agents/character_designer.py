import json
from mcp.client import mcp_call


def character_designer(state):
    print("\n[Character Designer] Extracting character identities...")

    scenes = state.get("scene_manifest", {}).get("scenes", [])

    # Collect unique character names across all scenes
    character_names = set()
    for scene in scenes:
        for name in scene.get("characters", []):
            if name and isinstance(name, str):
                character_names.add(name.strip())

    characters = []

    for name in character_names:
        prompt = f"""
Return ONLY valid JSON for a screenplay character. No markdown. No explanation.

Character name: {name}

Format:
{{
  "name": "{name}",
  "traits": ["trait1", "trait2", "trait3"],
  "appearance": "detailed physical description for an artist",
  "style": "visual art style direction"
}}

Rules:
- traits must be real descriptive personality words
- appearance must be 2-3 sentences of physical detail
- style describes the visual rendering style for image generation
"""

        try:
            result = mcp_call("generate_script_segment", {"prompt": prompt})

            if isinstance(result, str):
                char_data = json.loads(result)
            else:
                char_data = result

            # Validate required fields
            required = {"name", "traits", "appearance", "style"}
            if not required.issubset(char_data.keys()):
                raise ValueError(f"Missing fields in character response: {char_data.keys()}")

            # Ensure name matches
            char_data["name"] = name

        except Exception as e:
            print(f"  ⚠️  Fallback used for {name}: {e}")
            char_data = {
                "name": name,
                "traits": ["determined", "intelligent", "adaptive"],
                "appearance": f"{name} with a cinematic, detailed appearance suited to the story world.",
                "style": "stylized fantasy with cinematic realism"
            }

        style = char_data.get("style", "cinematic")
        print(f"  → Character: {name} | Style: {style}")
        characters.append(char_data)

    print(f"[Character Designer] ✅ {len(characters)} characters processed.")
    state["character_db"] = characters
    return state
