import os
from mcp.client import mcp_call


def image_generator(state):
    print("\n[Image Synthesizer] Generating character visuals...")

    characters = state.get("character_db", [])
    images = []

    for char in characters:
        name = char.get("name", "").strip()
        if not name:
            print("  ⚠️  Skipping character with no name.")
            continue

        appearance = char.get("appearance", "")
        style = char.get("style", "cinematic, ultra realistic")

        sd_prompt = (
            f"Portrait of {name}, {appearance}, "
            f"{style}, ultra realistic, cinematic lighting, detailed"
        )

        print(f"  → Synthesizing image for: {name}")
        print(f"  → SD Prompt: {sd_prompt}")

        # Filename: replace spaces with underscores
        filename = name.replace(" ", "_")

        try:
            filepath = mcp_call("generate_image", {
                "prompt": sd_prompt,
                "filename": filename
            })

            print(f"  ✅ Image saved: {filepath}")
            images.append(filepath)

        except Exception as e:
            print(f"  ❌ Skipping {name} after all retries: {e}")

    print(f"[Image Synthesizer] ✅ {len(images)} images generated.")
    state["images"] = images
    return state