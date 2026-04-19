import json
import os
from workflows.langgraph_flow import build_graph

BANNER = """
╔══════════════════════════════╗
║   PROJECT MONTAGE - Phase 1  ║
╚══════════════════════════════╝
"""

def main():
    print(BANNER)

    mode = input("Enter mode (manual/auto): ").strip().lower()
    if mode not in ("manual", "auto"):
        print("Invalid mode. Defaulting to 'auto'.")
        mode = "auto"

    if mode == "auto":
        user_input = input("Enter prompt: ").strip()
    else:
        print("Paste your script below. Press Enter twice when done:")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        user_input = "\n".join(lines)

    state = {
        "mode":  mode,
        "input": user_input
    }

    print("\n[Pipeline] Starting LangGraph workflow...\n")

    graph = build_graph()
    result = graph.invoke(state)

    # ── Save outputs ──────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/images", exist_ok=True)

    scene_path = "outputs/scene_manifest.json"
    char_path  = "outputs/character_db.json"

    with open(scene_path, "w", encoding="utf-8") as f:
        json.dump(result.get("scene_manifest", {}), f, indent=2)

    with open(char_path, "w", encoding="utf-8") as f:
        json.dump(result.get("character_db", []), f, indent=2)

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ Phase 1 Completed Successfully!")
    print("=" * 60)
    print(f"  📄 Script:     {scene_path}")
    print(f"  👥 Characters: {char_path}")
    print(f"  🖼️  Images:     outputs/images/")
    print(f"  🧠 Memory:     memory_db/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
