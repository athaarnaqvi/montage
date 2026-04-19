import json
from mcp.client import mcp_call


def scene_parser_node(state):
    print("\n[Scene Parser] Parsing scene_manifest.json into executable tasks...")

    scene_manifest = state.get("scene_manifest", {})

    if not scene_manifest:
        # Try loading from disk if not in state
        try:
            with open("outputs/scene_manifest.json", "r", encoding="utf-8") as f:
                scene_manifest = json.load(f)
            state["scene_manifest"] = scene_manifest
            print("[Scene Parser] Loaded scene_manifest.json from disk.")
        except FileNotFoundError:
            raise RuntimeError("[Scene Parser] scene_manifest.json not found in state or disk.")

    scenes = scene_manifest.get("scenes", [])
    if not scenes:
        raise RuntimeError("[Scene Parser] No scenes found in scene_manifest.")

    # Build task graph using MCP tool
    task_graph = mcp_call("get_task_graph", {"scenes": scenes})

    print(f"[Scene Parser] Task graph built: {len(task_graph.get('tasks', []))} tasks.")
    print(f"[Scene Parser] ✅ {len(scenes)} scenes parsed and queued for parallel processing.")

    state["task_graph"] = task_graph
    state["scenes"] = scenes
    state["current_scene_index"] = 0

    # Commit intermediate state
    mcp_call("commit_memory", {
        "stage": "scene_parser",
        "task_graph": task_graph,
        "scene_count": len(scenes)
    })

    return state