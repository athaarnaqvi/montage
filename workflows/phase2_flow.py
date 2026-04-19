import concurrent.futures
from langgraph.graph import StateGraph

from agents.scene_parser import scene_parser_node
from agents.voice_synthesis import voice_synthesis_node
from agents.video_generation import video_generation_node
from agents.face_swap import face_swap_node
from agents.lip_sync import lip_sync_node


# ─────────────────────────────────────────────────────────────────────────────
# Parallel Execution Node
# Runs audio branch (voice_synthesis) and video branch (video_generation)
# concurrently, then merges their results into a single state dict.
# ─────────────────────────────────────────────────────────────────────────────

def parallel_av_node(state):
    """
    Parallel branch node.
    Executes audio and video generation branches concurrently via ThreadPoolExecutor.
    Merges results back into shared state.
    """
    print("\n[Parallel AV] Launching audio and video branches concurrently...")

    # Take snapshots so both branches start from the same state
    audio_state = dict(state)
    video_state = dict(state)

    def run_audio(s):
        return voice_synthesis_node(s)

    def run_video(s):
        return video_generation_node(s)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        audio_future = executor.submit(run_audio, audio_state)
        video_future = executor.submit(run_video, video_state)

        audio_result = audio_future.result()
        video_result = video_future.result()

    # Merge parallel outputs into state
    state["audio_tracks"]  = audio_result.get("audio_tracks", [])
    state["video_outputs"] = video_result.get("video_outputs", [])

    print("[Parallel AV] ✅ Audio and video branches complete. Merging outputs...")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_phase2_graph():
    builder = StateGraph(dict)

    # Register nodes
    builder.add_node("scene_parser",  scene_parser_node)
    builder.add_node("parallel_av",   parallel_av_node)   # audio + video run inside here
    builder.add_node("face_swap",     face_swap_node)
    builder.add_node("lip_sync",      lip_sync_node)

    # Entry point
    builder.set_entry_point("scene_parser")

    # Linear flow: parse → parallel(audio+video) → face swap → lip sync
    builder.add_edge("scene_parser", "parallel_av")
    builder.add_edge("parallel_av",  "face_swap")
    builder.add_edge("face_swap",    "lip_sync")

    return builder.compile()