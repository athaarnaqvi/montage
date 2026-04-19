from langgraph.graph import StateGraph

from agents.scriptwriter import scriptwriter
from agents.validator import script_validator
import agents.hitl as hitl_mod
from agents.character_designer import character_designer
from agents.image_synthesizer import image_generator
from agents.memory_agent import memory_commit
from agents.visual_cue_generator import visual_cue_generator


def mode_selector(state):
    """Entry node — routes to manual or auto based on state['mode']."""
    return state


def build_graph():
    builder = StateGraph(dict)

    # Register nodes
    builder.add_node("mode_selector",   mode_selector)
    builder.add_node("validator",       script_validator)
    builder.add_node("visual_cues",     visual_cue_generator)
    builder.add_node("scriptwriter",    scriptwriter)
    builder.add_node("hitl",            hitl_mod.human_checkpoint)
    builder.add_node("character",       character_designer)
    builder.add_node("image",           image_generator)
    builder.add_node("memory",          memory_commit)

    # Entry point
    builder.set_entry_point("mode_selector")

    # Route: manual → validator, auto → scriptwriter
    builder.add_conditional_edges(
        "mode_selector",
        lambda state: state.get("mode", "auto"),
        {
            "manual": "validator",
            "auto":   "scriptwriter"
        }
    )

    # Linear pipeline after script is ready
    builder.add_edge("validator",    "visual_cues")
    builder.add_edge("visual_cues",  "hitl")
    builder.add_edge("scriptwriter", "hitl")
    builder.add_edge("hitl",         "character")
    builder.add_edge("character",    "image")
    builder.add_edge("image",        "memory")

    return builder.compile()
