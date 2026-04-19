from mcp.client import mcp_call


def memory_commit(state):
    print("\n[Memory] Committing pipeline state to memory store...")
    mcp_call("commit_memory", state)
    print("[Memory] ✅ State saved to memory_db/")
    return state
