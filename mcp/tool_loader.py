import json
import os

MCP_REGISTRY_PATH = os.getenv("MCP_REGISTRY_PATH", "config/mcp_registry.json")

def load_tools():
    """
    Load MCP tools dynamically from registry.
    Returns a dictionary: {tool_name: tool_info}
    """
    if not os.path.exists(MCP_REGISTRY_PATH):
        raise FileNotFoundError(f"MCP registry not found: {MCP_REGISTRY_PATH}")

    with open(MCP_REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    tools = {}
    for tool in registry.get("tools", []):
        name = tool.get("name")
        if name:
            tools[name] = tool

    return tools
