import os
from dotenv import load_dotenv

load_dotenv()

MCP_REGISTRY_PATH = os.getenv("MCP_REGISTRY_PATH", "config/mcp_registry.json")
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
OUTPUTS_DIR = "outputs"
IMAGES_DIR = "outputs/images"
MEMORY_DIR = "memory_db"
