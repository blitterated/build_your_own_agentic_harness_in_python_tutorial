#import tracemalloc; tracemalloc.start()

import agent_harness
import json
import os
import sys

from pathlib import Path
from pydantic import BaseModel
from openai import AsyncOpenAI


# Imports for MCP
import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


#MODEL = "Qwen3.5-9B-MLX-8bit"
ROOT = Path(__file__).parent
WORKSPACE = ROOT / "workspace"
MEMORY_FILE = ROOT / "memory.md"
MCP_CONFIG = ROOT / "mcp_servers.json"


# ----------------------------------------------------------
# MODEL REGISTRY
#
# Every entry is just an OpenAI-compatible enpoint.
# ----------------------------------------------------------

class ModelConfig(BaseModel):
    base_url: str
    api_key: str = "none"
    model: str


MODELS: dict[str, ModelConfig] = {
    "local-small": ModelConfig(
        base_url="http://127.0.0.1:8000/v1",
        api_key=agent_harness.get_api_key(),
        model="Llama-3.2-3B-Instruct-8bit",
    ),
    "local": ModelConfig(
        base_url="http://127.0.0.1:8000/v1",
        api_key=agent_harness.get_api_key(),
        model="Qwen3.5-9B-MLX-8bit",
    ),
    "local-large": ModelConfig(
        base_url="http://127.0.0.1:8000/v1",
        api_key=agent_harness.get_api_key(),
        model="Qwen3.8-27B-8bit",
    ),
    "local-coder": ModelConfig(
        base_url="http://127.0.0.1:8000/v1",
        api_key=agent_harness.get_api_key(),
        model="Qwen2.5-Coder-7B-Instruct-MLX-8bit",
    ),
    "gpt": ModelConfig(
        base_url="http://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model="gpt-5.2",
    ),
    "claude": ModelConfig(
        base_url="http://api.anthropic.com/v1/",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model="claude-sonnet-5",
    ),
}


# Switch at runtime with /model
model_name = "local"


# ----------------------------------------------------------
# MEMORY
# ----------------------------------------------------------

def load_memory() -> str:
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return "(nothing saved yet)"


def save_memory(fact: str) -> str:
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    return f"saved {fact}"


# ----------------------------------------------------------
# LOCAL TOOLS
#
# Tool functions exposed to the models via schemas.
# ----------------------------------------------------------

def list_files() -> str:
    """List the files in the assistant's workspace folder."""
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    """Read a file from the workspace folder."""
    path = WORKSPACE / filename

    if not path.is_file():
        return f"ERROR: no file named {filename}"

    return path.read_text(encoding="utf-8")


def write_file(filename: str, content: str) -> str:
    """Write (or overwrite) a file in the workspace folder."""
    (WORKSPACE / filename).write_text(content, encoding="utf-8")
    return f"wrote {filename}"


# LOCAL TOOL LOOKUP TABLE
LOCAL_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "save_memory": save_memory,
}


# DEFINITELY NOT TOOL SCHEMAS
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in user's workpace folder.",
            "parameters": { "type": "object", "properties": {} },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": { "filename": { "type": "string" } },
                "required": [ "filename" ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": { "type": "string" },
                    "content": { "type": "string" },
                },
                "required": [ "filename", "content" ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save one short fact about the user to long-term memory.",
                "Use whenever you learn something worth remembering: their ",
                "name, preferences, projects, recurring tasks.",
            ),
            "parameters": {
                "type": "object",
                "properties": { "fact": { "type": "string" } },
                "required": [ "fact" ],
            },
        },
    },
]


# ----------------------------------------------------------
# SYSTEM PROMPT
# ----------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful personal assitant running inside a \
custom harness. Be concise.

The user's workspace folder holds their personal files: notes, todo lists, \
ideas. you have tools to lists, read, and write those files. Never claim you \
lack access to the user's files or tasks - use your tools to look.

Here is what you remember about the user from previous sessions:
{memory}

When you learn a new lasting fact about the user, save it with save_memory."""


# ----------------------------------------------------------
# MESSAGES
#
# The running conversation - shared by every model we switch to.
# ----------------------------------------------------------

messages = [
    { "role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory()) },
]


# ----------------------------------------------------------
# MCP
#
# Tool functions exposed to the models via schemas.
# ----------------------------------------------------------

# Hold a map of servers to client sessions (tool name -> live session)
mcp_sessions: dict[str, ClientSession] = {}


async def connect_mcp(stack: AsyncExitStack):
    """Launch every server in mcp_servers.json and merge in its tools."""
    config = json.loads(MCP_CONFIG.read_text(encoding="utf-8"))

    for name, spec in config["mcpServers"].items():
        params = StdioServerParameters(command=spec["command"], args=spec["args"])

        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception as e:
            print(f"  [mcp] '{name}' ERROR: Failed to start: {e}")
            continue

        tools = (await session.list_tools()).tools
        for tool in tools:
            mcp_sessions[tool.name] = session
            TOOL_SCHEMAS.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
            })
        print(f"  [mcp] connected '{name}': {[t.name for t in tools]}")


# ----------------------------------------------------------
# MCP
#
# Figure out if a tool is local, MCP, or imaginary.
# ----------------------------------------------------------

async def call_tool(name: str, args: dict) -> str:
    """Run a tool, either ours directly, or an MCP server's over the protocol."""

    # Try local tools, our functions defined above, first.
    if name in LOCAL_TOOLS:
        return LOCAL_TOOLS[name](**args)

    # Then see if it's an MCP request.
    if name in mcp_sessions:
        result = await mcp_sessions[name].call_tool(name, args)
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    # Otherwise, return an error message.
    return f"ERROR: Unknown Tool: {name}"


# ----------------------------------------------------------
# AGENT LOOP
#
# Loop over tool calls until the Model has an answer.
# ----------------------------------------------------------

async def run_agent(client, user_message: str) -> str:
    cfg = MODELS[model_name]
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    messages.append({ "role": "user", "content": user_message })
    while True:
        response = await client.chat.completions.create(
            model=cfg.model, messages=messages, tools=TOOL_SCHEMAS
        )
        response_msg = response.choices[0].message

        # No tool calls, so just print the response message.
        if not response_msg.tool_calls:
            messages.append({ "role": "assistant", "content": response_msg.content })
            return response_msg.content or ""

        # Run the tools the model asked for, and give it the results.
        messages.append(response_msg)
        for call in response_msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"  [tool] {call.function.name}({args})")
            result = await call_tool(call.function.name, args)
            messages.append(
                { "role": "tool", "tool_call_id": call.id, "content": result }
            )


# ----------------------------------------------------------
# SLASH COMMANDS
# ----------------------------------------------------------

def list_models():
    for name, cfg in MODELS.items():
        marker = "*" if name == model_name else " "
        print(f" {marker} {name:<12} {cfg.model}  ({cfg.base_url})")


def switch_model(model):
    global model_name

    if model in MODELS:
        model_name = model
        print(f"  switched to {model} ({MODELS[model].model})")
    else:
        print(f"  unknown model '{model}' - try /models")


def list_tools():
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        origin = "local" if fn["name"] in LOCAL_TOOLS else "mcp"
        print(f"  [{origin}] {fn['name']}: {fn['description'][:60]}")


def handle_command(line: str) -> bool:
    """Returns True if the user entered a command."""

    # Bail early if it's not a command.
    if not line.startswith("/"):
        return False

    cmd, _, arg = line.partition(" ")
    if cmd == "/models":
        list_models()
    elif cmd == "/model":
        switch_model(arg)
    elif cmd == "/tools":
        list_tools()
    elif cmd == "/memory":
        print(load_memory())
    elif cmd == "/quit":
        raise SystemExit
    else:
        print("  commands: /models, /model <name>, /tools, /memory, /quit")

    return True



# ----------------------------------------------------------
# USER LOOP
#
# Loop over user prompts in between agent loops.
# ----------------------------------------------------------

async def user_loop():
    async with AsyncExitStack() as stack:
        print("Starting harness...")

        client = agent_harness.create_async_client()

        await connect_mcp(stack)
        print(
            f"\nHarness read - model: {model_name} "
            f"{MODELS[model_name].model}), "
            f"{len(TOOL_SCHEMAS)} tools, memory loaded"
        )

        print("Type /models, /model <name>, /tools, /memory, or just chat.\n")
        try:
            # User prompt loop to continue prompting the model.
            while True:
                line = (await asyncio.to_thread(input, "\nYou: ")).strip()

                if not line or handle_command(line):
                    continue

                print("\nBot: ", await run_agent(client, line))

        except(EOFError, KeyboardInterrupt, SystemExit):
            print("\nbye!")


# Entrypoint into async
def main():
    asyncio.run(user_loop())


if __name__ == "__main__":
    main()
