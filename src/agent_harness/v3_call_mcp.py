import tracemalloc; tracemalloc.start()

import sys
import json
import agent_harness


# Imports for MCP
import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


MODEL = "Qwen3.5-9B-MLX-8bit"
WORKSPACE = Path(__file__).parent / "workspace"


# Each server is just a command to launch. Same format Claude Code uses.
MCP_SERVERS = {
    "time": StdioServerParameters(command="uvx", args=["mcp-server-time"]),
    "fetch": StdioServerParameters(command="uvx", args=["mcp-server-fetch"]),
}


# TOOL FUNCTIONS
def list_files() -> str:
    """List the files in the assistant's workspace folder."""
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    """Read a file from the workspace folder."""
    path = WORKSPACE / filename

    if not path.is_file():
        return f"ERROR: no file named {filename}"

    return path.read_text(encoding="utf-8")

    #return path.read_text(encoding="utf-8") if path.is_file() else f"ERROR: no file named {filename}"


def write_file(filename: str, content: str) -> str:
    """Write (or overwrite) a file in the workspace folder."""
    (WORKSPACE / filename).write_text(content, encoding="utf-8")
    return f"wrote {filename}"


# LOCAL TOOL LOOKUP TABLE
LOCAL_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
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
]


# MCP

# Hold a map of servers to client sessions (tool name -> live session)
mcp_sessions: dict[str, ClientSession] = {}


async def connect_mcp(stack: AsyncExitStack):
    """Launch each MCP server and merge its tools into TOOL_SCHEMAS."""
    for name, params in MCP_SERVERS.items():
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

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


async def call_tool(name: str, args: dict) -> str:
    """Run a tool, either ours directly, or an MCP server's over the protocol."""

    # Try local tools, our functions defined above, first.
    if name in LOCAL_TOOLS:
        return LOCAL_TOOLS[name](**args)

    # Otherwise, run an MCP request.
    result = await mcp_sessions[name].call_tool(name, args)
    return "\n".join(c.text for c in result.content if hasattr(c, "text"))


# AGENT LOOP

async def run_agent(client, user_message: str) -> str:
    messages = [
        { "role": "system", "content": "You are a helpful personal assistant." },
        { "role": "user", "content": user_message },
    ]

    # THE agent loop. This is the whole trick.
    while True:
        response = await client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS
        )
        response_msg = response.choices[0].message

        # No tool calls, so just print the response message.
        if not response_msg.tool_calls:
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


async def user_loop():
    async with AsyncExitStack() as stack:
        client = agent_harness.create_async_client()

        await connect_mcp(stack)
        print(f"v3 assistant ({MODEL}) - ctrl+c to quit")

        try:
            # User prompt loop to continue prompting the model.
            while True:
                question = await asyncio.to_thread(input, "\nYou: ")
                print("\nBot: ", await run_agent(client, question))

        except(EOFError, KeyboardInterrupt):
            print("\nbye!")


# Entrypoint into async
def main():
    asyncio.run(user_loop())


if __name__ == "__main__":
    main()

