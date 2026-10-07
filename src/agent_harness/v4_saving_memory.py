import sys
import json
import agent_harness
from pathlib import Path


MODEL = "Qwen3.5-9B-MLX-8bit"
WORKSPACE = Path(__file__).parent / "workspace"


# MEMORY
MEMORY_FILE = Path(__file__).parent / "workspace/memory.md"


def load_memory() -> str:
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return"(nothing saved yet)"


def save_memory(fact: str) -> str:
    """Append one fact about the user to long-term memory."""
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    return f"saved: {fact}"


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
TOOLS = {
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


SYSTEM_PROMPT = """You are a helpful personal assistant.

Here is what you remember about the user from previous sessions:
{memory}

When you learn a new lasting fact about the user, save it with save_memory."""


# AGENT LOOP
def run_agent(client, messages: list) -> str:
    while True:
        response = client.chat.completions.create(
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
            result = TOOLS[call.function.name](**args)
            messages.append(
                { "role": "tool", "tool_call_id": call.id, "content": result }
            )


def user_loop():
    client = agent_harness.create_client()

    messages = [
        { "role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory()) },
    ]
    print(f"v4 assistant ({MODEL}) - memory loadedl - ctrl+c to quit")

    try:
        # User prompt loop to continue prompting the model.
        while True:
            messages.append({ "role": "user", "content": input("\nYou: ") })
            print("\nBot: ", run_agent(client, messages))

    except(EOFError, KeyboardInterrupt):
        print("\nbye!")


def main():
    user_loop()


if __name__ == "__main__":
    main()
