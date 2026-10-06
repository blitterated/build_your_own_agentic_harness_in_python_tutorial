import json
import sys
import agent_harness

from pathlib import Path


MODEL = "Qwen3.5-9B-MLX-8bit"
WORKSPACE = Path(__file__).parent / "workspace"


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

def write_file(filename: str, content: str) -> str:
    """Write (or overwrite) a file in the workspace folder."""
    (WORKSPACE / filename).write_text(content, encoding="utf-8")
    return f"wrote {filename}"


TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
}


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


def run_agent(client, user_message: str) -> str:
    messages = [
        { "role": "system", "content": "You are a helpful personal assistant." },
        { "role": "user", "content": user_message },
    ]

    # THE agent loop. This is the whole trick.
    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS
        )
        response_msg = response.choices[0].message

        # No tool calls, so just print the response message.
        if not response_msg.tool_calls:
            return response_msg.content or ""  # done - the model answered our request.

        # Run the tools the model asked for, and give it the results.
        messages.append(response_msg)
        for call in response_msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"  [tool] {call.function.name}({args})")
            result = TOOLS[call.function.name](**args)
            messages.append(
                { "role": "tool", "tool_call_id": call.id, "content": result }
            )


def loop():
    client = agent_harness.create_client()

    print(f"v2 assistant ({MODEL}) - ctrl+c to quit")

    try:
        while True:
            question = input("\nYou: ")
            print("\nBot: ", run_agent(client, question))

    except(EOFError, KeyboardInterrupt):
        print("\nbye!")


if __name__ == "__main__":
    loop()
