import json
import sys
import agent_harness

from pathlib import Path


MODEL = "Llama-3.2-3B-Instruct-8bit"
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


TOOLS = { "list_files": list_files, "read_file": read_file }


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
]


def chat(client, user_message: str) -> str:
    messages = [
        { "role": "system", "content": "You are a helpful personal assistant." },
        { "role": "user", "content": user_message },
    ]

    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOL_SCHEMAS
    )
    response_msg = response.choices[0].message

    # No tool calls, so just print the response message.
    if not response_msg.tool_calls:
        return response_msg.content or ""

    # Run the tool the model asked for, and give it the result... ONCE.
    messages.append(response_msg)
    for call in response_msg.tool_calls:
        args = json.loads(call.function.arguments or "{}")
        print(f"  [tool] {call.function.name}({args})")
        result = TOOLS[call.function.name](**args)
        messages.append(
            { "role": "tool", "tool_call_id": call.id, "content": result }
        )

    final = client.chat.completions.create(model=MODEL, messages=messages)
    return final.choices[0].message.content


def loop():
    client = agent_harness.create_client()

    print(f"v1 assistant ({MODEL}) - ctrl+c to quit")

    try:
        while True:
            question = input("\nYou: ")
            print("\nBot: ", chat(client, question))

    except(EOFError, KeyboardInterrupt):
        print("\nbye!")


if __name__ == "__main__":
    loop()
