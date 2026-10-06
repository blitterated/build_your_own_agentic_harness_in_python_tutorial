import sys
import agent_harness


# Uncomment the following line of code
# so emoji don't crash the Windows console.
#sys.stdout.reconfigure(encoding="utf-8")


MODEL = "Llama-3.2-3B-Instruct-8bit"


def chat(client, user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful personal assistant."},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""


def loop():
    client = agent_harness.create_client()

    print(f"v0 assistant ({MODEL}) - ctrl+c to quit")

    try:
        while True:
            question = input("\nYou: ")
            print("\nBot: ", chat(client, question))

    except(EOFError, KeyboardInterrupt):
        print("\nbye!")


if __name__ == "__main__":
    loop()
