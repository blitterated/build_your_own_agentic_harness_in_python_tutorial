import subprocess
from openai import OpenAI
from openai import AsyncOpenAI


KEY_PATH = "API/oMLX/api_key"


def get_api_key() -> str:
    try:
        result = subprocess.run(
            ["gopass", "show", "-o", KEY_PATH],
            capture_output=True,
            text=True,
            check=True
        )

        api_key = result.stdout.strip()
        return api_key

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gopass failed for '{KEY_PATH}': {e.stderr.strip()}")


def create_client():
    return OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key=get_api_key()
    )


def create_async_client():
    return AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key=get_api_key()
    )
