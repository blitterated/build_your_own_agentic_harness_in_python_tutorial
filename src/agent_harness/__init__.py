import subprocess

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
