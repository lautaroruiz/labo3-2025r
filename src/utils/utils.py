from pathlib import Path


def get_base_dir():
    """Get the base directory of the project."""
    return Path(__file__).resolve().parent.parent.parent


if __name__ == "__main__":
    print(get_base_dir())
