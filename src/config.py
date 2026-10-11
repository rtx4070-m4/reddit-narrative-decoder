from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_config():
    config = load_yaml(CONFIG_DIR / "config.yaml")
    model_config = load_yaml(CONFIG_DIR / "model_config.yaml")
    return config, model_config


if __name__ == "__main__":
    config, model_config = load_config()

    print("Configuration loaded successfully.")
    print("Project:", config["project"]["name"])
    print("Version:", config["project"]["version"])
    print("Device:", config["hardware"]["device"])
    print("GPU enabled:", config["hardware"]["gpu_enabled"])
    print("Embedding model:", model_config["embedding"]["model_name"])
