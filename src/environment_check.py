import sys
import torch
import numpy
import pandas
import sklearn
import transformers
import sentence_transformers
import bertopic
import spacy

from config import load_config


def main():
    config, model_config = load_config()

    print("=" * 60)
    print("REDDIT NARRATIVE DECODER")
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    print("Python:", sys.version.split()[0])
    print("Executable:", sys.executable)

    print()
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA runtime:", torch.version.cuda)

    print()
    print("NumPy:", numpy.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("Transformers:", transformers.__version__)
    print("Sentence Transformers:", sentence_transformers.__version__)
    print("BERTopic:", bertopic.__version__)
    print("spaCy:", spacy.__version__)

    print()
    print("Project:", config["project"]["name"])
    print("Version:", config["project"]["version"])
    print("Configured device:", config["hardware"]["device"])
    print("GPU enabled:", config["hardware"]["gpu_enabled"])
    print("Embedding model:", model_config["embedding"]["model_name"])

    print()
    print("=" * 60)
    print("ENVIRONMENT CHECK: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
