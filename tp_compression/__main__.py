from sys import argv

from yaml import safe_load

from .compress import compress
from .train import train

if __name__ == "__main__":
    if len(argv) != 2:
        raise ValueError("The module main program requires exactly 1 argument")
    with open("params.yaml", encoding="utf8") as fh:
        config = safe_load(fh)
    if argv[1] == "train":
        train(
            config["train_data"],
            config["model_type"],
            config["image_size"],
            config["learning_rate"],
        )
    if argv[1] == "compress":
        compress(
            config["train_data"],
            config["test_data"],
            config["model_path"],
            config["tf_lite_model_path"],
            config["image_size"],
        )
