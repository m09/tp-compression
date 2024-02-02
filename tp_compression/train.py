from pathlib import Path
from typing import Literal

from dvclive.keras import DVCLiveCallback

from dvclive import Live  # type: ignore

from .data import get_images
from .models.alexnet import get_alexnet
from .models.lenet import get_lenet


def train(
    data_dir: str,
    model_type: Literal["alexnet", "lenet"],
    image_size: tuple[int, int],
    learning_rate: float,
) -> None:
    images, labels, paths = get_images(Path(data_dir), image_size)
    print(images.shape)
    model = (
        get_lenet(image_size, learning_rate)
        if model_type == "lenet"
        else get_alexnet(image_size, learning_rate)
    )
    with Live("dvclive/train") as live:
        model.fit(images, labels, 128, epochs=3, callbacks=[DVCLiveCallback(live=live)])
        model.save("landscape_classifier.keras")
        live.log_artifact("landscape_classifier.keras", "model", "landscape_classifier")
