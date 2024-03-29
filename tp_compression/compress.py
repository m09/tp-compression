from pathlib import Path
from typing import Any, Iterator

import numpy
import tensorflow as tf

from dvclive import Live  # type: ignore

from .data import get_images


def _evaluate_tflite_model(
    tflite_model: Any, test_images: numpy.ndarray, test_labels: numpy.ndarray
) -> float:
    # Initialize TFLite interpreter using the model.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for test_image in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = numpy.expand_dims(test_image, axis=0)
        interpreter.set_tensor(input_tensor_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        digit = numpy.argmax(output()[0])
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_labels[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_digits)

    return accuracy


def compress(
    train_data_dir: str,
    test_data_dir: str,
    model_path: str,
    tflite_model_path: str,
    image_size: tuple[int, int],
) -> None:
    model = tf.keras.models.load_model(model_path)

    def representative_dataset_gen() -> Iterator[list[numpy.ndarray]]:
        images, labels, paths = get_images(Path(train_data_dir), image_size)
        images = images[numpy.random.choice(images.shape[0], size=1_000, replace=False)]
        for i in range(images.shape[0]):
            # Get sample input data as a numpy array in a method of your choosing.
            yield [images[[i]]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()
    Path(tflite_model_path).write_bytes(tflite_quant_model)
    test_images, test_labels, _test_paths = get_images(Path(test_data_dir), image_size)
    test_acc = _evaluate_tflite_model(tflite_quant_model, test_images, test_labels)
    with Live("dvclive/compress") as live:
        live.log_artifact(tflite_model_path, "model", "landscape_classifier_lite")
        live.log_metric("test_acc", test_acc)
