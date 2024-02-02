import tensorflow as tf


def get_alexnet(
    image_size: tuple[int, int], learning_rate: float = 1e-4
) -> tf.keras.models.Model:
    def conv(
        filters: int, kernel_size: int, padding: str = "same", strides: int = 1
    ) -> tf.keras.layers.Conv2D:
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation="relu",
        )

    def pooling() -> tf.keras.layers.MaxPooling2D:
        return tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)

    def dropout(rate: float = 0.5) -> tf.keras.layers.Dropout:
        return tf.keras.layers.Dropout(rate)

    def dense(units: int, activation: str = "relu") -> tf.keras.layers.Dense:
        return tf.keras.layers.Dense(units, activation=activation)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer((*image_size, 3)),
            conv(96, 11, "valid", 4),
            pooling(),
            conv(256, 5),
            pooling(),
            conv(384, 3),
            conv(384, 3),
            conv(384, 3),
            pooling(),
            tf.keras.layers.Flatten(),
            dense(4096),
            dropout(),
            dense(4096),
            dropout(),
            dense(6, activation="softmax"),
        ],
        name="alex_net",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
