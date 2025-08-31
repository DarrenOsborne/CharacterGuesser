import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


NUM_CLASSES = 26  # a-z


def fix_emnist_orientation(image: tf.Tensor) -> tf.Tensor:
    # EMNIST images are rotated; correct by transpose + horizontal flip
    image = tf.image.transpose(image)
    image = tf.image.flip_left_right(image)
    return image


def make_dataset(batch_size: int, cache: bool = False):
    ds_train = tfds.load('emnist/letters', split='train', as_supervised=True)
    ds_test = tfds.load('emnist/letters', split='test', as_supervised=True)

    def prep(x, y):
        x = tf.cast(x, tf.float32)
        x = fix_emnist_orientation(x)
        x = x / 255.0  # [28,28,1]
        # EMNIST letters labels are 1..26 -> map to 0..25
        y = tf.cast(y, tf.int32)
        y = (y - 1)
        return x, y

    autotune = tf.data.AUTOTUNE

    train = ds_train.map(prep, num_parallel_calls=autotune)
    test = ds_test.map(prep, num_parallel_calls=autotune)
    if cache:
        train = train.cache()
        test = test.cache()
    train = train.shuffle(20000).batch(batch_size).prefetch(autotune)
    test = test.batch(batch_size).prefetch(autotune)
    return train, test


def build_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main(epochs: int, batch_size: int, cache: bool, verbose: int):
    train, test = make_dataset(batch_size, cache=cache)
    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ]

    try:
        train_card = int(tf.data.experimental.cardinality(train).numpy())
        test_card = int(tf.data.experimental.cardinality(test).numpy())
        print(f"Batches per epoch: train={train_card}, val={test_card}")
    except Exception:
        pass

    model.fit(
        train,
        epochs=epochs,
        validation_data=test,
        callbacks=callbacks,
        verbose=verbose,
    )

    test_loss, test_acc = model.evaluate(test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    os.makedirs('artifacts', exist_ok=True)
    model_path = 'artifacts/letters26_cnn.keras'
    model.save(model_path)

    # Export to TF.js (reuse our safe exporter)
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    from export_tfjs import main as export_main
    export_main(model_path, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EMNIST lowercase letters (a-z) and export to TF.js')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.cache, args.verbose)

