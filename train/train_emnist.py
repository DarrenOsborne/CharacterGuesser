import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


NUM_CLASSES = 36  # 0-9 + a-z


def fix_emnist_orientation(image: tf.Tensor) -> tf.Tensor:
    # EMNIST images are rotated; correct by transpose + horizontal flip
    image = tf.image.transpose(image)
    image = tf.image.flip_left_right(image)
    return image


def make_dataset(batch_size: int, limit_train: int | None = None, limit_test: int | None = None, cache: bool = False):
    # Digits: labels 0-9 -> keep as 0..9
    ds_digits_train = tfds.load('emnist/digits', split='train', as_supervised=True)
    ds_digits_test = tfds.load('emnist/digits', split='test', as_supervised=True)

    # Letters: labels are 1..26, map to 10..35 by subtracting 1 then adding 10
    ds_letters_train = tfds.load('emnist/letters', split='train', as_supervised=True)
    ds_letters_test = tfds.load('emnist/letters', split='test', as_supervised=True)

    def prep_digits(x, y):
        x = tf.cast(x, tf.float32)
        x = fix_emnist_orientation(x)
        x = x / 255.0
        # EMNIST images already have a channel dim; do NOT expand dims
        y = tf.cast(y, tf.int32)
        return x, y  # y in [0..9]

    def prep_letters(x, y):
        x = tf.cast(x, tf.float32)
        x = fix_emnist_orientation(x)
        x = x / 255.0
        # EMNIST letters labels are 1..26 -> map to 10..35
        y = tf.cast(y, tf.int32)
        y = (y - 1) + 10
        return x, y

    autotune = tf.data.AUTOTUNE
    train = (
        ds_digits_train.map(prep_digits, num_parallel_calls=autotune)
        .concatenate(ds_letters_train.map(prep_letters, num_parallel_calls=autotune))
    )
    if limit_train:
        train = train.take(int(limit_train))
    if cache:
        train = train.cache()
    train = train.shuffle(10000).batch(batch_size).prefetch(autotune)

    test = (
        ds_digits_test.map(prep_digits, num_parallel_calls=autotune)
        .concatenate(ds_letters_test.map(prep_letters, num_parallel_calls=autotune))
    )
    if limit_test:
        test = test.take(int(limit_test))
    if cache:
        test = test.cache()
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


def main(epochs: int, batch_size: int, limit_train: int | None, limit_test: int | None, cache: bool, verbose: int):
    train, test = make_dataset(batch_size, limit_train=limit_train, limit_test=limit_test, cache=cache)
    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy')
    ]

    # Show dataset sizes to make progress expectations clear
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
    model.save('artifacts/alnum36_cnn.keras')

    # Export to TF.js
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    os.makedirs(out_dir, exist_ok=True)
    from export_tfjs import main as export_main  # use our safe exporter
    export_main('artifacts/alnum36_cnn.keras', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EMNIST 0-9 + a-z model and export to TF.js')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--limit-train', type=int, default=None, help='Limit number of train examples (for quick runs)')
    parser.add_argument('--limit-test', type=int, default=None, help='Limit number of test examples (for quick runs)')
    parser.add_argument('--cache', action='store_true', help='Cache datasets in memory (uses more RAM)')
    parser.add_argument('--verbose', type=int, default=1, help='Keras fit verbosity (0,1,2)')
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.limit_train, args.limit_test, args.cache, args.verbose)
