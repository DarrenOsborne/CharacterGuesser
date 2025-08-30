import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main(epochs: int, batch_size: int):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Scale to [0,1] and add channel dim
    x_train = (x_train.astype('float32') / 255.0)[..., None]
    x_test = (x_test.astype('float32') / 255.0)[..., None]

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy')
    ]

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save Keras model (optional backup)
    os.makedirs('artifacts', exist_ok=True)
    model.save('artifacts/mnist_cnn.keras')

    # Export to TensorFlow.js
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    os.makedirs(out_dir, exist_ok=True)
    try:
        from tensorflowjs.converters import save_keras_model
    except Exception as e:
        raise RuntimeError(
            'tensorflowjs is required to export the model. Install with:\n'
            '  pip install tensorflowjs\n'
            'Then re-run this script.'
        ) from e

    save_keras_model(model, out_dir)
    print(f"Saved TF.js model to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST CNN and export to TF.js')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    main(args.epochs, args.batch_size)

