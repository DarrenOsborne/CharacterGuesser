import argparse
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','c','d','e','f','g','h','i','j',
    'k','l','m','n','o','p','q','r','s','t',
    'u','v','w','x','y','z'
]


def fix_emnist_orientation(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.transpose(image)
    image = tf.image.flip_left_right(image)
    return image


def make_test_datasets(limit_digits: int | None = None,
                       limit_letters: int | None = None,
                       batch_size: int = 256) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    autotune = tf.data.AUTOTUNE

    ds_digits = tfds.load('emnist/digits', split='test', as_supervised=True)
    ds_letters = tfds.load('emnist/letters', split='test', as_supervised=True)

    def prep_digits(x, y):
        x = tf.cast(x, tf.float32)
        x = fix_emnist_orientation(x) / 255.0
        y = tf.cast(y, tf.int32)
        return x, y

    def prep_letters(x, y):
        x = tf.cast(x, tf.float32)
        x = fix_emnist_orientation(x) / 255.0
        y = tf.cast(y, tf.int32)
        y = (y - 1) + 10  # map 1..26 -> 10..35
        return x, y

    ds_digits = ds_digits.map(prep_digits, num_parallel_calls=autotune)
    ds_letters = ds_letters.map(prep_letters, num_parallel_calls=autotune)
    if limit_digits:
        ds_digits = ds_digits.take(int(limit_digits))
    if limit_letters:
        ds_letters = ds_letters.take(int(limit_letters))
    return (
        ds_digits.batch(batch_size).prefetch(autotune),
        ds_letters.batch(batch_size).prefetch(autotune),
    )


def evaluate_split(model: tf.keras.Model, ds: tf.data.Dataset) -> Tuple[float, np.ndarray, np.ndarray]:
    total = 0
    correct = 0
    y_true_all = []
    y_pred_all = []
    for xb, yb in ds:
        logits = model(xb, training=False)
        preds = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
        correct += int(tf.reduce_sum(tf.cast(preds == yb, tf.int32)).numpy())
        total += int(yb.shape[0])
        y_true_all.append(yb.numpy())
        y_pred_all.append(preds.numpy())
    acc = correct / max(1, total)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int32)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int32)
    return acc, y_true, y_pred


def count_confusions(y_true: np.ndarray, y_pred: np.ndarray, a: str, b: str) -> str:
    ia = LABELS.index(a)
    ib = LABELS.index(b)
    a_as_b = int(((y_true == ia) & (y_pred == ib)).sum())
    b_as_a = int(((y_true == ib) & (y_pred == ia)).sum())
    return f"{a}->{b}: {a_as_b}, {b}->{a}: {b_as_a}"


def main(model_path: str, batch_size: int, limit_digits: int | None, limit_letters: int | None):
    model = tf.keras.models.load_model(model_path)

    ds_digits, ds_letters = make_test_datasets(limit_digits, limit_letters, batch_size)
    acc_d, ytd, ypd = evaluate_split(model, ds_digits)
    acc_l, ytl, ypl = evaluate_split(model, ds_letters)

    print(f"Digits test accuracy:  {acc_d:.4f}")
    print(f"Letters test accuracy: {acc_l:.4f}")
    if ytd.size and ytl.size:
        y_true = np.concatenate([ytd, ytl])
        y_pred = np.concatenate([ypd, ypl])
        print("Confusions (0 vs o):", count_confusions(y_true, y_pred, '0', 'o'))
        print("Confusions (b vs 6):", count_confusions(y_true, y_pred, 'b', '6'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick eval of EMNIST model per split and key confusions')
    parser.add_argument('-m', '--model', default=os.path.join('artifacts', 'alnum36_cnn.keras'))
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('--limit-digits', type=int, default=None)
    parser.add_argument('--limit-letters', type=int, default=None)
    args = parser.parse_args()
    main(args.model, args.batch_size, args.limit_digits, args.limit_letters)

