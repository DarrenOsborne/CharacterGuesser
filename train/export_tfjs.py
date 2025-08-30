import argparse
import os
import sys
import types


def main(in_model: str, out_dir: str):
    # Work around tensorflowjs importing packages we don't need for Keras export.
    # 1) tensorflow_decision_forests native ops may be missing on Windows.
    if 'tensorflow_decision_forests' not in sys.modules:
        sys.modules['tensorflow_decision_forests'] = types.ModuleType('tensorflow_decision_forests')

    # 2) Some tensorflowjs versions import jax.experimental.jax2tf.shape_poly unconditionally.
    #    Provide a minimal stub so import succeeds.
    if 'jax' not in sys.modules:
        jax_mod = types.ModuleType('jax')
        exp_mod = types.ModuleType('jax.experimental')
        jax2tf_mod = types.ModuleType('jax.experimental.jax2tf')
        # Create a minimal shape_poly module with PolyShape symbol
        shape_poly_mod = types.ModuleType('jax.experimental.jax2tf.shape_poly')
        class PolyShape:  # type: ignore
            pass
        shape_poly_mod.PolyShape = PolyShape
        # Attach to jax2tf
        jax2tf_mod.shape_poly = shape_poly_mod
        # Link submodules
        exp_mod.jax2tf = jax2tf_mod
        jax_mod.experimental = exp_mod
        # Register modules
        sys.modules['jax'] = jax_mod
        sys.modules['jax.experimental'] = exp_mod
        sys.modules['jax.experimental.jax2tf'] = jax2tf_mod
        sys.modules['jax.experimental.jax2tf.shape_poly'] = shape_poly_mod
    # jaxlib sometimes imported alongside jax; provide empty stub
    if 'jaxlib' not in sys.modules:
        sys.modules['jaxlib'] = types.ModuleType('jaxlib')

    # Import after stubbing
    import tensorflow as tf  # noqa: E402
    from tensorflowjs.converters import save_keras_model  # noqa: E402

    model = tf.keras.models.load_model(in_model)
    os.makedirs(out_dir, exist_ok=True)
    save_keras_model(model, out_dir)
    print(f"Saved TF.js model to: {os.path.abspath(out_dir)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a Keras model to TF.js format (Windows-safe).')
    parser.add_argument('-i', '--input', default=os.path.join('artifacts', 'mnist_cnn.keras'), help='Path to Keras model (.keras or .h5)')
    parser.add_argument('-o', '--output', default=os.path.join('..', 'model'), help='Output directory for TF.js model')
    args = parser.parse_args()
    main(args.input, args.output)
