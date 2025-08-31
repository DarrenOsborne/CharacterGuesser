import argparse
import os
import sys
import types


def _ensure_stubs():
    """Ensure problematic optional deps don't break TF.js conversion.

    - tensorflow_decision_forests: stub to avoid missing native ops
    - jax/jax2tf.shape_poly: ensure importable regardless of installed JAX
    """
    # Always stub TF-DF early
    sys.modules['tensorflow_decision_forests'] = sys.modules.get(
        'tensorflow_decision_forests', types.ModuleType('tensorflow_decision_forests')
    )

    # Ensure JAX shape_poly import works even if real JAX is present
    try:
        import importlib  # noqa: F401
        import jax  # type: ignore
        # Ensure experimental module exists
        if not hasattr(jax, 'experimental'):
            exp_mod = types.ModuleType('jax.experimental')
            setattr(jax, 'experimental', exp_mod)
            sys.modules['jax.experimental'] = exp_mod
        else:
            exp_mod = jax.experimental  # type: ignore
        # Ensure jax2tf exists
        try:
            import jax.experimental.jax2tf as jax2tf  # type: ignore
        except Exception:
            jax2tf = types.ModuleType('jax.experimental.jax2tf')
            sys.modules['jax.experimental.jax2tf'] = jax2tf
            setattr(exp_mod, 'jax2tf', jax2tf)
        # Ensure shape_poly exists with PolyShape symbol
        try:
            import jax.experimental.jax2tf.shape_poly as sp  # type: ignore
            if not hasattr(sp, 'PolyShape'):
                class PolyShape:  # type: ignore
                    pass
                sp.PolyShape = PolyShape  # type: ignore
        except Exception:
            shape_poly_mod = types.ModuleType('jax.experimental.jax2tf.shape_poly')
            class PolyShape:  # type: ignore
                pass
            shape_poly_mod.PolyShape = PolyShape
            sys.modules['jax.experimental.jax2tf.shape_poly'] = shape_poly_mod
            # expose as attribute on jax2tf module
            setattr(jax2tf, 'shape_poly', shape_poly_mod)
    except Exception:
        # No JAX installed or import failed; create minimal stub tree
        jax_mod = types.ModuleType('jax')
        exp_mod = types.ModuleType('jax.experimental')
        jax2tf_mod = types.ModuleType('jax.experimental.jax2tf')
        shape_poly_mod = types.ModuleType('jax.experimental.jax2tf.shape_poly')
        class PolyShape:  # type: ignore
            pass
        shape_poly_mod.PolyShape = PolyShape
        jax2tf_mod.shape_poly = shape_poly_mod
        exp_mod.jax2tf = jax2tf_mod
        jax_mod.experimental = exp_mod
        sys.modules['jax'] = jax_mod
        sys.modules['jax.experimental'] = exp_mod
        sys.modules['jax.experimental.jax2tf'] = jax2tf_mod
        sys.modules['jax.experimental.jax2tf.shape_poly'] = shape_poly_mod
    # jaxlib sometimes imported alongside jax; provide empty stub
    sys.modules['jaxlib'] = sys.modules.get('jaxlib', types.ModuleType('jaxlib'))


def main(in_model: str, out_dir: str):
    _ensure_stubs()
    # Import after stubbing/patching
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
