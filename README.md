# Setup

- To enable comparison against torch, using `tch-rs`:
    ```shell
    VENV_PATH="$HOME/.venv"
    PYTHON_VERSION=3.13

    source $VENV_PATH/bin/activate
    export LIBTORCH_USE_PYTORCH=1
    export DYLD_LIBRARY_PATH=$VENV_PATH/lib/$PYTHON_VERSION/site-packages/torch/lib:$DYLD_LIBRARY_PATH
    cargo clean
    cargo test
    ```

Next steps:
- Engineering and ergonomics:
    - Avoid the need for using `value.clone()` everywhere, by implementing operator traits like `Add` for `Value, Value`, `Value, &Value`, `&Value, Value`, and `&Value, &Value`
    - Be sure all ops can be used on literals of `i64` or `f64`, and add tests
    - Deduplicate

- More ops:
    - Implement: +=, -=, *=, /=, unary negation

- Finish models:
    - SGD optimizer
    - loss

- Try on toy dataset task
