# TODO

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

# Setup

To compare against torch, there are unit tests that use [tch-rs](https://github.com/LaurentMazare/tch-rs) and scripts that use [pytorch](https://github.com/pytorch/pytorch).

Installing pytorch enables both comparisons, since `tch-rs` requires access to libtorch and this is most easily provided by having pytorch installed into the currently active python environment.

## Comparison against Pytorch

Create frozen requirements:
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r pytorch-examples/requirements.in
pip freeze > pytorch-examples/requirements.txt
```

Create environment from frozen requirements:
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r pytorch-examples/requirements.txt
```

## Comparison against tch

To compare against torch using `tch-rs`:
```shell
VENV_PATH="$(pwd)/venv"
PYTHON_VERSION=3.13

source $VENV_PATH/bin/activate
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH=$VENV_PATH/lib/$PYTHON_VERSION/site-packages/torch/lib:$DYLD_LIBRARY_PATH
cargo clean
cargo test
```

# Usage

```shell
cargo clean
cargo check
cargo test
cargo clippy

cargo run mlp  # Example end-to-end library usage
cargo flamegraph --root --bin mlp   # View output SVG in firefox
```