# TODO

Next steps:
- Engineering and ergonomics:
    - Be sure all ops can be used on literals of `i64` or `f64`, and add tests
    - Deduplicate

- More ops:
    - Implement: +=, -=, *=, /=, unary negation

- Try on toy dataset (MNIST, etc)

# Setup

To compare against torch, there are unit tests that use [tch-rs](https://github.com/LaurentMazare/tch-rs) and scripts that use [pytorch](https://github.com/pytorch/pytorch).

# Usage

```shell
cargo clean
cargo check
cargo test
cargo clippy

cargo run mlp  # Example end-to-end library usage
cargo flamegraph --root --bin mlp   # View output SVG in firefox
```

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

`tch` requires libtorch.

A simple way is to install using: `cargo add tch --features download-libtorch`

Alternatively, a python installation with pytorch installed can also be used:
```shell
VENV_PATH="$(pwd)/venv"
PYTHON_VERSION=3.13

source $VENV_PATH/bin/activate
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH=$VENV_PATH/lib/$PYTHON_VERSION/site-packages/torch/lib:$DYLD_LIBRARY_PATH
cargo clean
cargo test
```

