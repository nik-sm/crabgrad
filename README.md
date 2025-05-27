A simple and non-optimized backward-mode automatic differentiation library, heavily inspired by https://github.com/karpathy/micrograd/.

To check for correctness against torch, there are unit tests that use [tch-rs](https://github.com/LaurentMazare/tch-rs) and scripts that use [pytorch](https://github.com/pytorch/pytorch).

# Usage

Typical cargo commands are used to lint, check, compile, and run examples:

```shell
cargo check  # static checks
cargo test  # run tests
cargo clippy  # lints
cargo bench  # run benchmarks before & after changes

cargo run --example mlp  # Example end-to-end library usage
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

Run scripts:
```shell
python pytorch-examples/mlp.py
cargo run --example mlp

python pytorch-examples/losses.py
cargo run --example losses

python pytorch-examples/unit-vector.py
cargo run --example unit-vector

# NOTE - performance after 1 epoch is worse in ours, possibly because:
# - Weight init methods are definitely different and this can be important
# - AdamW implementation may be buggy
python pytorch-examples/mnist.py
cargo run --example mnist
```

## Comparison against tch

`tch` requires libtorch. A simple way is to install using: `cargo add tch --features download-libtorch`

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

# TODO

- Engineering and ergonomics:
    - Be sure all ops can be used on literals of `i64` or `f64`, and add tests
    - Deduplicate
    - More convenience ops: +=, -=, *=, /=, unary negation

- Parallelize trainer (inside batch, synchronize before step)

- When doing `Module::score()`, add progress bar and also parallelize
