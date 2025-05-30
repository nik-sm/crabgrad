A simple and non-optimized reverse-mode automatic differentiation library, heavily inspired by https://github.com/karpathy/micrograd/.

To check for correctness against torch, there are unit tests that use [tch-rs](https://github.com/LaurentMazare/tch-rs) and scripts that use [pytorch](https://github.com/pytorch/pytorch).

# Usage

Typical cargo commands are used to lint, check, compile, and run examples:

```shell
cargo check  # static checks
cargo test  # run tests
cargo clippy  # lints
cargo bench --profile release-lto  # run benchmarks before & after changes
cargo build --profile release-lto

cargo run --example mlp  # Example end-to-end library usage
cargo flamegraph --root --example mlp --profile release-lto  # View output SVG in firefox
```

## Comparison against Pytorch

Re-create frozen requirements after adding python dependencies:
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r pytorch-examples/requirements.in
pip freeze > pytorch-examples/requirements.txt
```

Or create environment from existing frozen requirements:
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r pytorch-examples/requirements.txt
```

Run scripts:
```shell
python pytorch-examples/losses.py
cargo run --example losses

python pytorch-examples/unit-vector.py
cargo run --example unit-vector

python pytorch-examples/mlp.py
cargo run --example mlp --profile release-lto

python pytorch-examples/mnist.py
cargo run --example mnist --profile release-lto
```

Note that MNIST performance is worse than pytorch, possibly because:
- Weight init methods are definitely different and this can be important
- AdamW implementation here may be buggy

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
    - Deduplicate and use macros for boilerplate impl blocks
    - More convenience ops: +=, -=, *=, /=, unary negation

- Consider simple opportunities for speedup
    - Data parallelism in trainer: copy model and items to each worker, forward, accumulate, backward.
      A bit invasive - would probably need to change `Value` from being `Rc<RefCell<ValueInner>>` to `Arc<Mutex<ValueInner>>` or `Arc<RwLock<ValueInner>>`

- When doing `Module::score()`, add progress bar to avoid long silence and also parallelize
    - Add a `no_grad` mode
