# TODO

- Test by comparing results to torch:
    ```shell
    VENV_PATH=~/.venv
    PYTHON_VERSION=3.13
    source $VENV_PATH/bin/activate
    export LIBTORCH_USE_PYTORCH=1
    export DYLD_LIBRARY_PATH=$VENV_PATH/lib/$PYTHON_VERSION/site-packages/torch/lib:$DYLD_LIBRARY_PATH
    cargo clean
    cargo test
    ```

- Finish models:
    - SGD optimizer
    - loss
- Try on toy dataset task

- Engineering:
    - deduplicate
    - if possible: avoid annoying `.clone()` by implementing ops (`+`, etc) without move. unclear if copy semantics works for `Rc<RefCell<_>>`

