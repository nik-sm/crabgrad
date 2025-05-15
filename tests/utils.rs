pub fn is_close(a: f32, b: f32, rtol: f32, atol: f32) -> bool {
    let close = (a - b).abs() < (atol + rtol * b.abs());
    let finite = a.is_finite() && b.is_finite();
    let perfect_equal = a == b;
    let mut result = close && finite || perfect_equal;
    // If both are NaN, consider them equal
    result |= a.is_nan() & b.is_nan();
    result
}

#[macro_export]
macro_rules! assert_close {
    ($left:expr, $right:expr) => {
        assert_close!($left, $right, 1e-5, 1e-8)
    };
    ($left:expr, $right:expr, $rtol:expr, $atol:expr) => {
        if !utils::is_close($left, $right, $rtol, $atol) {
            panic!("assertion failed: `is_close(left, right)`\n  left: `{:?}`\n right: `{:?}`\n rtol: `{:?}`\n atol: `{:?}`", $left, $right, $rtol, $atol)
        }
    };
}

#[macro_export]
macro_rules! assert_not_close {
    ($left:expr, $right:expr) => {
        assert_not_close!($left, $right, 1e-5, 1e-8)
    };
    ($left:expr, $right:expr, $rtol:expr, $atol:expr) => {
        if utils::is_close($left, $right, $rtol, $atol) {
            panic!("assertion failed: `is_close(left, right)`\n  left: `{:?}`\n right: `{:?}`\n rtol: `{:?}`\n atol: `{:?}`", $left, $right, $rtol, $atol)
        }
    };
}
