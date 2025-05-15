use crate::engine::{Dataset, DiscreteLabel, FloatDataScalar};
use anyhow::Result;
use chrono::Local;
use colored::Colorize;
use env_logger::{Builder, Env};
use itertools::Itertools;
use log::Level;
use rand;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal};
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn get_workspace_dir() -> PathBuf {
    let output = std::process::Command::new(env!("CARGO"))
        .arg("locate-project")
        .arg("--workspace")
        .arg("--message-format=plain")
        .output()
        .unwrap()
        .stdout;
    let cargo_path = Path::new(std::str::from_utf8(&output).unwrap().trim());
    cargo_path.parent().unwrap().to_path_buf()
}

pub fn try_init_logging() -> Result<()> {
    let env = Env::default().filter_or("RUST_LOG", "debug").write_style_or("LOG_STYLE", "always");

    Builder::from_env(env)
        .format(|buf, record| {
            let level = record.level();
            let level_color = match level {
                Level::Trace => (0, 220, 220),
                Level::Debug => (0, 120, 220),
                Level::Info => (0, 220, 120),
                Level::Warn => (255, 209, 102),
                Level::Error => (240, 80, 120),
            };
            // A bit wasteful, but looks nice: string interpolate, then add brackets, then pad, then colorize
            let level_str = format!("[{}]", level);
            let level_final = format!("{L:<7}", L = level_str).truecolor(level_color.0, level_color.1, level_color.2);

            let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string().truecolor(120, 120, 120);
            // let target = record.target().to_string().truecolor(147, 182, 182);

            // let path = record.file().unwrap_or("unknown").to_string().truecolor(200, 200, 200);
            let path = record.module_path().unwrap_or("unknown").to_string().truecolor(200, 200, 200);
            let lineno = record.line().unwrap_or(0).to_string().truecolor(200, 200, 200);

            let args = record.args();
            writeln!(buf, "{timestamp} {level_final} {path}:{lineno} | {args}")
        })
        .try_init()?;
    Ok(())
}

pub fn is_close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    let close = (a - b).abs() < (atol + rtol * b.abs());
    let finite = a.is_finite() && b.is_finite();
    let perfect_equal = a == b;
    let mut result = close && finite || perfect_equal;
    // If both are NaN, consider them equal
    result |= a.is_nan() & b.is_nan();
    result
}

pub const DEFAULT_RTOL: f64 = 1e-5;
pub const DEFAULT_ATOL: f64 = 1e-8;

#[macro_export]
macro_rules! assert_close {
    ($left:expr, $right:expr) => {
        assert_close!($left, $right, $crate::utils::DEFAULT_RTOL, $crate::utils::DEFAULT_ATOL)
    };
    ($left:expr, $right:expr, $rtol:expr, $atol:expr) => {
        assert!(
            $crate::utils::is_close($left, $right, $rtol, $atol),
            "assertion failed: `is_close(left, right)`\n  left: `{:?}`\n right: `{:?}`\n rtol: `{:?}`\n atol: `{:?}`",
            $left,
            $right,
            $rtol,
            $atol
        )
    };
}

#[macro_export]
macro_rules! assert_not_close {
    ($left:expr, $right:expr) => {
        assert_not_close!($left, $right, $crate::utils::DEFAULT_RTOL, $crate::utils::DEFAULT_ATOL)
    };
    ($left:expr, $right:expr, $rtol:expr, $atol:expr) => {
        assert!(
            !$crate::utils::is_close($left, $right, $rtol, $atol),
            "assertion failed: `is_close(left, right)`\n  left: `{:?}`\n right: `{:?}`\n rtol: `{:?}`\n atol: `{:?}`",
            $left,
            $right,
            $rtol,
            $atol
        )
    };
}

#[macro_export]
macro_rules! assert_vec_close {
    ($left:expr, $right:expr) => {
        $left.iter().zip($right).for_each(|(l, r)| assert_close!(l.data(), r.data()))
    };
    ($left:expr, $right:expr, $rtol:expr, $atol:expr) => {
        $left.iter().zip($right).for_each(|(l, r)| assert_close!(l.data(), r.data(), $rtol, $atol))
    };
}

pub fn make_binary_classification(
    n_samples_each_class: usize,
    n_features: usize,
) -> (Vec<Vec<FloatDataScalar>>, Vec<DiscreteLabel>) {
    let mut rng = rand::rng();
    let mut data: Vec<Vec<FloatDataScalar>> = Vec::with_capacity(2 * n_samples_each_class * n_features);
    let mut labels: Vec<DiscreteLabel> = Vec::with_capacity(2 * n_samples_each_class);

    // NOTE - simple approach; spherical gaussians aligned on the diagonal line y = x1 + x2 + x3 + ...

    // class 0
    let normal = Normal::new(0.0, 1.0).unwrap();

    data.extend(
        normal
            .sample_iter(&mut rng)
            .take(n_samples_each_class * n_features)
            .chunks(n_features)
            .into_iter()
            .map(|chunk| chunk.collect())
            .collect::<Vec<Vec<_>>>(),
    );
    labels.extend(vec![0; n_samples_each_class]);

    // class 1
    let normal = Normal::new(2.0, 1.0).unwrap();
    data.extend(
        normal
            .sample_iter(&mut rng)
            .take(n_samples_each_class * n_features)
            .chunks(n_features)
            .into_iter()
            .map(|chunk| chunk.collect())
            .collect::<Vec<Vec<_>>>(),
    );
    labels.extend(vec![1; n_samples_each_class]);

    (data, labels)
}

pub fn train_test_split<X: Clone, Y: Clone>(
    mut zipped_data_labels: Dataset<X, Y>,
    train_frac: f64,
    test_frac: f64,
) -> (Dataset<X, Y>, Dataset<X, Y>) {
    let mut rng = rand::rng();
    let n_total = zipped_data_labels.len() as f64;
    let n_train = (train_frac / (train_frac + test_frac) * n_total) as usize;

    zipped_data_labels.shuffle(&mut rng);
    let (train_part, test_part) = zipped_data_labels.split_at(n_train);
    (train_part.to_vec(), test_part.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_workspace_dir() {
        let workspace_dir = get_workspace_dir();
        println!("workspace_dir: {:?}", workspace_dir);
    }

    #[test]
    fn logging() {
        let res = try_init_logging();
        assert!(res.is_ok());

        log::trace!("A trace message");
        log::debug!("A debug message");
        log::info!("An info message");
        log::warn!("A warning message");
        log::error!("An error message");
    }

    #[test]
    fn close1() {
        assert_close!(1.00001, 1.00002);
        assert_close!(1.0 + 2.0, 3.0);
        assert_close!(1e-3, 1e-3 + 1e-10);
    }

    #[test]
    fn not_close1() {
        assert_not_close!(1.0, 2.0);

        assert_not_close!(1e-3, 2e-3);
        assert_not_close!(1e-3, 1e-3 + 1e-5);
    }

    #[test]
    fn toy_data() {
        let n_samples_each_class = 10;
        let n_features = 8;
        let (data, labels) = make_binary_classification(n_samples_each_class, n_features);

        assert_eq!(data.len(), 2 * n_samples_each_class, "n items");
        assert_eq!(data[0].len(), n_features, "n features");
        assert_eq!(labels.len(), 2 * n_samples_each_class);
    }
}
