use anyhow::Result;
use chrono::Local;
use colored::Colorize;
use env_logger::{Builder, Env};
use log::Level;
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
            let level_str = format!("[{}]", level.to_string());
            let level_final = format!("{L:<7}", L = level_str).truecolor(level_color.0, level_color.1, level_color.2);

            let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string().truecolor(120, 120, 120);
            // let target = record.target().to_string().truecolor(147, 182, 182);

            let path = record.file_static().unwrap_or("unknown").to_string().truecolor(200, 200, 200);
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
        assert_close!($left, $right, $crate::util::DEFAULT_RTOL, $crate::util::DEFAULT_ATOL)
    };
    ($left:expr, $right:expr, $rtol:expr, $atol:expr) => {
        assert!(
            $crate::util::is_close($left, $right, $rtol, $atol),
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
        assert_not_close!($left, $right, $crate::util::DEFAULT_RTOL, $crate::util::DEFAULT_ATOL)
    };
    ($left:expr, $right:expr, $rtol:expr, $atol:expr) => {
        assert!(
            !$crate::util::is_close($left, $right, $rtol, $atol),
            "assertion failed: `is_close(left, right)`\n  left: `{:?}`\n right: `{:?}`\n rtol: `{:?}`\n atol: `{:?}`",
            $left,
            $right,
            $rtol,
            $atol
        )
    };
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
    }

    #[test]
    fn not_close1() {
        assert_not_close!(1.0, 2.0);
    }
}
