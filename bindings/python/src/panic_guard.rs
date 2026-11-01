//! Suppress panic payloads only inside this extension's guarded operations.

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Once;

thread_local! {
    static GUARDED: Cell<usize> = const { Cell::new(0) };
}

/// Install once during module initialization while Python still holds the GIL.
/// Other Rust code keeps the host's prior hook outside our guarded sections.
pub(crate) fn install() {
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !GUARDED.with(|depth| depth.get() > 0) {
                previous(info);
            }
        }));
    });
}

struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        GUARDED.with(|depth| depth.set(depth.get() - 1));
    }
}

pub(crate) fn catch<T>(run: impl FnOnce() -> T) -> std::thread::Result<T> {
    GUARDED.with(|depth| depth.set(depth.get() + 1));
    let guard = Guard;
    let result = catch_unwind(AssertUnwindSafe(run));
    drop(guard);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarded_payload_is_private_and_the_host_hook_still_runs_elsewhere() {
        const CHILD: &str = "COFFER_PANIC_HOOK_CHILD";
        if std::env::var_os(CHILD).is_some() {
            install();
            assert!(catch(|| panic!("sensitive-panic-payload")).is_err());
            assert!(std::panic::catch_unwind(|| panic!("host-hook-marker")).is_err());
            return;
        }
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "panic_guard::tests::guarded_payload_is_private_and_the_host_hook_still_runs_elsewhere", "--nocapture"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(result.status.success());
        let stderr = String::from_utf8(result.stderr).unwrap();
        assert!(!stderr.contains("sensitive-panic-payload"));
        assert!(stderr.contains("host-hook-marker"));
    }
}
