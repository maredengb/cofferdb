//! Suppress panic payloads only inside this extension's guarded operations.

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Once;

// A panic hook runs before catch_unwind returns. Mark only the OS thread currently
// crossing a guarded boundary; an unrelated Rust thread still needs its ordinary
// panic reporting.
thread_local! {
    static GUARDED: Cell<usize> = const { Cell::new(0) };
}

/// Install once during module initialization while Python still holds the GIL.
/// Other Rust code keeps the host's prior hook outside our guarded sections.
pub(crate) fn install() {
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        // A single installation avoids wrapping the same process-wide hook again
        // on later module initializations.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !GUARDED.with(|depth| depth.get() > 0) {
                previous(info);
            }
        }));
    });
}

// One guard balances one nesting increment. Its destructor restores the previous
// depth even when the guarded closure unwinds through its own local destructors.
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        GUARDED.with(|depth| depth.set(depth.get() - 1));
    }
}

pub(crate) fn catch<T>(run: impl FnOnce() -> T) -> std::thread::Result<T> {
    // Track nesting on this thread because an extension operation can enter another
    // guarded helper. The hook resumes forwarding panics to the host only when the
    // last guard on that thread leaves.
    GUARDED.with(|depth| depth.set(depth.get() + 1));
    let guard = Guard;
    // Catch the unwind without requiring every PyO3 or native value captured by
    // the caller to implement UnwindSafe. Callers determine how to quarantine an
    // affected owner or translate the panic after this function returns.
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
