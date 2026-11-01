//! Process-generation isolation for the two native open-path registries.
//!
//! A fork child must not acquire a mutex or wait for initialization inherited
//! from a vanished thread. Published generations are deliberately retained:
//! replacing one in a child must never drop its parent's files or ledgers.

use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

struct Generation<T> {
    pid: u32,
    value: T,
}

pub(crate) struct ProcessLocal<T: Send + Sync + 'static> {
    generation: AtomicPtr<Generation<T>>,
}

impl<T: Default + Send + Sync + 'static> ProcessLocal<T> {
    pub(crate) const fn new() -> Self {
        Self {
            generation: AtomicPtr::new(ptr::null_mut()),
        }
    }

    pub(crate) fn get(&'static self) -> &'static T {
        self.get_for_pid(std::process::id())
    }

    fn get_for_pid(&'static self, pid: u32) -> &'static T {
        loop {
            let observed = self.generation.load(Ordering::Acquire);
            if !observed.is_null() {
                // SAFETY: published objects are fully initialized before the
                // release CAS and are never freed, even across a fork.
                let generation = unsafe { &*observed };
                if generation.pid == pid {
                    return &generation.value;
                }
            }
            let prepared = Box::into_raw(Box::new(Generation {
                pid,
                value: T::default(),
            }));
            match self.generation.compare_exchange(
                observed,
                prepared,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // SAFETY: this process just published the complete object;
                    // no thread will free it or any earlier generation.
                    return unsafe { &(*prepared).value };
                }
                Err(_) => {
                    // SAFETY: an unpublished, process-local allocation; no
                    // other thread could have borrowed this pointer.
                    drop(unsafe { Box::from_raw(prepared) });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier, Mutex};

    #[cfg(unix)]
    fn isolated(name: &str) -> bool {
        const ENV: &str = "COFFERDB_FORK_PROBE_CHILD";
        if std::env::var_os(ENV).is_some() {
            return false;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--test-threads=1", "--nocapture"])
            .env(ENV, "1")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        true
    }

    #[cfg(unix)]
    fn fork_probe(probe: impl FnOnce() -> bool) -> bool {
        use std::time::{Duration, Instant};
        extern "C" {
            fn fork() -> i32;
            fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
            fn kill(pid: i32, signal: i32) -> i32;
            fn _exit(status: i32) -> !;
        }
        let child = unsafe { fork() };
        if child == 0 {
            let successful = probe();
            // Never unwind or drop inherited objects in the fork child.
            unsafe { _exit(if successful { 0 } else { 1 }) };
        }
        if child < 0 {
            return false;
        }
        let started = Instant::now();
        let mut status = 0;
        loop {
            let result = unsafe { waitpid(child, &mut status, 1) };
            if result == child {
                return status == 0;
            }
            if result < 0 || started.elapsed() > Duration::from_secs(5) {
                unsafe {
                    kill(child, 9);
                    waitpid(child, &mut status, 0);
                }
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn one_generation_is_shared_by_concurrent_initializers() {
        static STATE: ProcessLocal<Mutex<Vec<usize>>> = ProcessLocal::new();
        let barrier = Arc::new(Barrier::new(9));
        let threads: Vec<_> = (0..8)
            .map(|n| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    let state = STATE.get();
                    state.lock().unwrap().push(n);
                    state as *const _ as usize
                })
            })
            .collect();
        barrier.wait();
        let addresses: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert!(addresses.iter().all(|p| *p == addresses[0]));
        assert_eq!(STATE.get().lock().unwrap().len(), 8);
    }

    #[test]
    fn new_generation_ignores_a_locked_previous_generation() {
        static STATE: ProcessLocal<Mutex<Vec<usize>>> = ProcessLocal::new();
        let old = STATE.get_for_pid(1);
        let held = old.lock().unwrap();
        let new = STATE.get_for_pid(2);
        assert!(!std::ptr::eq(old, new));
        assert!(new.try_lock().unwrap().is_empty());
        assert!(old.try_lock().is_err());
        drop(held);
    }

    #[cfg(unix)]
    #[test]
    fn fork_open_bypasses_locked_parent_registries() {
        if isolated("process_state::tests::fork_open_bypasses_locked_parent_registries") {
            return;
        }
        use crate::{Database, DatabaseConfig, ErrorCode};
        use std::sync::mpsc;
        use std::time::Duration;
        let directory =
            std::env::temp_dir().join(format!("cofferdb-fork-registry-{}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("parent.coffer");
        let writer = Database::create(&path, DatabaseConfig::default()).unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let holding = std::thread::spawn(move || {
            crate::lock::with_registry_locked_for_test(|| {
                crate::physical_file::with_registry_locked_for_test(|| {
                    ready_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                });
            });
        });
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        // The probe is isolated from other engine tests. Only our helper thread
        // exists at this boundary, holding both registries across the fork.
        let child_ok = fork_probe(|| {
            (|| {
                let reader = Database::open_read_only(&path).ok()?;
                reader.stats().ok()?;
                reader.close().ok()?;
                match Database::open(&path) {
                    Err(error) if error.code() == ErrorCode::LockContention => {}
                    _ => return None,
                }
                let other =
                    Database::create(directory.join("child.coffer"), DatabaseConfig::default())
                        .ok()?;
                other.close().ok()?;
                Some(())
            })()
            .is_some()
        });
        release_tx.send(()).unwrap();
        holding.join().unwrap();
        assert!(
            child_ok,
            "fork child failed or waited for a parent registry"
        );
        writer.stats().unwrap();
        assert!(
            matches!(Database::open(&path), Err(error) if error.code() == ErrorCode::LockContention)
        );
        writer.close().unwrap();
        Database::open(&path).unwrap().close().unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn fork_during_unpublished_initialization_does_not_wait() {
        if isolated("process_state::tests::fork_during_unpublished_initialization_does_not_wait") {
            return;
        }
        use std::sync::atomic::{AtomicBool, AtomicU32};
        use std::time::{Duration, Instant};
        static PARENT: AtomicU32 = AtomicU32::new(0);
        static STARTED: AtomicBool = AtomicBool::new(false);
        static RELEASE: AtomicBool = AtomicBool::new(false);
        struct Initializing(u32);
        impl Default for Initializing {
            fn default() -> Self {
                let pid = std::process::id();
                if pid == PARENT.load(Ordering::Acquire) {
                    STARTED.store(true, Ordering::Release);
                    while !RELEASE.load(Ordering::Acquire) {
                        std::thread::yield_now();
                    }
                }
                Self(pid)
            }
        }
        static STATE: ProcessLocal<Initializing> = ProcessLocal::new();
        PARENT.store(std::process::id(), Ordering::Release);
        let initializing = std::thread::spawn(|| STATE.get().0);
        let started = Instant::now();
        while !STARTED.load(Ordering::Acquire) {
            assert!(started.elapsed() < Duration::from_secs(5));
            std::thread::yield_now();
        }
        let child_ok = fork_probe(|| STATE.get().0 == std::process::id());
        RELEASE.store(true, Ordering::Release);
        assert_eq!(initializing.join().unwrap(), std::process::id());
        assert!(child_ok, "fork child waited for inherited initialization");
    }
}
