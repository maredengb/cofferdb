//! One native owner and a non-queuing operation gate shared by collections.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cofferdb::{Database, Error};

#[derive(Debug)]
pub(crate) enum Failure {
    Engine(Error),
    Closed,
    Busy,
    Forked,
    Poisoned,
    Panic,
}

struct State {
    native: Option<Database>,
    closed: bool,
    poisoned: bool,
}

pub(crate) struct Owner {
    pid: u32,
    busy: AtomicBool,
    // A child must not drop a mutex, file or ledger inherited from a parent.
    state: ManuallyDrop<Mutex<State>>,
}

impl Owner {
    pub(crate) fn pid(&self) -> u32 {
        self.pid
    }

    pub(crate) fn new(native: Database) -> Arc<Self> {
        Arc::new(Self {
            pid: std::process::id(),
            busy: AtomicBool::new(false),
            state: ManuallyDrop::new(Mutex::new(State {
                native: Some(native),
                closed: false,
                poisoned: false,
            })),
        })
    }

    pub(crate) fn enter(self: &Arc<Self>, closing: bool) -> Result<Call, Failure> {
        // In particular, this precedes *all* mutex access, including close.
        if self.pid != std::process::id() {
            return Err(Failure::Forked);
        }
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(Failure::Busy);
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let failure = if state.closed && !closing {
            Some(Failure::Closed)
        } else if state.poisoned && !closing {
            Some(Failure::Poisoned)
        } else {
            None
        };
        if let Some(error) = failure {
            self.busy.store(false, Ordering::Release);
            return Err(error);
        }
        Ok(Call {
            owner: Arc::clone(self),
            native: state.native.take(),
            poisoned: state.poisoned,
        })
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        if self.pid == std::process::id() {
            // SAFETY: this is the last Arc, so no operation can borrow State.
            // In a fork child the entire inherited state is quarantined instead.
            let _ = crate::panic_guard::catch(|| unsafe { ManuallyDrop::drop(&mut self.state) });
        }
    }
}

pub(crate) struct Call {
    owner: Arc<Owner>,
    native: Option<Database>,
    poisoned: bool,
}

impl Call {
    fn check_pid(&self) -> Result<(), Failure> {
        if self.owner.pid == std::process::id() {
            Ok(())
        } else {
            Err(Failure::Forked)
        }
    }

    pub(crate) fn run<T>(
        &mut self,
        run: impl FnOnce(&Database) -> cofferdb::Result<T>,
    ) -> Result<T, Failure> {
        self.check_pid()?;
        let native = self.native.as_ref().ok_or(Failure::Closed)?;
        match crate::panic_guard::catch(|| run(native)) {
            Ok(result) => result.map_err(Failure::Engine),
            Err(_) => {
                self.poisoned = true;
                Err(Failure::Panic)
            }
        }
    }

    pub(crate) fn close(&mut self) -> Result<(), Failure> {
        self.check_pid()?;
        let Some(native) = self.native.take() else {
            return Ok(());
        };
        match crate::panic_guard::catch(|| native.close()) {
            Ok(result) => result.map_err(Failure::Engine),
            Err(_) => {
                self.poisoned = true;
                Err(Failure::Panic)
            }
        }
    }

    pub(crate) fn protect<T>(&mut self, run: impl FnOnce() -> T) -> Result<T, Failure> {
        self.check_pid()?;
        match crate::panic_guard::catch(run) {
            Ok(value) => Ok(value),
            Err(_) => {
                self.poisoned = true;
                Err(Failure::Panic)
            }
        }
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        if self.owner.pid != std::process::id() {
            // This is defensive: an inherited native operation normally belongs
            // to a vanished thread. Never unlock its shared file description.
            if let Some(native) = self.native.take() {
                std::mem::forget(native);
            }
            return;
        }
        let mut state = self
            .owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.closed = self.native.is_none();
        state.native = self.native.take();
        state.poisoned = self.poisoned;
        drop(state);
        self.owner.busy.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cofferdb::{DatabaseConfig, ErrorCode};
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct File(PathBuf);
    impl File {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                "coffer-python-owner-{}-{}.coffer",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            )))
        }
        fn owner(&self) -> Arc<Owner> {
            Owner::new(Database::create(&self.0, DatabaseConfig::default()).unwrap())
        }
    }
    impl Drop for File {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn an_in_flight_call_rejects_work_and_close_without_waiting() {
        let file = File::new();
        let owner = file.owner();
        let call = owner.enter(false).unwrap();
        let other = Arc::clone(&owner);
        std::thread::spawn(move || {
            assert!(matches!(other.enter(false), Err(Failure::Busy)));
            assert!(matches!(other.enter(true), Err(Failure::Busy)));
        })
        .join()
        .unwrap();
        drop(call);
        let mut call = owner.enter(false).unwrap();
        assert!(call.run(|db| db.stats()).is_ok());
        call.close().unwrap();
    }

    #[test]
    fn explicit_close_releases_writer_despite_retained_owner_reference() {
        let file = File::new();
        let owner = file.owner();
        let retained = Arc::clone(&owner);
        owner.enter(true).unwrap().close().unwrap();
        assert!(matches!(retained.enter(false), Err(Failure::Closed)));
        retained.enter(true).unwrap().close().unwrap();
        let reopened = Database::open(&file.0).unwrap();
        reopened.close().unwrap();
    }

    #[test]
    fn an_engine_error_keeps_owner_usable() {
        let file = File::new();
        let owner = file.owner();
        let mut call = owner.enter(false).unwrap();
        let error = call.run(|db| db.collection("missing")).err().unwrap();
        assert!(
            matches!(error, Failure::Engine(error) if error.code() == ErrorCode::CollectionNotFound)
        );
        drop(call);
        assert!(owner.enter(false).unwrap().run(|db| db.stats()).is_ok());
        owner.enter(true).unwrap().close().unwrap();
    }

    #[test]
    fn a_caught_panic_poisoned_owner_only_allows_close() {
        let file = File::new();
        let owner = file.owner();
        let mut call = owner.enter(false).unwrap();
        assert!(matches!(
            call.run::<()>(|_| panic!("controlled owner test")),
            Err(Failure::Panic)
        ));
        drop(call);
        assert!(matches!(owner.enter(false), Err(Failure::Poisoned)));
        owner.enter(true).unwrap().close().unwrap();
        let reopened = Database::open(&file.0).unwrap();
        reopened.close().unwrap();
    }
}
