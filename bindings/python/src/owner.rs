//! One native owner and a non-queuing operation gate shared by collections.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cofferdb::{Database, Error};

#[derive(Debug)]
pub(crate) enum Failure {
    // Carry the original engine error and its structured details to the Python
    // error projector. A normal engine error does not by itself poison the owner.
    Engine(Error),
    // The handle was explicitly closed. The close call itself still permits this
    // state, allowing context cleanup and repeated close.
    Closed,
    // Another call already holds this owner's admission, including while
    // delivering a result or checking Python signals.
    Busy,
    // The current PID differs from the creator. Close and destruction must also
    // respect this boundary because they could release inherited file locks.
    Forked,
    // An earlier protected call unwound. Keep identifying this owner as poisoned
    // on later ordinary calls while still permitting explicit close.
    Poisoned,
    // The present call observed an unwind. Report it separately from a later
    // attempt that finds an already poisoned owner.
    Panic,
}

struct State {
    // The native handle at rest. An admitted call takes it out of the mutex without
    // closing it.
    native: Option<Database>,
    // An explicit close consumed the native handle. This is distinct from the
    // temporarily empty slot occupied by an active call.
    closed: bool,
    // An unwind crossed a protected native boundary. Further ordinary calls are
    // rejected, while explicit close remains available.
    poisoned: bool,
}

pub(crate) struct Owner {
    // Creating process ID checked before any gate or mutex access in a child.
    pid: u32,
    // Non-queuing admission shared by the database and all collection handles, held
    // through Python conversion and error delivery.
    busy: AtomicBool,
    // A child must not drop a mutex, file or ledger inherited from a parent.
    // Protects idle ownership and the close and panic flags. The active call takes
    // the database out while holding this mutex.
    state: ManuallyDrop<Mutex<State>>,
}

impl Owner {
    pub(crate) fn pid(&self) -> u32 {
        self.pid
    }

    pub(crate) fn new(native: Database) -> Arc<Self> {
        // Every collection returned through this database shares this Arc. Keep
        // admission and lifecycle state together so closing the database
        // invalidates all those handles.
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
        // Admission covers Python argument and result conversion as well as
        // the detached engine call. Re-entry must return Busy even when the
        // state mutex is free and the handle is on the original caller's stack.
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(Failure::Busy);
        }
        // After winning admission the short mutex section only transfers ownership
        // and reads lifecycle flags. The explicit poisoned flag records operation
        // unwinds even when no Rust mutex guard was held during the failing work.
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        // Closing is idempotent and remains available for a poisoned owner. All
        // callers still pass the process and busy checks first, so close cannot
        // race an in-flight operation.
        let failure = if state.closed && !closing {
            Some(Failure::Closed)
        } else if state.poisoned && !closing {
            Some(Failure::Poisoned)
        } else {
            None
        };
        // No Call guard exists on this rejection path. Release the gate here, or a
        // failed closed/poisoned lookup would leave every later call reporting
        // Busy, including close.
        if let Some(error) = failure {
            self.busy.store(false, Ordering::Release);
            return Err(error);
        }
        // Move the native database into the call while the gate is held. Releasing
        // the mutex permits long detached work and Python callbacks without
        // locking the owner's state for their duration.
        Ok(Call {
            owner: Arc::clone(self),
            native: state.native.take(),
            poisoned: state.poisoned,
        })
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        // Run the destructor only in the process that owns these resources and
        // contain an unwind from cleanup. In a child, ManuallyDrop keeps the
        // inherited mutex and native handle out of destruction.
        if self.pid == std::process::id() {
            // SAFETY: this is the last Arc, so no operation can borrow State.
            // In a fork child the entire inherited state is quarantined instead.
            let _ = crate::panic_guard::catch(|| unsafe { ManuallyDrop::drop(&mut self.state) });
        }
    }
}

// An admitted call owns the temporarily detached database handle. The gate remains
// busy until its destructor has restored the idle state or recorded closure.
pub(crate) struct Call {
    // Keeps the shared owner alive until the native handle and terminal flags have
    // been restored.
    owner: Arc<Owner>,
    // Native database moved out of the owner's state for this call. Close takes it
    // permanently, including on a flush error.
    native: Option<Database>,
    // Panic state captured at admission and updated inside guarded code, then
    // published back before the busy flag is released.
    poisoned: bool,
}

impl Call {
    // Check the process on every guarded entry. Python conversion can run
    // arbitrary signal handlers between admission and later native work.
    fn check_pid(&self) -> Result<(), Failure> {
        if self.owner.pid == std::process::id() {
            Ok(())
        } else {
            Err(Failure::Forked)
        }
    }

    // Borrow the native handle only after process validation. A call admitted for
    // close can legitimately hold no database, so ordinary execution checks the
    // slot again.
    pub(crate) fn run<T>(
        &mut self,
        run: impl FnOnce(&Database) -> cofferdb::Result<T>,
    ) -> Result<T, Failure> {
        self.check_pid()?;
        let native = self.native.as_ref().ok_or(Failure::Closed)?;
        // Returning an engine error leaves the binding's poison flag untouched. An
        // unwind sets it before returning a panic failure.
        match crate::panic_guard::catch(|| run(native)) {
            Ok(result) => result.map_err(Failure::Engine),
            Err(_) => {
                self.poisoned = true;
                Err(Failure::Panic)
            }
        }
    }

    // Consume the file owner while holding admission. The caller can still report
    // a native close error without handing the handle back to another operation.
    pub(crate) fn close(&mut self) -> Result<(), Failure> {
        self.check_pid()?;
        // Native close consumes the handle. Leave the slot empty even on a
        // flush error so Drop records the owner as closed.
        // An empty slot is a successful repeated close. Once taken, the native
        // value is moved into the guarded close routine.
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

    // Guard host result construction that does not require a native database
    // borrow. A panic here still poisons the same owner because it occurred inside
    // the admitted call.
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
        // Reject inherited cleanup before touching the state mutex or gate. A
        // child can inherit this stack while the parent's process still owns file
        // locks.
        if self.owner.pid != std::process::id() {
            // This is defensive: an inherited native operation normally belongs
            // to a vanished thread. Never unlock its shared file description.
            if let Some(native) = self.native.take() {
                std::mem::forget(native);
            }
            return;
        }
        // Return the handle to its idle slot only on the originating process. The
        // mutex guards the combined update of the handle and terminal flags.
        let mut state = self
            .owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Restore the database and terminal flags before releasing admission.
        // A new caller must never observe the in-flight empty slot as closed.
        state.closed = self.native.is_none();
        state.native = self.native.take();
        state.poisoned = self.poisoned;
        drop(state);
        // Publish release after restoring state and dropping the mutex. A new
        // caller's admission must observe either a live handle or the terminal
        // flags.
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
