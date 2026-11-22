//! Deterministic rendezvous for a dedicated native test build. A test owns
//! the directory and sets the environment only after creating its fixture.
use crate::Result;
use std::{
    env, fs,
    path::PathBuf,
    time::{Duration, Instant},
};

pub(crate) fn pause(stage: &str) -> Result<()> {
    if env::var("COFFER_TEST_MAINTENANCE_STAGE").as_deref() != Ok(stage) {
        return Ok(());
    }
    let Some(directory) = env::var_os("COFFER_TEST_MAINTENANCE_DIRECTORY") else {
        return Ok(());
    };
    let directory = PathBuf::from(directory);
    fs::write(directory.join("ready"), stage)?;
    let start = Instant::now();
    loop {
        match fs::read_to_string(directory.join("resume")) {
            Ok(action) if action == "continue" => return Ok(()),
            Ok(action) if action == "enospc" => {
                return Err(std::io::Error::from_raw_os_error(libc::ENOSPC).into())
            }
            Ok(action) if action == "eio" => {
                return Err(std::io::Error::from_raw_os_error(libc::EIO).into())
            }
            Ok(action) if action == "panic" => panic!("injected maintenance panic at {stage}"),
            Ok(_) => return Err(std::io::Error::other("invalid maintenance probe response").into()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => return Err(error.into()),
        }
        if start.elapsed() >= Duration::from_secs(15) {
            return Err(std::io::Error::other("maintenance probe timed out").into());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}
