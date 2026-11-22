// The binary chooses presentation, delivers the result and sets process status.
// The CLI library returns the result together with any receipt needed if delivery
// fails.
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    // Keep the original invocation's command hint for error rendering. The CLI
    // layer separately parses and validates the complete argument list.
    let command = args.get(1).cloned();
    // Choose the presentation mode before CLI parsing so even a parse error can
    // use the JSON envelope.
    let json = args.iter().any(|arg| arg == "--json");
    match cofferdb::cli::run_for_display(args, json) {
        Ok((output, exit_code, receipt)) => {
            use std::io::Write;
            // Hold the stdout lock through writing and flushing one complete
            // result. A successful command can still fail to deliver it to the
            // caller.
            let mut stdout = std::io::stdout().lock();
            if let Err(failure) = stdout
                .write_all(output.as_bytes())
                .and_then(|_| stdout.flush())
            {
                // Preserve the maintenance receipt when output delivery fails. The
                // file effect may have occurred before the caller received any
                // result.
                let err = receipt.error(failure.into(), "deliver_result");
                print_error(command.as_deref(), json, &err);
                std::process::exit(2);
            }
            // Apply the exit status from a successfully delivered report. Some
            // check outcomes carry a nonzero status while still returning a valid
            // result body.
            if exit_code != 0 {
                std::process::exit(exit_code);
            }
        }
        // Execution or rendering failed before the library returned a usable
        // result. Classify its code using the same process contract as other CLI
        // failures.
        Err(err) => {
            print_error(command.as_deref(), json, &err);
            std::process::exit(error_exit_code(err.code()));
        }
    }
}

// Send diagnostics to stderr so stdout remains reserved for the command result.
// JSON mode uses the machine-readable error envelope.
fn print_error(command: Option<&str>, json: bool, err: &cofferdb::Error) {
    if json {
        eprintln!("{}", cofferdb::cli::error_output(command, err));
    } else {
        // Print the code and message in human mode. When the error carries an
        // effect field, include all details and any recovery action.
        eprintln!("{}: {}", err.code().as_str(), err.message());
        if err.detail().iter().any(|(key, _)| key == "effect") {
            for (key, value) in err.detail() {
                eprintln!("  {key}: {value}");
            }
            if let Some(action) = err
                .to_json()
                .get("suggested_action")
                .and_then(|v| v.as_str())
            {
                eprintln!("  {action}");
            }
        }
    }
}

// Map failed operations to stable process codes independently of successful report
// statuses. Keep the table aligned with the values advertised by CLI help.
fn error_exit_code(code: cofferdb::ErrorCode) -> i32 {
    use cofferdb::ErrorCode;
    // Report storage, recovery and resource failures under the shared failure
    // class. Lock contention has its own status so scripts can distinguish it from
    // general errors.
    match code {
        ErrorCode::IoError
        | ErrorCode::FileFormatError
        | ErrorCode::ChecksumMismatch
        | ErrorCode::CorruptTail
        | ErrorCode::IndexCorruption
        | ErrorCode::RecoveryError
        | ErrorCode::ResourceExhausted => 2,
        ErrorCode::LockContention => 4,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cofferdb::ErrorCode;

    #[test]
    fn resource_exhaustion_uses_exit_two_without_changing_existing_categories() {
        for code in [
            ErrorCode::IoError,
            ErrorCode::FileFormatError,
            ErrorCode::ChecksumMismatch,
            ErrorCode::CorruptTail,
            ErrorCode::IndexCorruption,
            ErrorCode::RecoveryError,
            ErrorCode::ResourceExhausted,
        ] {
            assert_eq!(error_exit_code(code), 2);
        }
        assert_eq!(error_exit_code(ErrorCode::LockContention), 4);
        for code in [
            ErrorCode::InvalidJsonlSchema,
            ErrorCode::InvalidMetadataFilter,
            ErrorCode::InvalidSearchWeights,
            ErrorCode::InvalidMetadataBoost,
            ErrorCode::CliUsage,
        ] {
            assert_eq!(error_exit_code(code), 1);
        }
    }
}
