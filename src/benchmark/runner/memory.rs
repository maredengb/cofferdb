// This is the lifetime high-water mark of the current process, not a per-query sample.
// Use the native process's peak for the CofferDB run. A SQLite baseline executes
// in child processes and leaves its own metric unavailable rather than reusing
// this value.
pub(super) fn process_memory_peak_mb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        // Read the current process's status at the point the group is reported.
        // VmHWM can include allocations from import, reference preparation or
        // earlier runs in this process.
        linux_peak_mb(&std::fs::read_to_string("/proc/self/status").ok()?)
    }
    #[cfg(target_os = "macos")]
    {
        macos::peak_mb()
    }
    // An unsupported platform has no measurement for this field. The result
    // serializer preserves None as JSON null so consumers need not infer
    // availability from a numeric value.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(any(target_os = "linux", test))]
fn linux_peak_mb(status: &str) -> Option<f64> {
    // Select the resident high-water mark explicitly. VmRSS in the same file
    // describes current usage and can drop after the benchmark frees its working
    // data.
    let (_, value) = status
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(field, _)| *field == "VmHWM")?;
    let mut words = value.split_whitespace();
    let kib = words.next()?.parse::<u64>().ok()?;
    // Accept the kernel's positive integer followed by the exact kB label and
    // nothing else. A malformed or differently scaled reading becomes unavailable
    // rather than a plausible but incomparable metric.
    if kib == 0 || words.next()? != "kB" || words.next().is_some() {
        return None;
    }
    // The Linux kB value is expressed in units of 1024 bytes. Divide by 1024 to
    // report the same binary megabytes produced by the Darwin path.
    Some(kib as f64 / 1024.0)
}

#[cfg(target_os = "macos")]
mod macos {
    use std::os::raw::{c_int, c_long};

    // Darwin SDK: sys/_types/_timeval.h and sys/resource.h. ru_maxrss is bytes.
    #[derive(Default)]
    #[repr(C)]
    // Retain the native timeval layout even though no CPU time is read. Both
    // fields and their trailing padding determine the offset of max_rss in the
    // outer record.
    struct Timeval {
        // Whole seconds in Darwin's timeval layout preceding the microsecond field.
        seconds: c_long,
        // Subsecond microseconds with the SDK's integer width and C padding.
        microseconds: c_int,
    }

    #[derive(Default)]
    #[repr(C)]
    // getrusage writes a complete Darwin rusage. The two preceding timeval fields
    // and all following counters must remain present even though the benchmark
    // consumes a single member.
    struct Rusage {
        // User CPU time occupying the first timeval in the Darwin rusage layout.
        user_time: Timeval,
        // System CPU time occupying the second timeval before the RSS field.
        system_time: Timeval,
        // Darwin's maximum resident size in bytes, converted to MiB for the report.
        max_rss: c_long,
        // Space for the remaining SDK counters so getrusage can write the complete
        // structure.
        remaining_counters: [c_long; 13],
    }

    pub(super) fn peak_mb() -> Option<f64> {
        extern "C" {
            fn getrusage(who: c_int, usage: *mut Rusage) -> c_int;
        }
        // Provide initialized storage for the entire output structure. The kernel
        // may leave its fields unusable on error, so a positive value is recorded
        // only after a successful return.
        let mut usage = Rusage::default();
        // SAFETY: The writable, aligned Rusage has Darwin's C layout and lives
        // through the call. RUSAGE_SELF requests statistics for this process.
        let status = unsafe { getrusage(0, &mut usage) };
        // Darwin reports ru_maxrss in bytes. Divide by 2^20 for the report and
        // treat a failed call or nonpositive reading as unavailable.
        (status == 0 && usage.max_rss > 0).then_some(usage.max_rss as f64 / 1_048_576.0)
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn rusage_matches_supported_darwin_abi() {
        assert_eq!(std::mem::size_of::<Timeval>(), 16);
        assert_eq!(std::mem::size_of::<Rusage>(), 144);
        assert_eq!(std::mem::align_of::<Rusage>(), 8);
        assert_eq!(std::mem::offset_of!(Rusage, max_rss), 32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_peak_uses_only_valid_high_water_marks_across_100_samples() {
        for seed in 1..=100 {
            let kib = seed * 1024 + 17;
            for status in [
                format!("Name:\tcoffer\nVmRSS:\t1 kB\nVmHWM:\t{kib} kB\n"),
                format!("VmHWM: {kib} kB\nVmRSS: 999999999 kB\n"),
            ] {
                assert_eq!(linux_peak_mb(&status), Some(kib as f64 / 1024.0));
            }
            for invalid in [
                "-1 kB",
                "0 kB",
                "NaN kB",
                "inf kB",
                "1.5 kB",
                "1 MB",
                "1",
                "1 kB extra",
                "18446744073709551616 kB",
            ] {
                assert_eq!(
                    linux_peak_mb(&format!("VmRSS: {kib} kB\nVmHWM: {invalid}\n")),
                    None
                );
            }
            assert_eq!(linux_peak_mb(&format!("VmRSS: {kib} kB\n")), None);
        }
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn peak_survives_release_of_touched_memory() {
        const CHILD: &str = "COFFERDB_TEST_PEAK_RSS_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let before = process_memory_peak_mb().expect("supported host reports peak RSS");
            let pages = std::hint::black_box(vec![0x5a_u8; 96 * 1024 * 1024]);
            let during = process_memory_peak_mb().unwrap();
            assert!(during >= before + 64.0, "before={before} during={during}");
            drop(pages);
            let after = process_memory_peak_mb().unwrap();
            // Linux VmHWM uses approximate RSS counters; verify retention, not exact monotonicity.
            assert!(after >= before + 64.0, "before={before} after={after}");
            #[cfg(target_os = "macos")]
            assert!(after >= during, "during={during} after={after}");
            return;
        }
        // Isolate the measurement from memory peaks of concurrently executing tests.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "benchmark::runner::memory::tests::peak_survives_release_of_touched_memory",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    }
}
