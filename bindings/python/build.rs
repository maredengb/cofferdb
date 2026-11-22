fn main() {
    // Re-run this check when the deployment target changes, even when no Rust
    // source file has changed. The value influences the native minimum OS version
    // recorded in the wheel.
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    // Test Cargo's target OS rather than the host running the build script. A
    // macOS cross-build must satisfy the same wheel contract; builds targeting
    // other systems do not use this deployment setting.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && std::env::var("MACOSX_DEPLOYMENT_TARGET").as_deref() != Ok("15.0")
    {
        // Require the exact packaging value, including its minor component. The
        // script can check Cargo's environment here, while the wheel builder needs
        // the same setting to choose a matching platform tag.
        eprintln!(
            "The CofferDB Python package requires MACOSX_DEPLOYMENT_TARGET=15.0. \
             Set it for both maturin and Cargo so the wheel tag and native minimum agree."
        );
        std::process::exit(1);
    }
}
