fn main() {
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && std::env::var("MACOSX_DEPLOYMENT_TARGET").as_deref() != Ok("15.0")
    {
        eprintln!(
            "The CofferDB Python package requires MACOSX_DEPLOYMENT_TARGET=15.0. \
             Set it for both maturin and Cargo so the wheel tag and native minimum agree."
        );
        std::process::exit(1);
    }
}
