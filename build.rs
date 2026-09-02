fn main() {
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");

    // Only the `openblas` feature links a system BLAS on Linux. Without it the
    // MaxSim kernel uses the bundled pure-Rust SGEMM, so the extension module
    // has no external library dependency and cannot be paired at load time
    // with a BLAS that is slower than the one it was measured against.
    //
    // macOS needs nothing here: `blas-src` with the `accelerate` feature emits
    // its own link flags for a framework that ships with the OS.
    if std::env::var_os("CARGO_FEATURE_OPENBLAS").is_some() {
        if let Ok(directory) = std::env::var("OPENBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={directory}");
        }
        println!("cargo:rustc-link-lib=openblas");
    }
}
