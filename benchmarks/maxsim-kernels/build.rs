fn main() {
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");

    // Unlike the main crate, this benchmark exists to compare kernels against
    // BLAS, so BLAS is not optional here. It is still worth honouring
    // OPENBLAS_LIB_DIR: a machine that keeps libopenblas.so.0 outside the
    // linker's default search path can build the benchmark without a global
    // environment change.
    #[cfg(target_os = "linux")]
    {
        if let Ok(directory) = std::env::var("OPENBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={directory}");
        }
        println!("cargo:rustc-link-lib=openblas");
    }
}
