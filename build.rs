fn main() {
    println!("cargo:rustc-link-arg=-mmacosx-version-min=14.0");
    println!("cargo:rerun-if-changed=native/resample/resample.cc");
    println!("cargo:rerun-if-changed=native/resample/resample.h");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .flag("-mmacosx-version-min=14.0")
        .file("native/resample/resample.cc")
        .compile("parakeet_resample");
}
