//Copyright (c) 2025, tree-chutes

#[cfg(target_os = "linux")]
fn main() {
    println!("cargo:rerun-if-changed=./src/c/co5_dl_osx.c");
    cc::Build::new()
                .files(vec!["../core/src/c/co5_dl_osx.c"])
                .compile("co5_dl_c");
}

#[cfg(not(target_os = "linux"))]
fn main() {
    println!("cargo:rerun-if-changed=./src/c/co5_dl_osx.c");
    cc::Build::new()
                .files(vec!["./src/c/co5_dl_osx.c"])
                .flag("-w")
                .flag("-mavx")
                .compile("co5_dl_c");
}
