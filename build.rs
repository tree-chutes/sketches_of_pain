//Copyright (c) 2025, tree-chutes

#[cfg(target_os = "linux")]
fn main() {
    println!("cargo:rerun-if-changed=../core/src/c/co5_lib.c");
    cc::Build::new()
                .files(vec!["../core/src/c/co5_lib.c"])
                .include("../core/src/c/headers")
                .compile("co5_c_lib");
}

#[cfg(not(target_os = "linux"))]
fn main() {
    println!("cargo:rerun-if-changed=./src/c/co5_osx_lib.c");
    cc::Build::new()
                .files(vec!["./src/c/co5_dl_osx.c"])
                .include("./src/c/headers")
                .flag("-w")
                .flag("-mavx")
                .compile("co5_dl_c");
}