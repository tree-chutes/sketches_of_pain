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
    let source_files = vec!["./src/c/co5_dl_double_osx.c",
        "./src/c/co5_dl_losses_double_osx.c",
        "./src/c/co5_dl_float_osx.c",
        "./src/c/co5_dl_losses_float_osx.c"];
    println!("cargo:rerun-if-changed=./src/c/*");
    cc::Build::new()
                .files(source_files)
                .include("./src/c/headers")
                .flag("-w")
                .flag("-g")
                //.flag("-fsanitize=address")
                .flag("-mavx")
                .flag("-mfma")
                .flag("-mavx512f")
                .compile("co5_dl_c");
}
