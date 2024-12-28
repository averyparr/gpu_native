use std::{fs, path::PathBuf, process::Command};

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{ItemFn, parse_macro_input, parse_str};

fn assert_cmd_success(prog_name: &str, out: &std::process::Output) {
    assert!(
        out.status.success(),
        "{prog_name} failure:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8(out.stdout.clone()).expect("String utf-8 error [stdout]"),
        String::from_utf8(out.stderr.clone()).expect("String utf-8 error [stderr]"),
    );
}

fn run_with_cuda_docker(loc: &PathBuf, program: &str, arg_list: &[&str]) {
    let loc_arg = format!("{}:/work", loc.display());
    let mut args = vec![
        "run",
        "--rm",
        "-v",
        &loc_arg,
        "-w",
        "/work",
        "nvidia/cuda:12.6.3-devel-ubuntu24.04",
        program,
    ];
    args.extend_from_slice(arg_list);
    let out = Command::new("docker")
        .current_dir(&loc)
        .args(args)
        .output()
        .expect(&format!("Unable to docker-run {program}"));

    assert_cmd_success(program, &out);
}

#[proc_macro_attribute]
pub fn gpu_kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;

    println!("Compiling CUDA kernel {fn_name}");

    // Create a temporary crate for the kernel
    let kernel_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernel_build")
        .join(fn_name.to_string());
    fs::create_dir_all(&kernel_dir).expect("Could not `create_dir_all`");

    fs::write(
        kernel_dir.join(format!("{}_kernel.rs", fn_name)),
        format!(
            r#"#![no_std]
#![feature(abi_ptx)]
#![feature(stdarch_nvptx)]

use core::panic::PanicInfo;

#[unsafe(no_mangle)]
pub extern "C" fn _ZN4core9panicking5panic17hac183deac67580b9E(
    msg: *const u8,
    _: u64,
    _file_line: *const u64,
) -> ! {{
    unsafe {{
        core::arch::nvptx::vprintf(msg as *const u8, core::ptr::null_mut());
        core::arch::nvptx::trap()
    }}
}}

#[no_mangle]
#[panic_handler]
pub fn panic_handler(info: &PanicInfo) -> ! {{
    static PANIC_MSG: &[u8] = b"NVPTX kernel panic: %s\0";
    
    unsafe {{
        core::arch::nvptx::vprintf(PANIC_MSG.as_ptr() as *const u8, info.message().as_str().unwrap_or("<No Msg>").as_ptr() as *const _);
        core::arch::nvptx::trap()
    }}
}}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn {}_kernel({}) {{
    {}
}}
    "#,
            fn_name,
            input_fn.sig.inputs.to_token_stream().to_string(),
            input_fn.block.to_token_stream().to_string(),
        ),
    )
    .expect("Unable to write kernel .rs file");

    // Compile using rustc directly
    let rustc_ir = Command::new("rustc")
        .current_dir(&kernel_dir)
        .args(&[
            "--crate-type=cdylib",
            "--target=nvptx64-nvidia-cuda",
            &format!("--emit=asm={}_kernel.ptx", fn_name),
            "-C",
            "opt-level=3",
            &format!("{}_kernel.rs", fn_name),
        ])
        .output()
        .expect("Failed to compile kernel with rustc");

    assert_cmd_success("rustc", &rustc_ir);

    let ptx_path = &format!("{}_kernel.ptx", fn_name);
    let cubin_path = &format!("{}_kernel.cubin", fn_name);

    run_with_cuda_docker(&kernel_dir, "ptxas", &[ptx_path, "-o", cubin_path]);

    let static_ptx_name = format!("{}_PTX", fn_name.to_string().to_uppercase());
    let static_ptx_ident = parse_str::<syn::Ident>(&static_ptx_name).unwrap();
    let ptx_contents = fs::read_to_string(kernel_dir.join(ptx_path)).expect("PTX string read.");

    let static_cubin_name = format!("{}_CUBIN", fn_name.to_string().to_uppercase());
    let static_cubin_ident = parse_str::<syn::Ident>(&static_cubin_name).unwrap();
    let cubin_contents = fs::read(kernel_dir.join(cubin_path)).expect("cubin bytes read.");

    return quote! {
        static #static_ptx_ident: &str = #ptx_contents;
        static #static_cubin_ident: &[u8] = &[#(#cubin_contents),*];
    }
    .into();
}
