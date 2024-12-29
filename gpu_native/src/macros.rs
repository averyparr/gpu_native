#[inline]
#[cold]
pub fn __cold() {}

#[macro_export]
macro_rules! link_in_sass {
    () => {
        compile_error!(
            r#"Format examle: `link_in_sass!(
                sm = ("sm_80", "sm_86"),
                host_arch = ("x86_64", "armv7", "aarch64"),
                cargo_rustc_args = ("--release"),
                docker_image = "nvidia/cuda:12.6.3-devel-ubuntu24.04"
            )`"#
        )
    };
    (
        sm=($($sms: literal),+),
        host_arch=($($host_archs: literal),+),
        cargo_rustc_args=($($cargo_rustc_args: literal),*)$(,)?
        $(docker_image = $docker_image: literal)?) => {
        || -> () {
            struct TemporaryDir(std::path::PathBuf);
            impl TemporaryDir {
                fn new(path: std::path::PathBuf) -> Self {
                    std::fs::create_dir(path.as_path())
                        .expect("Could not create pseudorandom directory. Try again?");
                    Self(path)
                }
                fn as_path(&self) -> &std::path::Path {
                    self.0.as_path()
                }
            }
            impl Drop for TemporaryDir {
                fn drop(&mut self) {
                    std::fs::remove_dir_all(self.0.as_path())
                        .expect("Could not remove the entire pseudorandom directory.");
                }
            }
            use std::{collections::HashSet, io::Write, process::Command, str::FromStr};

            let docker_image: Option<&str> = None$(.or_else(|| Some($docker_image)))?;
            let host_archs: &[&str] = &[$($host_archs),*];

            let target = std::env::var("TARGET").expect("OS '$TARGET' env not set");
            if !host_archs.iter().any(|host| target.contains(host)) {
                if target != "nvptx64-nvidia-cuda" {
                    panic!("Attempted to compile for target '{target}' which is neither
                    \r\t a (user-defined) host nor (library-defined) GPU architecture.
                    \r\t Consider adding \"{}\" to the `host_arch` list.",
                    target.split("-").next().unwrap()
                );
                }
                return;
            }
            // I want to have zero build dependencies, so we use this instead of `tempfile`.
            let pseudorandom = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
                % (u16::MAX as u32);
            let prnd_filename = format!(".link_sass_{pseudorandom}");
            let guard = TemporaryDir::new(std::path::PathBuf::from_str(&prnd_filename).unwrap());
            let build_dir = guard.as_path();

            let mut use_docker = docker_image.is_some();
            if use_docker {
                let docker_exists = match Command::new("docker").spawn() {
                    Ok(_) => true,
                    Err(e) => {
                        if let std::io::ErrorKind::NotFound = e.kind() {
                            false
                        }
                        else {
                            panic!("Weird error '{e}' appeared when spawning Docker.");
                        }
                    }
                };
                if !docker_exists {
                    println!("cargo::warning=Docker doesn't exist. Trying to compile with system executables.");
                    use_docker = false;
                }
            }


            let docker_cmd_prefix = [
                "run",
                "--rm",
                "-v",
                &format!("{}:/work", build_dir.display()),
                "-w",
                "/work",
                docker_image.unwrap_or("nvidia/cuda:12.6.3-devel-ubuntu24.04"),
            ];

            let cmd_prefix = if use_docker {
                docker_cmd_prefix.as_slice()
            } else {
                &[]
            };

            let mut fatbin_args: Vec<String> = vec![];
            let mut fatbin_exec = "fatbinary";
            if use_docker {
                fatbin_exec = "docker";
                fatbin_args.push("fatbinary".to_string());
            }
            fatbin_args.push("--create=lib.fatbin".to_string());
            fatbin_args.push("--64".to_string());

            let mut already_compiled_ptx: HashSet<String> = HashSet::new();

            let sm_versions: &[&str] = &[$($sms,)*];

            let rustc_args: &[&str] = &[$($cargo_rustc_args,)*];
            let include_ddash = !rustc_args.iter().any(|arg| *arg == "--");
            for sm_version in sm_versions {
                let required_prefix_args = [
                    "rustc",
                    &format!("--target-dir={}", build_dir.display()),
                    "--target=nvptx64-nvidia-cuda",
                ];
                let maybe_ddash = if include_ddash { Some("--") } else { None };
                let required_suffix_args = [
                    "-C",
                    &format!("target-cpu={sm_version}"),
                    "-C",
                    "panic=abort",
                    "--emit=asm",
                ];
                let status = Command::new("cargo")
                    .args(
                        required_prefix_args
                            .iter()
                            .chain(rustc_args.iter())
                            .chain(maybe_ddash.iter())
                            .chain(required_suffix_args.iter()),
                    )
                    .status()
                    .expect("Unable to launch `cargo`");

                if !status.success() {
                    panic!("`cargo rustc` failed");
                }

                let nvptx_build_dir = build_dir.join("nvptx64-nvidia-cuda");
                let target_dir = nvptx_build_dir
                    .read_dir()
                    .expect("Can't read `$TMP_TARGET/nvptx64-nvidia-cuda`")
                    .map(|f| f.expect("Non-existant child directory?"))
                    .find(|f| {
                        f.file_type()
                            .expect("Can't get file type for child")
                            .is_dir()
                    })
                    .expect("No child of $TMP_TARGET/nvptx64-nvidia-cuda is a directory!")
                    .path();
                let deps_build_dir = target_dir
                    .join("deps")
                    .read_dir()
                    .expect("Can't read $TMP_TARGET/nvptx64-nvidia-cuda/$FIRST_DIR/deps");

                let mut compiling_cmds = vec![];

                for dep in deps_build_dir {
                    let dep = dep.expect("Cannot get at DirEntry in `deps` dir.").path();
                    let rust_asm_extension = std::ffi::OsStr::new("s");
                    match dep.extension() {
                        Some(s) => {
                            if s != rust_asm_extension {
                                continue;
                            }
                        },
                        None => continue,
                    }

                    let dep_str_rep = dep.to_str().expect("Dependency path isn't utf-8 encoded.");
                    if already_compiled_ptx.contains(dep_str_rep) {
                        continue;
                    } else {
                        already_compiled_ptx.insert(dep_str_rep.to_string());
                    }

                    let rel_path = dep.strip_prefix(build_dir).expect("Unable to strip prefix");
                    let mut ptxas_exec = "ptxas";
                    let mut ptxas_as_arg = None;
                    if use_docker {
                        ptxas_exec = "docker";
                        ptxas_as_arg = Some("ptxas");
                    }

                    let ptxas_cmd = Command::new(ptxas_exec)
                        .args(
                            cmd_prefix.iter().chain(ptxas_as_arg.iter()).chain(
                                [
                                    format!("-arch={sm_version}").as_str(),
                                    format!("{}", rel_path.display()).as_str(),
                                    "-o",
                                    format!("{}.cubin", rel_path.display()).as_str(),
                                ]
                                .iter(),
                            ),
                        )
                        .current_dir(build_dir)
                        .spawn()
                        .expect("Could not spawn a `ptxas` process.");
                    compiling_cmds.push((rel_path.to_path_buf(), ptxas_cmd));
                }

                for (subfile, mut ptxas_proc) in compiling_cmds {
                    let status = ptxas_proc.wait().expect("Unable to compile PTX...");
                    if !status.success() {
                        panic!("ptxas failure!");
                    }
                    fatbin_args.push(format!(
                        "--image=profile={},file={}.cubin",
                        sm_version,
                        subfile.display()
                    ));
                }
            }

            let fatbin_status = Command::new(fatbin_exec)
                .args(
                    cmd_prefix
                        .iter()
                        .map(|s| *s)
                        .chain(fatbin_args.iter().map(|s| s.as_str())),
                )
                .current_dir(build_dir)
                .status()
                .expect("Could not spawn a `fatbinary` process.");

            if !fatbin_status.success() {
                panic!("`fatbinary` process panicked!");
            }

            let bytes = std::fs::read(build_dir.join("lib.fatbin")).expect("Can't read in fatbin");
            let mut source_file = String::with_capacity(6 * bytes.len());
            let link_name = concat!(env!("CARGO_PKG_NAME"), "_FATBIN_CODE", "_7B4EA9D2");
            source_file.push_str(&format!(
                "#[unsafe(no_mangle)] static {}: &[u8] = &[",
                link_name
            ));
            const MAGIC_NUM: [u8; 8] = 0xB0BACAFEB0BACAFEu64.to_le_bytes();
            let mut write_byte = |byte: u8| -> () {
                let high = if (byte >> 4) < 10 {
                    b'0' + (byte >> 4)
                } else {
                    b'a' + (byte >> 4) - 10
                };
                let low = if (byte & 0xf) < 10 {
                    b'0' + (byte & 0xf)
                } else {
                    b'a' + (byte & 0xf) - 10
                };
                source_file.push('0');
                source_file.push('x');
                source_file.push(high as char);
                source_file.push(low as char);
                source_file.push(',');
                source_file.push('\n');
            };
            for byte in MAGIC_NUM {
                write_byte(byte);
            }
            for byte in bytes {
                write_byte(byte);
            }

            source_file.push_str("];");

            let mut embedded_rs =
                std::fs::File::create(build_dir.join("fatbin.rs")).expect("Can't create fatbin.rs");

            embedded_rs
                .write_all(source_file.as_bytes())
                .expect("Can't write to fatbin.rs");

            let linkable_obj = Command::new("rustc")
                .current_dir(&build_dir)
                .args(["--crate-type=lib", "--emit=obj", "fatbin.rs"])
                .status()
                .expect("Cannot start rustc!");

            if !linkable_obj.success() {
                panic!("rustc compilation of inline fatbinary failed!");
            }

            let out_dir = std::path::PathBuf::from_str(&std::env::var("OUT_DIR").expect("No OUT_DIR"))
                .expect("OUT_DIR isn't a path");
            let final_fatbin_filename = concat!(env!("CARGO_PKG_NAME"), "_fatbin.o");
            let final_fatbin_path = out_dir.as_path().join(final_fatbin_filename);
            std::fs::rename(build_dir.join("fatbin.o"), final_fatbin_path.as_path())
                .expect("Unable to move `fatbin.o` to OUT_DIR");
            std::fs::rename(build_dir.join("lib.fatbin"), out_dir.join("lib.fatbin"))
                .expect("Unable to move `lib.fatbin` to OUT_DIR");
            let copy_build_dir = out_dir.join("target-nvptx64");
            if copy_build_dir.exists() {
                std::fs::remove_dir_all(copy_build_dir.as_path())
                    .expect("Cannot remove $OUT_DIR/target-nvptx");
            }
            std::fs::rename(
                build_dir.join("nvptx64-nvidia-cuda"),
                copy_build_dir,
            )
            .expect("Unable to move `nvptx64-nvidia-cuda` to OUT_DIR");

            println!("cargo:rustc-link-arg={}", final_fatbin_path.display());


        }()
    };
}

#[cfg(not(any(target_arch = "nvptx64")))]
pub fn is_probably_ide() -> bool {
    std::env::var("RUSTC_WRAPPER")
        .map(|s| s.contains("rust-analyzer"))
        .unwrap_or(false)
}

#[macro_export]
macro_rules! assert_universal {
    ($expr: expr) => {
        #[cfg(target_arch = "nvptx64")]
        $crate::cu_assert!($expr);
        #[cfg(not(target_arch = "nvptx64"))]
        assert!($expr);

        unsafe { core::hint::assert_unchecked($expr)};
    };
    ($expr: expr, $($arg:tt)+) => {
        #[cfg(target_arch = "nvptx64")]
        $crate::cu_assert!($expr, $($arg)+);
        #[cfg(not(target_arch = "nvptx64"))]
        assert!($expr, $($arg)+);

        unsafe { core::hint::assert_unchecked($expr)};
    };
}

#[macro_export]
macro_rules! panic_universal {
    ($($arg:tt)*) => {
        #[cfg(target_arch = "nvptx64")]
        {
            crate::cu_panic!($($arg)*);
        }

        #[cfg(not(target_arch = "nvptx64"))]
        {
            panic!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! cu_assert {
    ($expr: expr) => {
        #[cfg(target_arch = "nvptx64")]
        if !($expr) {
            $crate::macros::__cold();
            unsafe {
                $crate::cuda::intrinsics::__assert_fail(
                    stringify!($cond).as_ptr() as *const u8,
                    file!().as_ptr() as *const u8,
                    line!(),
                    "[fn cannot be captured]".as_ptr() as *const u8,
                )
            };
        }
        #[cfg(not(target_arch = "nvptx64"))]
        compile_error!("Cannot call cu_assert outside PTX!");
    };
    ($expr: expr, $($arg:tt)+) => {
        #[cfg(target_arch = "nvptx64")]
        if !($expr) {
            $crate::macros::__cold();
            unsafe {
                $crate::cuda::intrinsics::__assert_fail(
                    format_args!($($arg)+).as_str().unwrap_or("[no msg]").as_ptr() as *const u8,
                    file!().as_ptr() as *const u8,
                    line!(),
                    "[fn cannot be captured]".as_ptr() as *const u8,
                )
            };
        }
        #[cfg(not(target_arch = "nvptx64"))]
        compile_error!("Cannot call cu_assert outside PTX!");
    };
}

#[macro_export]
macro_rules! cu_panic {
    ($($arg:tt)*) => {
        #[cfg(target_arch = "nvptx64")]
        unsafe {
            $crate::cuda::intrinsics::vprintf(format_args!($($arg)*).as_str().unwrap_or("[no msg]").as_ptr() as *const u8, core::ptr::null_mut());
            $crate::cuda::intrinsics::__trap();
        }
        #[cfg(not(target_arch = "nvptx64"))]
        {
            compile_error!("Cannot call cu_panic outside PTX!");
        }
    };
}
