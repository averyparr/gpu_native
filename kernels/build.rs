use gpu_native::link_in_sass;

fn main() {
    link_in_sass!(
        sm = ("sm_86"),
        host_arch = ("x86_64", "armv7", "aarch64"),
        cargo_rustc_args = ("--release"),
        // docker_image = "nvidia/cuda:12.6.3-devel-ubuntu24.04"
    );
}
