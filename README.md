# `gpu_native`

## Making it easy to write GPU kernels in Rust

One of the things that has made writing GPU kernels (for me, CUDA kernels) in C++ the path of least resistance
was the syntax support that is provided when you compile with `nvcc`. It is (comparatively) _very simple_ to write
```cpp
// Implicitly runs across N >> 1 threads on a different device.
__global__ void add_kernel(f32 *a, f32 *b, f32 *c) {
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    c[gid] = a[gid] + b[gid];
}
// Runs on CPU, launches a GPU kernel
void add_cpu(f32 *a, f32 *b, f32 *c, uint64_t len) {
    uint32_t block_dim = 128;
    // because `len` may not be divisible by 128
    uint32_t grid_dim = (block_dim + block_dim - 1) / block_dim;
    add_kernel<<<block_dim, grid_dim, 0, 0>>>(a, b, c);
}
```

if you consider how much complexity is being hidden away from the user (and not what it looks like in PyTorch, `c = a + b`). 

- `nvcc` compiles for several compute platforms
- The CUDA runtime API loads device code from static/dynamic linkage
- Arguments and code are copied to the device (dynamic sized copies!)
- Threadblocks are set up and scheduled according to your configuration
- [etc]

## What does it look like with `gpu_native`?

The equivalent code in Rust looks like this: 

```rust
#[gpu_kernel(Kernel1D, CUDA | CUDASliceMut(2); CUDASlice(2))]
pub fn add_kernel(a: CUDASlice<f32>, b: CUDASlice<f32>, mut c: CUDASliceMut<f32>) {
    let idx: UniqueId = FlatLayout1D.uid();
    c[idx] = a[idx] + b[idx];
}

void add_cpu(mut a: CUDASlice<f32>, mut b: CUDASlice<f32>, mut c: CUDASliceMut<f32>) {
    let block_dim = 128;
    let grid_dim = (c.len() + block_dim - 1) / block_dim;
    add_kernel::launch(
        GridDim1D::new(grid_dim),
        BlockDim1D::new(block_dim),
        0,
        DEFAULT_STREAM.clone()
    )(
        &mut a, 
        &mut b, 
        &mut c
    );
}
```

which, to me, looks about as close as I could hope for without explicit language support (except for that `.clone()`!).
The only hitch is that `nvcc` automatically links in your device code; `rustc` doesn't support
anything like that, so I _do_ require a build script. Here, I've tried to make it as painless
as possible to get what you _probably_ want to work: 

```rust
/* build.rs */
fn main() {
    link_in_sass!(
        sm = ("sm_86"),
        host_arch = ("x86_64", "armv7"),
        cargo_rustc_args = ("--release"),
        /* optional; uses default executables otherwise */
        docker_image = "nvidia/cuda:12.6.3-devel-ubuntu24.04" 
    );
}
```

## Safety in GPU kernels?

It's arguable that memory safety is not as important for GPU workloads as it is for CPU ones. I certainly don't love paying for bounds checks if I can avoid them. 
That said, I've also spent days of my life tracking down out-of-bounds writes causing nondeterminstic test failures. To that end, generated code is (as far as I have been able to check) **safe**, but also adds **minimal** overhead within that constraint. Bounds checks _typically_ add only a single `JMP` instruction per kernel (if done carefully) and with restructuring of FFI arguments, I can consistently get the LLVM backend to generate `LDG` instructions instead of `LD` (by default). Some of the other safety checks performed:

- FFI safety: Kernel parameters are safely destructured to ensure FFI safety (that's what the `2` in `CUDASlice(2)` means)
- Mutable memory accesses can only be indexed with `UniqueID` values which are guaranteed distinct per-thread. 
- Lifetime bounding ensures that all kernel arguments remain valid

## Try `cargo expand`!

The two macros `gpu_kernel` and `link_in_sass` are really quite complex. 
Try writing out a few kernels and seeing what they expand to!
If you need a base starting case, look in `src/lib.rs`'s `test_kernel_launch_works()`.