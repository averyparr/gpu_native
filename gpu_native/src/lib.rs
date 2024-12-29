#![cfg_attr(target_arch = "nvptx64", no_std)]
// Necessary because PTX + kernels are not really stable
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]
// LLVM intrinsics are used for e.g. getting blockDim.x from a special register.
#![allow(internal_features)]
#![feature(link_llvm_intrinsics)]

/// Atomic primitives, particularly because some core::sync::atomic types
/// do not link well, and atomic floats are not supported.
pub mod atomic;
/// Wrappers around PTX assembly + syscalls as well as the CUDA driver/runtime
pub mod cuda;
/// Core primitive types and traits for safety in the GPU compute regime.
pub mod gpu_safety;
/// Declarative macros, especially `assert_universal!` and `link_in_sass!`
pub mod macros;
/// Opaque slice types to ensure that kernel parameter passing is safe.
pub mod slices;
/// Thread layout primitives like `ThreadIdx3D` and pre-baked known-safe
/// access patterns like FlatLayout1D
pub mod thread_layout;
