/// PTX assembly wrappers + syscalls
pub mod intrinsics;

/// Reasonably thin safe wrapper around a subset of the
/// CUDA driver API.
#[cfg(not(any(target_arch = "nvptx64")))]
pub mod driver_wrapper;
