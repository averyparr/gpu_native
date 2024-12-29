#![cfg_attr(target_arch = "nvptx64", no_std)]
#![feature(abi_ptx)]

use gpu_macro::{device, gpu_kernel};

#[device]
use gpu_native::{assert_universal, thread_layout::FlatLayout1D, thread_layout::ThreadLayout};

use gpu_native::slices::{CUDASlice, CUDASliceMut};

#[gpu_kernel(Kernel1D, CUDA | CUDASliceMut(2); CUDASlice(2))]
pub fn separate_kernel(
    a: CUDASlice<f32>,
    b: CUDASlice<f32>,
    mut c: CUDASliceMut<f32>,
    mut d: CUDASliceMut<f32>,
) {
    use core::sync::atomic::Ordering;

    let idx = FlatLayout1D.uid();

    assert_universal!((idx < a.len()) & (idx < b.len()) & (idx < c.len()) & (idx < d.len()));

    let ac = c.atomic_ref(idx);
    let z = ac.fetch_add(a[idx], Ordering::Relaxed);

    let ad = d.atomic_ref(idx.idx() + 5);

    ad.fetch_add(z, Ordering::Relaxed);
}

#[gpu_kernel(Kernel1D, CUDA | CUDASliceMut(2); CUDASlice(2))]
pub fn add_kernel(a: CUDASlice<f32>, b: CUDASlice<f32>, mut c: CUDASliceMut<f32>) {
    let idx = FlatLayout1D.uid();
    c[idx] = a[idx] + b[idx];
}
