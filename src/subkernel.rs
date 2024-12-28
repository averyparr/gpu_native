use gpu_macro::gpu_kernel;

use crate::{FlatLayout1D, assert_universal, slices, thread_layout::ThreadLayout};

#[gpu_kernel(CUDA)]
fn new_kernel(
    a: slices::CUDASlice<f32>,
    b: slices::CUDASlice<f32>,
    mut c: slices::CUDASliceMut<f32>,
    mut d: slices::CUDASliceMut<f32>,
) {
    use core::sync::atomic::Ordering;

    let idx = FlatLayout1D.uid();

    assert_universal!((idx < a.len()) & (idx < b.len()) & (idx < c.len()) & (idx < d.len()));

    let ac = c.atomic_ref(idx.0);
    let z = ac.fetch_add(a[idx], Ordering::Relaxed);

    let ad = d.atomic_ref(idx.0);

    ad.fetch_add(z, Ordering::Relaxed);
}
