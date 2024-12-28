use gpu_macro::{device, gpu_kernel, host};

#[device]
use crate::{FlatLayout1D, assert_universal, thread_layout::ThreadLayout};

use crate::slices::{CUDASlice, CUDASliceMut};

#[host]
use crate::cuda_driver_wrapper::DEFAULT_STREAM;

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

    let ac = c.atomic_ref(idx.0);
    let z = ac.fetch_add(a[idx], Ordering::Relaxed);

    let ad = d.atomic_ref(idx.0);

    ad.fetch_add(z, Ordering::Relaxed);
}

#[host]
pub fn working_kernel_launch() {
    // Note that this isn't yet safe because we're going to be referencing
    // CPU memory from the GPU, but proper safety can be later enforced using
    // these types.
    use crate::cuda_safe::{BlockDim1D, GridDim1D};
    use crate::slices::{CUDASlice, CUDASliceMut};
    let x = [1.0, 2.0];
    let y = [1.0, 2.4];
    let mut z = [1.0, 2.2];
    let mut w = [1.0, 2.1];

    let (mut x_sl, mut y_sl, mut z_sl, mut w_sl) = (
        CUDASlice::from(x.as_slice()),
        CUDASlice::from(y.as_slice()),
        CUDASliceMut::from(z.as_mut_slice()),
        CUDASliceMut::from(w.as_mut_slice()),
    );
    separate_kernel::launch(
        GridDim1D::new(1),
        BlockDim1D::new(1),
        0,
        DEFAULT_STREAM.clone(),
    )(&mut x_sl, &mut y_sl, &mut z_sl, &mut w_sl);
}
