use gpu_native::{
    cuda::driver_wrapper::DEFAULT_STREAM,
    slices::{CUDASlice, CUDASliceMut},
    thread_layout::{BlockDim1D, GridDim1D},
};

pub fn working_kernel_launch() {
    // Note that this isn't yet safe because we're going to be referencing
    // CPU memory from the GPU, but proper safety can be later enforced using
    // these types.
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
    kernels::separate_kernel::launch(
        GridDim1D::new(1),
        BlockDim1D::new(1),
        0,
        DEFAULT_STREAM.clone(),
    )(&mut x_sl, &mut y_sl, &mut z_sl, &mut w_sl);
}
