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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_kernel_launch_works() {
        let arr_len = 1024;
        type TA = f32;
        use gpu_native::cuda::driver_wrapper as driver;
        let host_a: Vec<_> = (0..arr_len).map(|i| (i as f32) * 0.3).collect();
        let host_b: Vec<_> = (0..arr_len).map(|i| (i as f32) * 0.1).collect();
        let host_c: Vec<_> = host_a
            .iter()
            .zip(host_b.iter())
            .map(|(a, b)| a + b)
            .collect();
        let mut a_buf = driver::malloc_sync::<TA>(arr_len).expect("Alloc a error");
        let mut b_buf = driver::malloc_sync::<TA>(arr_len).expect("Alloc b error");
        let mut c_buf = driver::malloc_sync::<TA>(arr_len).expect("Alloc c error");
        a_buf.copy_htod(host_a.as_slice()).expect("htod a failure");
        b_buf.copy_htod(host_b.as_slice()).expect("htod b failure");

        let mut a_slice = a_buf.as_slice();
        let mut b_slice = b_buf.as_slice();
        let mut c_slice = c_buf.as_mut_slice();

        kernels::add_kernel::launch(
            GridDim1D::new(1),
            BlockDim1D::new(1024),
            0,
            DEFAULT_STREAM.clone(),
        )(&mut a_slice, &mut b_slice, &mut c_slice);
        let mut from_gpu_c = vec![0.0; arr_len];
        c_buf.copy_dtoh(&mut from_gpu_c).expect("dtoh c failure");
        assert_eq!(from_gpu_c, host_c);
    }
}
