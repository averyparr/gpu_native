#[cfg(target_arch = "nvptx64")]
pub mod vector_sub_kernel {
    use crate::{FlatLayout1D, slices, thread_layout::ThreadLayout};

    pub extern "ptx-kernel" fn vector_sub_kernel(
        a_ptr: *const f64,
        a_len: u64,
        b_ptr: *const f64,
        b_len: u64,
        c_ptr: *mut f64,
        c_len: u64,
    ) {
        let a = unsafe { core::slice::from_raw_parts(a_ptr, a_len as usize) };
        let b = unsafe { core::slice::from_raw_parts(b_ptr, b_len as usize) };
        let c = unsafe { core::slice::from_raw_parts_mut(c_ptr, c_len as usize) };

        vector_sub(a.into(), b.into(), c.into());
    }

    fn vector_sub(
        a: slices::CUDASlice<f64>,
        b: slices::CUDASlice<f64>,
        mut c: slices::CUDASliceMut<f64>,
    ) {
        let idx = FlatLayout1D.uid();

        crate::assert_universal!((idx < a.len()) & (idx < b.len()) & (idx < c.len()));

        c[idx] = a[idx] - b[idx];
    }
}

#[cfg(not(any(target_arch = "nvptx64")))]
pub mod vector_sub {
    pub fn launch<'kernel>(
        grid_dim: crate::cuda_safe::GridDim1D,
        block_dim: crate::cuda_safe::BlockDim1D,
        shared_mem_bytes: u32,
        stream: crate::cuda_driver_wrapper::CUDAStream,
    ) -> impl Fn(
        crate::slices::CUDASlice<'kernel, f64>,
        crate::slices::CUDASlice<'kernel, f64>,
        crate::slices::CUDASliceMut<'kernel, f64>,
    ) -> &'kernel crate::cuda_driver_wrapper::CUDASyncObject {
        use crate::cuda_driver_wrapper::{CUDAKernel, CUDAModule};
        use std::sync::LazyLock;

        static MODULE: LazyLock<CUDAModule> = LazyLock::new(|| {
            // TODO MAKE THIS ACTUALLY DO ANYTHING TO LOAD THE SASS/PTX
            let fatbin_data = std::ptr::null_mut();
            unsafe { CUDAModule::new_from_fatbin(fatbin_data) }
                .unwrap_or_else(|e| panic!("Unable to load module: Driver error '{e:#?}'"))
        });
        static KERNEL: LazyLock<CUDAKernel> = LazyLock::new(|| {
            MODULE.get_function(c"vector_sub").unwrap_or_else(|e| {
                panic!("Unable to load `vector_sub` from module: Driver error '{e:#?}'")
            })
        });

        let launch_kernel =
            move |a: crate::slices::CUDASlice<'kernel, f64>,
                  b: crate::slices::CUDASlice<'kernel, f64>,
                  c: crate::slices::CUDASliceMut<'kernel, f64>| {
                let mut a_ptr = a.as_ptr();
                let mut a_len = a.len();
                let mut b_ptr = b.as_ptr();
                let mut b_len = b.len();
                let mut c_ptr = c.as_ptr();
                let mut c_len = c.len();

                use core::ffi::c_void;

                let mut kernel_params = [
                    &mut a_ptr as *mut _ as *mut c_void,
                    &mut a_len as *mut _ as *mut c_void,
                    &mut b_ptr as *mut _ as *mut c_void,
                    &mut b_len as *mut _ as *mut c_void,
                    &mut c_ptr as *mut _ as *mut c_void,
                    &mut c_len as *mut _ as *mut c_void,
                ];
                unsafe {
                    KERNEL
                        .launch_1d(
                            grid_dim.x as u32,
                            block_dim.x as u32,
                            shared_mem_bytes,
                            &stream,
                            &mut kernel_params,
                        )
                        .unwrap_or_else(|e| panic!("Kernel launch failed! Driver error '{e:#?}'"))
                }
                &crate::cuda_driver_wrapper::CUDASyncObject
            };

        return launch_kernel;
    }
}
