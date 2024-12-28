#![cfg_attr(target_arch = "nvptx64", no_std)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]
// I _could_ remove this if I allowed myself inline assembly. No guarantee it'd work as nicely, though.
#![allow(internal_features)]
#![feature(link_llvm_intrinsics)]
// I could remove this if I were confident that the niche optimization/etc we get from
// being able to bound e.g. ThreadIdx.x is not worth it.
#![feature(rustc_attrs)]

mod cuda_atomic;
mod cuda_intrinsics;
mod cuda_safe;
mod host_device;
pub mod other_kernel;
mod slices;
pub mod subkernel;
mod thread_layout;

//
#[cfg(not(any(target_arch = "nvptx64")))]
mod cuda_driver_wrapper;

use num_traits::float::FloatCore;
use thread_layout::ThreadLayout;

#[allow(unused)]
struct FlatLayout1D;

#[inline(always)]
#[allow(unused)]
fn do_a_div<T: FloatCore>(a: T, b: T) -> T {
    a.recip() * b
}

unsafe impl ThreadLayout for FlatLayout1D {
    #[cfg(target_arch = "nvptx64")]
    fn uid(&self) -> slices::UniqueId {
        use cuda_safe::Thread1D;

        let uid = (Thread1D::thread_idx().x as usize)
            + (Thread1D::block_idx().x as usize) * (Thread1D::block_dim().x as usize);
        slices::UniqueId(uid)
    }
}

#[cfg(not(any(target_arch = "nvptx64")))]
pub fn ready() {
    use cuda_safe::{BlockDim1D, GridDim1D};
    use slices::{CUDASlice, CUDASliceMut};
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

    // I do *not* know whether this is safe, even _if_ the slices were on-device.
    vector_add::launch(
        GridDim1D::new(1),
        BlockDim1D::new(1),
        0,
        cuda_driver_wrapper::DEFAULT_STREAM.clone(),
    )(&mut x_sl, &mut y_sl, &mut z_sl, &mut w_sl);
}

#[cfg(target_arch = "nvptx64")]
pub mod vector_add_kernel {
    use super::*;
    pub extern "ptx-kernel" fn vector_add_kernel(
        a_ptr: *const f32,
        a_len: u64,
        b_ptr: *const f32,
        b_len: u64,
        c_ptr: *mut f32,
        c_len: u64,
        d_ptr: *mut f32,
        d_len: u64,
    ) {
        let a = unsafe { core::slice::from_raw_parts(a_ptr, a_len as usize) };
        let b = unsafe { core::slice::from_raw_parts(b_ptr, b_len as usize) };
        let c = unsafe { core::slice::from_raw_parts_mut(c_ptr, c_len as usize) };
        let d = unsafe { core::slice::from_raw_parts_mut(d_ptr, d_len as usize) };

        vector_add(a.into(), b.into(), c.into(), d.into());
    }

    #[inline(always)]
    fn vector_add(
        a: slices::CUDASlice<f32>,
        b: slices::CUDASlice<f32>,
        mut c: slices::CUDASliceMut<f32>,
        mut d: slices::CUDASliceMut<f32>,
    ) {
        let idx = FlatLayout1D.uid();

        assert_universal!((idx < a.len()) & (idx < b.len()) & (idx < c.len()) & (idx < d.len()));

        c[idx] = do_a_div(a[idx], b[idx]);
    }
}

#[cfg(not(any(target_arch = "nvptx64")))]
pub mod cuda_device_code {
    unsafe extern "Rust" {
        #[link_name = concat!(env!("CARGO_PKG_NAME"), "_FATBIN_CODE", "_7B4EA9D2")]
        static FATBIN_DATA: &'static [u8];
    }

    pub fn fatbin_data() -> &'static [u8] {
        let arr = unsafe { FATBIN_DATA };
        let mut first_8 = [0; 8];
        first_8
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = arr[i]);
        let maybe_magic = u64::from_le_bytes(first_8);
        assert_eq!(
            maybe_magic, 0xB0BACAFEB0BACAFE,
            "Linked file does not begin with magic value!"
        );
        &arr[8..]
    }
}

#[cfg(not(any(target_arch = "nvptx64")))]
pub mod vector_add {
    use crate::cuda_device_code;

    pub fn launch<'kernel>(
        grid_dim: crate::cuda_safe::GridDim1D,
        block_dim: crate::cuda_safe::BlockDim1D,
        shared_mem_bytes: u32,
        stream: crate::cuda_driver_wrapper::CUDAStream,
    ) -> impl Fn(
        &'kernel mut crate::slices::CUDASlice<f32>,
        &'kernel mut crate::slices::CUDASlice<f32>,
        &'kernel mut crate::slices::CUDASliceMut<f32>,
        &'kernel mut crate::slices::CUDASliceMut<f32>,
    ) -> &'kernel crate::cuda_driver_wrapper::CUDASyncObject {
        use crate::cuda_driver_wrapper::{CUDAKernel, CUDAModule};
        use std::sync::LazyLock;

        static MODULE: LazyLock<CUDAModule> = LazyLock::new(|| {
            let fatbin_data = cuda_device_code::fatbin_data();
            unsafe { CUDAModule::new_from_fatbin(fatbin_data as *const _ as *mut _) }
                .unwrap_or_else(|e| panic!("Unable to load module: Driver error '{e:#?}'"))
        });
        static KERNEL: LazyLock<CUDAKernel> = LazyLock::new(|| {
            MODULE.get_function(c"vector_add").unwrap_or_else(|e| {
                panic!("Unable to load `vector_add` from module: Driver error '{e:#?}'")
            })
        });

        let launch_kernel =
            move |a: &'kernel mut crate::slices::CUDASlice<f32>,
                  b: &'kernel mut crate::slices::CUDASlice<f32>,
                  c: &'kernel mut crate::slices::CUDASliceMut<f32>,
                  d: &'kernel mut crate::slices::CUDASliceMut<f32>| {
                let mut a_ptr = a.as_ptr();
                let mut a_len = a.len();
                let mut b_ptr = b.as_ptr();
                let mut b_len = b.len();
                let mut c_ptr = c.as_ptr();
                let mut c_len = c.len();
                let mut d_ptr = d.as_ptr();
                let mut d_len = d.len();

                use core::ffi::c_void;

                let mut kernel_params = [
                    &mut a_ptr as *mut _ as *mut c_void,
                    &mut a_len as *mut _ as *mut c_void,
                    &mut b_ptr as *mut _ as *mut c_void,
                    &mut b_len as *mut _ as *mut c_void,
                    &mut c_ptr as *mut _ as *mut c_void,
                    &mut c_len as *mut _ as *mut c_void,
                    &mut d_ptr as *mut _ as *mut c_void,
                    &mut d_len as *mut _ as *mut c_void,
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
