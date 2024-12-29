use core::ffi::CStr;
use core::ffi::c_void;
use core::mem::MaybeUninit;
use cudarc::driver::DriverError;
use cudarc::driver::result as cuda;
use cudarc::driver::sys as cuda_sys;
use std::sync::Arc;
use std::sync::LazyLock;

use crate::slices::CUDASlice;
use crate::slices::CUDASliceMut;

#[derive(Clone)]
pub struct CUDAContext(Arc<cudarc::driver::sys::CUcontext>);
unsafe impl Send for CUDAContext {}
unsafe impl Sync for CUDAContext {}

impl CUDAContext {
    #[allow(unused)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

#[derive(Clone)]
pub struct CUDAModule(Arc<cudarc::driver::sys::CUmodule>);
unsafe impl Send for CUDAModule {}
unsafe impl Sync for CUDAModule {}

impl CUDAModule {
    #[allow(unused)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    pub unsafe fn new_from_fatbin(fatbin_data: *mut c_void) -> Result<Self, DriverError> {
        let raw_ptr = unsafe { cuda::module::load_data(fatbin_data) }?;
        Ok(Self(Arc::new(raw_ptr)))
    }

    pub fn get_function(&self, fn_name: &CStr) -> Result<CUDAKernel, DriverError> {
        let mut fn_ptr = MaybeUninit::uninit();
        unsafe {
            cuda_sys::lib().cuModuleGetFunction(
                fn_ptr.as_mut_ptr(),
                *self.0,
                fn_name.as_ptr() as *const _,
            )
        }
        .result()?;
        Ok(CUDAKernel(Arc::new(unsafe { fn_ptr.assume_init() })))
    }
}

#[derive(Clone)]
pub struct CUDAKernel(Arc<cudarc::driver::sys::CUfunction>);
unsafe impl Send for CUDAKernel {}
unsafe impl Sync for CUDAKernel {}

impl CUDAKernel {
    #[allow(unused)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    #[allow(unused)]
    pub unsafe fn launch(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: &CUDAStream,
        kernel_params: &mut [*mut c_void],
    ) -> Result<(), DriverError> {
        unsafe {
            cuda::launch_kernel(
                *self.0,
                grid_dim,
                block_dim,
                shared_mem_bytes,
                *stream.0,
                kernel_params,
            )
        }
    }

    #[allow(unused)]
    pub unsafe fn launch_1d(
        &self,
        grid_dim: u32,
        block_dim: u32,
        shared_mem_bytes: u32,
        stream: &CUDAStream,
        kernel_params: &mut [*mut c_void],
    ) -> Result<(), DriverError> {
        unsafe {
            self.launch(
                (grid_dim, 1, 1),
                (block_dim, 1, 1),
                shared_mem_bytes,
                stream,
                kernel_params,
            )
        }
    }

    #[allow(unused)]
    pub unsafe fn launch_2d(
        &self,
        grid_dim: (u32, u32),
        block_dim: (u32, u32),
        shared_mem_bytes: u32,
        stream: &CUDAStream,
        kernel_params: &mut [*mut c_void],
    ) -> Result<(), DriverError> {
        unsafe {
            self.launch(
                (grid_dim.0, grid_dim.1, 1),
                (block_dim.0, block_dim.1, 1),
                shared_mem_bytes,
                stream,
                kernel_params,
            )
        }
    }
}

#[derive(Clone)]
pub struct CUDAStream(Arc<cudarc::driver::sys::CUstream>);
unsafe impl Send for CUDAStream {}
unsafe impl Sync for CUDAStream {}

impl CUDAStream {
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

#[allow(unused)]
pub static DRIVER_INIT: LazyLock<bool> = LazyLock::new(|| {
    cuda::init().unwrap_or_else(|e| panic!("CUDA driver error: '{e:#?}'"));

    true
});

#[allow(unused)]
pub static DEFAULT_DEVICE: LazyLock<i32> = LazyLock::new(|| {
    assert!(*DRIVER_INIT);
    let device_count =
        cuda::device::get_count().unwrap_or_else(|e| panic!("CUDA driver error: '{e:#?}'"));
    assert!(device_count > 0, "Must have at least one CUDA device.");
    let dev = cuda::device::get(0).unwrap_or_else(|e| panic!("CUDA driver error: '{e:#?}'"));
    dev
});

#[allow(unused)]
pub static DEFAULT_CTX: LazyLock<CUDAContext> = LazyLock::new(|| {
    assert!(*DRIVER_INIT);
    assert!(*DEFAULT_DEVICE > -1);
    let mut ctx: MaybeUninit<cuda_sys::CUcontext> = MaybeUninit::uninit();
    let reset_res = unsafe { cuda_sys::lib().cuDevicePrimaryCtxReset_v2(0) };
    match reset_res {
        cuda_sys::CUresult::CUDA_SUCCESS => {}
        _ => {
            panic!("Unable to reset default ctx: driver error '{reset_res:#?}'");
        }
    }
    let result = unsafe { cuda_sys::lib().cuCtxCreate_v2(ctx.as_mut_ptr(), 0, *DEFAULT_DEVICE) };
    if result != cuda_sys::cudaError_enum::CUDA_SUCCESS {
        panic!("Unable to establish default ctx: driver error '{result:#?}'");
    }
    CUDAContext(Arc::new(unsafe { ctx.assume_init() }))
});

#[allow(unused)]
pub static DEFAULT_STREAM: LazyLock<CUDAStream> = LazyLock::new(|| {
    assert!(!DEFAULT_CTX.0.is_null());
    let stream = cuda::stream::create(cuda::stream::StreamKind::Default)
        .unwrap_or_else(|e| panic!("Unable to create default stream: Driver error '{e:#?}'"));
    CUDAStream(Arc::new(stream))
});

pub struct CUDASyncObject;

pub struct CudaBuffer<T>(*mut T, usize, Option<CUDAStream>);

impl<T> CudaBuffer<T> {
    pub fn as_slice<'a>(&'a self) -> CUDASlice<'a, T> {
        // TODO: figure out how to remove.
        // See the conditional reference in host code...
        assert!(
            self.2.is_none(),
            "Async buffers aren't yet supported with slices."
        );
        // Safety: I hold on to a valid pointer at a contiguous range of elements.
        //  I am valid for at least 'a.
        CUDASlice(unsafe { std::slice::from_raw_parts(self.0, self.1) })
    }

    pub fn as_mut_slice<'a>(&'a mut self) -> CUDASliceMut<'a, T> {
        // TODO: figure out how to remove.
        // See the conditional reference in host code...
        assert!(
            self.2.is_none(),
            "Async buffers aren't yet supported with slices."
        );
        // Safety: I hold on to a valid pointer at a contiguous range of elements.
        //  I am valid for at least 'a.
        CUDASliceMut(unsafe { std::slice::from_raw_parts_mut(self.0, self.1) })
    }

    pub fn copy_htod(&mut self, buf: &[T]) -> Result<(), DriverError> {
        assert!(
            self.1 == buf.len(),
            "Attempted to `copy_htod` between different-sized buffers"
        );
        unsafe { cudarc::driver::result::memcpy_htod_sync(self.0 as u64, buf) }
    }

    pub fn copy_htod_async(&mut self, buf: &[T], stream: &CUDAStream) -> Result<(), DriverError> {
        assert!(
            self.1 == buf.len(),
            "Attempted to `copy_htod` between different-sized buffers"
        );
        unsafe { cudarc::driver::result::memcpy_htod_async(self.0 as u64, buf, *stream.0) }
    }

    pub fn copy_dtoh(&self, buf: &mut [T]) -> Result<(), DriverError> {
        assert!(
            self.1 == buf.len(),
            "Attempted to `copy_dtoh` between different-sized buffers"
        );
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(buf, self.0 as u64) }
    }

    pub fn copy_dtoh_async(&self, buf: &mut [T], stream: &CUDAStream) -> Result<(), DriverError> {
        assert!(
            self.1 == buf.len(),
            "Attempted to `copy_dtoh` between different-sized buffers"
        );
        unsafe { cudarc::driver::result::memcpy_dtoh_async(buf, self.0 as u64, *stream.0) }
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        match &self.2 {
            Some(stream) => unsafe { cudarc::driver::result::free_async(self.0 as u64, *stream.0) },
            None => unsafe { cudarc::driver::result::free_sync(self.0 as u64) },
        }
        .expect("cudaFree has failed. There's no way to return an error in Drop.");
    }
}

pub fn malloc_async<T>(
    stream: &CUDAStream,
    num_elements: usize,
) -> Result<CudaBuffer<T>, DriverError> {
    assert!(!DEFAULT_CTX.is_null());

    // Safety: DEFAULT_STREAM is non-null, initialized CUDA
    // stream because it doesn't allow modification and is
    // initailized before first use.
    // We otherwise defer trust to the CUDA driver.
    let ptr = unsafe {
        cudarc::driver::result::malloc_async(*stream.0, num_elements * std::mem::size_of::<T>())
    }?;

    // Safety: If there is no driver error, then malloc_async
    // has returned a valid address to device memory.
    let ptr: *mut T = unsafe { std::mem::transmute(ptr) };

    Ok(CudaBuffer(ptr, num_elements, Some(stream.clone())))
}

pub fn malloc_sync<T>(num_elements: usize) -> Result<CudaBuffer<T>, DriverError> {
    assert!(!DEFAULT_CTX.is_null());

    // Safety: DEFAULT_STREAM is non-null, initialized CUDA
    // stream because it doesn't allow modification and is
    // initailized before first use.
    // We otherwise defer trust to the CUDA driver.
    let ptr =
        unsafe { cudarc::driver::result::malloc_sync(num_elements * std::mem::size_of::<T>()) }?;

    // Safety: If there is no driver error, then malloc_async
    // has returned a valid address to device memory.
    let ptr: *mut T = unsafe { std::mem::transmute(ptr) };

    Ok(CudaBuffer(ptr, num_elements, None))
}
