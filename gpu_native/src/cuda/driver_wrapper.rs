use core::ffi::CStr;
use core::ffi::c_void;
use cudarc::driver::DriverError;
use cudarc::driver::result as cuda;
use cudarc::driver::sys as cuda_sys;
use std::sync::Arc;
use std::sync::LazyLock;

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
        let fn_ptr = std::ptr::null_mut();
        unsafe {
            cuda_sys::lib().cuModuleGetFunction(fn_ptr, *self.0, fn_name.as_ptr() as *const _)
        }
        .result()?;
        Ok(CUDAKernel(Arc::new(unsafe { *fn_ptr })))
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
    let ctx: *mut cuda_sys::CUcontext = std::ptr::null_mut();
    let result = unsafe { cuda_sys::lib().cuCtxCreate_v2(ctx, 0, *DEFAULT_DEVICE) };
    if result != cuda_sys::cudaError_enum::CUDA_SUCCESS {
        panic!("Unable to establish default ctx: driver error '{result:#?}'");
    }
    CUDAContext(Arc::new(unsafe { *ctx }))
});

#[allow(unused)]
pub static DEFAULT_STREAM: LazyLock<CUDAStream> = LazyLock::new(|| {
    assert!(!DEFAULT_CTX.0.is_null());
    let stream = cuda::stream::create(cuda::stream::StreamKind::Default)
        .unwrap_or_else(|e| panic!("Unable to create default stream: Driver error '{e:#?}'"));
    CUDAStream(Arc::new(stream))
});

pub struct CUDASyncObject;
