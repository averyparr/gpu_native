use core::ffi::c_void;

#[inline]
#[cold]
pub fn __cold() {}

#[allow(improper_ctypes)]
#[allow(dead_code)]
unsafe extern "C" {
    #[link_name = "llvm.nvvm.barrier0"]
    pub fn __syncthreads() -> ();
    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.x"]
    pub fn __block_dim_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.y"]
    pub fn __block_dim_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.z"]
    pub fn __block_dim_z() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.x"]
    pub fn __block_idx_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.y"]
    pub fn __block_idx_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.z"]
    pub fn __block_idx_z() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.x"]
    pub fn __grid_dim_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.y"]
    pub fn __grid_dim_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.z"]
    pub fn __grid_dim_z() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.x"]
    pub fn __thread_idx_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.y"]
    pub fn __thread_idx_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.z"]
    pub fn __thread_idx_z() -> i32;

    fn __assertfail(
        message: *const u8,
        file: *const u8,
        line: u32,
        function: *const u8,
        char_size: usize,
    );

    pub fn vprintf(format: *const u8, valist: *const c_void) -> i32;

}

#[inline]
pub unsafe fn __assert_fail(message: *const u8, file: *const u8, line: u32, function: *const u8) {
    unsafe { __assertfail(message, file, line, function, 1) };
}

pub unsafe fn __trap() -> ! {
    unsafe { core::arch::asm!("trap;") }
    loop {}
}

#[macro_export]
macro_rules! cu_panic {
    ($($arg:tt)*) => {{
        #[cfg(target_arch = "nvptx64")]
        unsafe {
            crate::cuda_intrinsics::vprintf(format_args!($($arg)*).as_str().unwrap_or("[no msg]").as_ptr() as *const u8, core::ptr::null_mut());
            crate::cuda_intrinsics::__trap();
        }
        #[cfg(not(target_arch = "nvptx64"))]
        compile_error!("Cannot call cu_panic outside PTX!");
    }};
}

#[macro_export]
macro_rules! cu_assert {
    ($expr: expr) => {
        #[cfg(target_arch = "nvptx64")]
        if !($expr) {
            crate::cuda_intrinsics::__cold();
            unsafe {
                crate::cuda_intrinsics::__assert_fail(
                    stringify!($cond).as_ptr() as *const u8,
                    file!().as_ptr() as *const u8,
                    line!(),
                    "[fn cannot be captured]".as_ptr() as *const u8,
                )
            };
        }
        #[cfg(not(target_arch = "nvptx64"))]
        compile_error!("Cannot call cu_assert outside PTX!");
    };
    ($expr: expr, $($arg:tt)+) => {
        #[cfg(target_arch = "nvptx64")]
        if !($expr) {
            crate::cuda_intrinsics::__cold();
            unsafe {
                crate::cuda_intrinsics::__assert_fail(
                    format_args!($($arg)+).as_str().unwrap_or("[no msg]").as_ptr() as *const u8,
                    file!().as_ptr() as *const u8,
                    line!(),
                    "[fn cannot be captured]".as_ptr() as *const u8,
                )
            };
        }
        #[cfg(not(target_arch = "nvptx64"))]
        compile_error!("Cannot call cu_assert outside PTX!");
    };
}
