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

    pub fn vprintf(format: *const u8, valist: *const core::ffi::c_void) -> i32;

}

#[inline]
pub unsafe fn __assert_fail(message: *const u8, file: *const u8, line: u32, function: *const u8) {
    unsafe { __assertfail(message, file, line, function, 1) };
}

#[cfg(target_arch = "nvptx64")]
pub unsafe fn __trap() -> ! {
    unsafe { core::arch::asm!("trap;") }
    loop {}
}
