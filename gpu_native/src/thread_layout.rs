pub use crate::gpu_safety::ThreadLayout;

// Out-of-the-box working thread layouts which guarantee index uniqueness per-thread.

#[allow(unused)]
pub struct FlatLayout1D;

unsafe impl ThreadLayout for FlatLayout1D {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)] // Actually very important! Cross-crate inlining required.
    fn uid(&self) -> crate::gpu_safety::UniqueId {
        let uid = (Thread1D::thread_idx().x as usize)
            + (Thread1D::block_idx().x as usize) * (Thread1D::block_dim().x as usize);
        unsafe { crate::gpu_safety::UniqueId::new(uid) }
    }
}

// All the basic threadIdx, blockIdx, blockDim, gridDim objects for different kernel dimensions

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct ThreadIdx1D {
    pub x: i32,
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct ThreadIdx2D {
    pub x: i32,
    pub y: i32,
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct ThreadIdx3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct BlockIdx1D {
    pub x: i32,
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct BlockIdx2D {
    pub x: i32,
    pub y: i32,
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct BlockIdx3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl BlockIdx1D {
    #[allow(unused)]
    pub fn new(x: i32) -> Self {
        Self { x }
    }
}

impl BlockIdx2D {
    #[allow(unused)]
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

impl BlockIdx3D {
    #[allow(unused)]
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct BlockDim1D {
    pub x: i32,
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct BlockDim2D {
    pub x: i32,
    pub y: i32,
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct BlockDim3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl BlockDim1D {
    #[allow(unused)]
    pub fn new(x: i32) -> Self {
        Self { x }
    }
}

impl BlockDim2D {
    #[allow(unused)]
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

impl BlockDim3D {
    #[allow(unused)]
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

impl From<BlockDim2D> for BlockDim3D {
    fn from(value: BlockDim2D) -> Self {
        Self::new(value.x, value.y, 1)
    }
}
impl From<BlockDim1D> for BlockDim3D {
    fn from(value: BlockDim1D) -> Self {
        Self::new(value.x, 1, 1)
    }
}

impl From<BlockDim1D> for (u32, u32, u32) {
    fn from(value: BlockDim1D) -> Self {
        (value.x as u32, 1, 1)
    }
}

impl From<BlockDim2D> for (u32, u32, u32) {
    fn from(value: BlockDim2D) -> Self {
        (value.x as u32, value.y as u32, 1)
    }
}

impl From<BlockDim3D> for (u32, u32, u32) {
    fn from(value: BlockDim3D) -> Self {
        (value.x as u32, value.y as u32, value.z as u32)
    }
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct GridDim1D {
    pub x: i32,
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct GridDim2D {
    pub x: i32,
    pub y: i32,
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct GridDim3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl GridDim1D {
    #[allow(unused)]
    pub fn new(x: i32) -> Self {
        Self { x }
    }
}

impl GridDim2D {
    #[allow(unused)]
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

impl GridDim3D {
    #[allow(unused)]
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

impl From<GridDim1D> for GridDim3D {
    fn from(value: GridDim1D) -> Self {
        Self::new(value.x, 1, 1)
    }
}
impl From<GridDim2D> for GridDim3D {
    fn from(value: GridDim2D) -> Self {
        Self::new(value.x, value.y, 1)
    }
}

impl From<GridDim1D> for (u32, u32, u32) {
    fn from(value: GridDim1D) -> Self {
        (value.x as u32, 1, 1)
    }
}

impl From<GridDim2D> for (u32, u32, u32) {
    fn from(value: GridDim2D) -> Self {
        (value.x as u32, value.y as u32, 1)
    }
}

impl From<GridDim3D> for (u32, u32, u32) {
    fn from(value: GridDim3D) -> Self {
        (value.x as u32, value.y as u32, value.z as u32)
    }
}

pub struct ThreadIdx<const DIM: usize>;

#[allow(unused)]
pub type Thread1D = ThreadIdx<1>;
#[allow(unused)]
pub type Thread2D = ThreadIdx<2>;
#[allow(unused)]
pub type Thread3D = ThreadIdx<3>;

#[cfg(target_arch = "nvptx64")]
impl Thread1D {
    #[inline(always)]
    #[allow(unused)]
    pub fn thread_idx() -> ThreadIdx1D {
        unsafe {
            ThreadIdx1D {
                x: crate::cuda::intrinsics::__thread_idx_x(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block_idx() -> BlockIdx1D {
        unsafe {
            BlockIdx1D {
                x: crate::cuda::intrinsics::__block_idx_x(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block_dim() -> BlockDim1D {
        unsafe {
            BlockDim1D {
                x: crate::cuda::intrinsics::__block_dim_x(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn grid_dim() -> GridDim1D {
        unsafe {
            GridDim1D {
                x: crate::cuda::intrinsics::__grid_dim_x(),
            }
        }
    }
}

#[cfg(target_arch = "nvptx64")]
impl Thread2D {
    #[inline(always)]
    #[allow(unused)]
    pub fn idx() -> ThreadIdx2D {
        unsafe {
            ThreadIdx2D {
                x: crate::cuda::intrinsics::__thread_idx_x(),
                y: crate::cuda::intrinsics::__thread_idx_y(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block_idx() -> BlockIdx2D {
        unsafe {
            BlockIdx2D {
                x: crate::cuda::intrinsics::__block_idx_x(),
                y: crate::cuda::intrinsics::__block_idx_y(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block() -> BlockDim2D {
        unsafe {
            BlockDim2D {
                x: crate::cuda::intrinsics::__block_dim_x(),
                y: crate::cuda::intrinsics::__block_dim_y(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn grid() -> GridDim2D {
        unsafe {
            GridDim2D {
                x: crate::cuda::intrinsics::__grid_dim_x(),
                y: crate::cuda::intrinsics::__grid_dim_y(),
            }
        }
    }
}

#[cfg(target_arch = "nvptx64")]
impl Thread3D {
    #[inline(always)]
    #[allow(unused)]
    pub fn idx() -> ThreadIdx3D {
        unsafe {
            ThreadIdx3D {
                x: crate::cuda::intrinsics::__thread_idx_x(),
                y: crate::cuda::intrinsics::__thread_idx_y(),
                z: crate::cuda::intrinsics::__thread_idx_z(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block_idx() -> BlockIdx3D {
        unsafe {
            BlockIdx3D {
                x: crate::cuda::intrinsics::__block_idx_x(),
                y: crate::cuda::intrinsics::__block_idx_y(),
                z: crate::cuda::intrinsics::__block_idx_z(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block() -> BlockDim3D {
        unsafe {
            BlockDim3D {
                x: crate::cuda::intrinsics::__block_dim_x(),
                y: crate::cuda::intrinsics::__block_dim_y(),
                z: crate::cuda::intrinsics::__block_dim_z(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn grid() -> GridDim3D {
        unsafe {
            GridDim3D {
                x: crate::cuda::intrinsics::__grid_dim_x(),
                y: crate::cuda::intrinsics::__grid_dim_y(),
                z: crate::cuda::intrinsics::__grid_dim_z(),
            }
        }
    }
}
