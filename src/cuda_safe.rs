#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
#[rustc_layout_scalar_valid_range_end(1023)]
struct ThreadIdxXY(i32);

#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
#[rustc_layout_scalar_valid_range_end(63)]
struct ThreadIdxZ(i32);

#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
#[rustc_layout_scalar_valid_range_end(2_147_483_646)]
struct BlockIdxX(i32);

#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
#[rustc_layout_scalar_valid_range_end(65_534)]
struct BlockIdxYZ(i32);

#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_layout_scalar_valid_range_end(1024)]
struct BlockDimXY(i32);

#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_layout_scalar_valid_range_end(64)]
struct BlockDimZ(i32);

#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_layout_scalar_valid_range_end(2_147_483_647)]
struct GridDimX(i32);

#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_layout_scalar_valid_range_end(65_535)]
struct GridDimYZ(i32);

impl From<ThreadIdxXY> for i32 {
    #[inline(always)]
    fn from(value: ThreadIdxXY) -> Self {
        value.0
    }
}

impl From<ThreadIdxZ> for i32 {
    #[inline(always)]
    fn from(value: ThreadIdxZ) -> Self {
        value.0
    }
}

impl From<BlockIdxX> for i32 {
    #[inline(always)]
    fn from(value: BlockIdxX) -> Self {
        value.0
    }
}

impl From<BlockIdxYZ> for i32 {
    #[inline(always)]
    fn from(value: BlockIdxYZ) -> Self {
        value.0
    }
}

impl From<BlockDimXY> for i32 {
    #[inline(always)]
    fn from(value: BlockDimXY) -> Self {
        value.0
    }
}
impl From<BlockDimZ> for i32 {
    #[inline(always)]
    fn from(value: BlockDimZ) -> Self {
        value.0
    }
}

impl From<GridDimX> for i32 {
    #[inline(always)]
    fn from(value: GridDimX) -> Self {
        value.0
    }
}

impl From<GridDimYZ> for i32 {
    #[inline(always)]
    fn from(value: GridDimYZ) -> Self {
        value.0
    }
}

#[allow(unused)]
pub struct ThreadIdx1D {
    pub x: i32,
}

#[allow(unused)]
pub struct ThreadIdx2D {
    pub x: i32,
    pub y: i32,
}

#[allow(unused)]
pub struct ThreadIdx3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[allow(unused)]
pub struct BlockIdx1D {
    pub x: i32,
}

#[allow(unused)]
pub struct BlockIdx2D {
    pub x: i32,
    pub y: i32,
}

#[allow(unused)]
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
pub struct BlockDim1D {
    pub x: i32,
}

#[allow(unused)]
pub struct BlockDim2D {
    pub x: i32,
    pub y: i32,
}

#[allow(unused)]
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

#[allow(unused)]
pub struct GridDim1D {
    pub x: i32,
}

#[allow(unused)]
pub struct GridDim2D {
    pub x: i32,
    pub y: i32,
}

#[allow(unused)]
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
                x: ThreadIdxXY(crate::cuda_intrinsics::__thread_idx_x()).into(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block_idx() -> BlockIdx1D {
        unsafe {
            BlockIdx1D {
                x: BlockIdxX(crate::cuda_intrinsics::__block_idx_x()).into(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block_dim() -> BlockDim1D {
        unsafe {
            BlockDim1D {
                x: BlockDimXY(crate::cuda_intrinsics::__block_dim_x()).into(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn grid_dim() -> GridDim1D {
        GridDim1D {
            x: unsafe { GridDimX(crate::cuda_intrinsics::__grid_dim_x()).into() },
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
                x: ThreadIdxXY(crate::cuda_intrinsics::__thread_idx_x()).into(),
                y: ThreadIdxXY(crate::cuda_intrinsics::__thread_idx_y()).into(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block_idx() -> BlockIdx2D {
        unsafe {
            BlockIdx2D {
                x: BlockIdxX(crate::cuda_intrinsics::__block_idx_x()).into(),
                y: BlockIdxYZ(crate::cuda_intrinsics::__block_idx_y()).into(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block() -> BlockDim2D {
        unsafe {
            BlockDim2D {
                x: BlockDimXY(crate::cuda_intrinsics::__block_dim_x()).into(),
                y: BlockDimXY(crate::cuda_intrinsics::__block_dim_y()).into(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn grid() -> GridDim2D {
        unsafe {
            GridDim2D {
                x: GridDimX(crate::cuda_intrinsics::__grid_dim_x()).into(),
                y: GridDimYZ(crate::cuda_intrinsics::__grid_dim_y()).into(),
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
                x: ThreadIdxXY(crate::cuda_intrinsics::__thread_idx_x()).into(),
                y: ThreadIdxXY(crate::cuda_intrinsics::__thread_idx_y()).into(),
                z: ThreadIdxZ(crate::cuda_intrinsics::__thread_idx_z()).into(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block_idx() -> BlockIdx3D {
        unsafe {
            BlockIdx3D {
                x: BlockIdxX(crate::cuda_intrinsics::__block_idx_x()).into(),
                y: BlockIdxYZ(crate::cuda_intrinsics::__block_idx_y()).into(),
                z: BlockIdxYZ(crate::cuda_intrinsics::__block_idx_z()).into(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn block() -> BlockDim3D {
        unsafe {
            BlockDim3D {
                x: BlockDimXY(crate::cuda_intrinsics::__block_dim_x()).into(),
                y: BlockDimXY(crate::cuda_intrinsics::__block_dim_y()).into(),
                z: BlockDimZ(crate::cuda_intrinsics::__block_dim_z()).into(),
            }
        }
    }
    #[inline(always)]
    #[allow(unused)]
    pub fn grid() -> GridDim3D {
        unsafe {
            GridDim3D {
                x: GridDimX(crate::cuda_intrinsics::__grid_dim_x()).into(),
                y: GridDimYZ(crate::cuda_intrinsics::__grid_dim_y()).into(),
                z: GridDimYZ(crate::cuda_intrinsics::__grid_dim_z()).into(),
            }
        }
    }
}