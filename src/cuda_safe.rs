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

pub unsafe trait GPUPassable {
    type FFIRep;
    #[allow(unused)]
    fn to_ffi(&mut self) -> Self::FFIRep;
    #[allow(unused)]
    unsafe fn from_ffi(other: Self::FFIRep) -> Self;
}

pub trait TupleIndex<const TUPLE_SIZE: u8, const INDEX: u8> {
    type Ty;
}

macro_rules! tuple_trait_decl {
    ($assoc_ty: ident, $tuple_size: literal, $valid_index: literal, $($generics: ident),+) => {
        impl<$($generics,)*> TupleIndex<$tuple_size, $valid_index> for ($($generics,)*) {
            type Ty = $assoc_ty;
        }
    };
}

// 1-tuple implementations
tuple_trait_decl!(A, 1, 0, A);

// 2-tuple implementations
tuple_trait_decl!(A, 2, 0, A, B);
tuple_trait_decl!(B, 2, 1, A, B);

// 3-tuple implementations
tuple_trait_decl!(A, 3, 0, A, B, C);
tuple_trait_decl!(B, 3, 1, A, B, C);
tuple_trait_decl!(C, 3, 2, A, B, C);

// 4-tuple implementations
tuple_trait_decl!(A, 4, 0, A, B, C, D);
tuple_trait_decl!(B, 4, 1, A, B, C, D);
tuple_trait_decl!(C, 4, 2, A, B, C, D);
tuple_trait_decl!(D, 4, 3, A, B, C, D);

// 5-tuple implementations
tuple_trait_decl!(A, 5, 0, A, B, C, D, E);
tuple_trait_decl!(B, 5, 1, A, B, C, D, E);
tuple_trait_decl!(C, 5, 2, A, B, C, D, E);
tuple_trait_decl!(D, 5, 3, A, B, C, D, E);
tuple_trait_decl!(E, 5, 4, A, B, C, D, E);

// 6-tuple implementations
tuple_trait_decl!(A, 6, 0, A, B, C, D, E, F);
tuple_trait_decl!(B, 6, 1, A, B, C, D, E, F);
tuple_trait_decl!(C, 6, 2, A, B, C, D, E, F);
tuple_trait_decl!(D, 6, 3, A, B, C, D, E, F);
tuple_trait_decl!(E, 6, 4, A, B, C, D, E, F);
tuple_trait_decl!(F, 6, 5, A, B, C, D, E, F);

// 7-tuple implementations
tuple_trait_decl!(A, 7, 0, A, B, C, D, E, F, G);
tuple_trait_decl!(B, 7, 1, A, B, C, D, E, F, G);
tuple_trait_decl!(C, 7, 2, A, B, C, D, E, F, G);
tuple_trait_decl!(D, 7, 3, A, B, C, D, E, F, G);
tuple_trait_decl!(E, 7, 4, A, B, C, D, E, F, G);
tuple_trait_decl!(F, 7, 5, A, B, C, D, E, F, G);
tuple_trait_decl!(G, 7, 6, A, B, C, D, E, F, G);

// 8-tuple implementations
tuple_trait_decl!(A, 8, 0, A, B, C, D, E, F, G, H);
tuple_trait_decl!(B, 8, 1, A, B, C, D, E, F, G, H);
tuple_trait_decl!(C, 8, 2, A, B, C, D, E, F, G, H);
tuple_trait_decl!(D, 8, 3, A, B, C, D, E, F, G, H);
tuple_trait_decl!(E, 8, 4, A, B, C, D, E, F, G, H);
tuple_trait_decl!(F, 8, 5, A, B, C, D, E, F, G, H);
tuple_trait_decl!(G, 8, 6, A, B, C, D, E, F, G, H);
tuple_trait_decl!(H, 8, 7, A, B, C, D, E, F, G, H);

// 9-tuple implementations
tuple_trait_decl!(A, 9, 0, A, B, C, D, E, F, G, H, I);
tuple_trait_decl!(B, 9, 1, A, B, C, D, E, F, G, H, I);
tuple_trait_decl!(C, 9, 2, A, B, C, D, E, F, G, H, I);
tuple_trait_decl!(D, 9, 3, A, B, C, D, E, F, G, H, I);
tuple_trait_decl!(E, 9, 4, A, B, C, D, E, F, G, H, I);
tuple_trait_decl!(F, 9, 5, A, B, C, D, E, F, G, H, I);
tuple_trait_decl!(G, 9, 6, A, B, C, D, E, F, G, H, I);
tuple_trait_decl!(H, 9, 7, A, B, C, D, E, F, G, H, I);
tuple_trait_decl!(I, 9, 8, A, B, C, D, E, F, G, H, I);

// 10-tuple implementations
tuple_trait_decl!(A, 10, 0, A, B, C, D, E, F, G, H, I, J);
tuple_trait_decl!(B, 10, 1, A, B, C, D, E, F, G, H, I, J);
tuple_trait_decl!(C, 10, 2, A, B, C, D, E, F, G, H, I, J);
tuple_trait_decl!(D, 10, 3, A, B, C, D, E, F, G, H, I, J);
tuple_trait_decl!(E, 10, 4, A, B, C, D, E, F, G, H, I, J);
tuple_trait_decl!(F, 10, 5, A, B, C, D, E, F, G, H, I, J);
tuple_trait_decl!(G, 10, 6, A, B, C, D, E, F, G, H, I, J);
tuple_trait_decl!(H, 10, 7, A, B, C, D, E, F, G, H, I, J);
tuple_trait_decl!(I, 10, 8, A, B, C, D, E, F, G, H, I, J);
tuple_trait_decl!(J, 10, 9, A, B, C, D, E, F, G, H, I, J);
