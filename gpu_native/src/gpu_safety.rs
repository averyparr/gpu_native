/// Typed wrapper which is used to signify that no other thread
/// has the same ID.
#[derive(Clone, Copy)]
pub struct UniqueId(usize);

impl UniqueId {
    #[inline(always)]
    pub unsafe fn new(idx: usize) -> Self {
        Self(idx)
    }
    #[inline(always)]
    pub fn idx(&self) -> usize {
        self.0
    }
}

impl From<UniqueId> for usize {
    fn from(value: UniqueId) -> Self {
        value.idx()
    }
}

impl PartialEq<usize> for UniqueId {
    #[inline(always)]
    fn eq(&self, other: &usize) -> bool {
        self.0 == *other
    }
    #[inline(always)]
    fn ne(&self, other: &usize) -> bool {
        self.0 != *other
    }
}

impl PartialOrd<usize> for UniqueId {
    #[inline(always)]
    fn ge(&self, other: &usize) -> bool {
        self.0 >= *other
    }
    #[inline(always)]
    fn gt(&self, other: &usize) -> bool {
        self.0 > *other
    }
    #[inline(always)]
    fn le(&self, other: &usize) -> bool {
        self.0 <= *other
    }
    #[inline(always)]
    fn lt(&self, other: &usize) -> bool {
        self.0 < *other
    }
    #[inline(always)]
    fn partial_cmp(&self, other: &usize) -> Option<core::cmp::Ordering> {
        if self.0 == *other {
            Some(core::cmp::Ordering::Equal)
        } else if self.0 < *other {
            Some(core::cmp::Ordering::Less)
        } else {
            Some(core::cmp::Ordering::Greater)
        }
    }
}

/// This trait allows users to promise that their type is able
/// to produce unique by-thread values (i.e. no two threads will)
/// produce the same value. Keep in mind that this property must hold
/// for _all_ values of the type.
#[allow(unused)]
pub unsafe trait ThreadLayout {
    #[cfg(target_arch = "nvptx64")]
    fn uid(&self) -> UniqueId;

    #[cfg(not(any(target_arch = "nvptx64")))]
    fn uid(&self) -> UniqueId {
        panic!("Attempting to call ThreadLayout::uid() from host code!");
    }
}

/// **FFIRep should be a tuple type, or gpu kernel macros will fail**
///
/// This trait allows users to promise that their type is able (potentially
/// after decomposing it into its component pieces) to be passed to a kernel.
/// Primitives and our own Slice types should be fine.
///
/// We use this because passing e.g. a struct{*mut T, usize} vs a *mut T and usize
/// and then recomposing them actually produce different LLVM IR (and importantly
/// lead to LD.E instructions rather than LDG.E instructions in generated SASS).
/// This is because the former essentially treats the struct as a serialized data
/// format and then extracting pointers from that; this loses ptr provenance.
pub unsafe trait GPUPassable {
    type FFIRep;
    #[allow(unused)]
    fn to_ffi(&mut self) -> Self::FFIRep;
    #[allow(unused)]
    unsafe fn from_ffi(other: Self::FFIRep) -> Self;
}

/// Utility trait used to ensure kernel argument destructuring/passing is as safe
/// as I can make it (i.e. the right number of entries are put in the right places).
#[allow(unused)]
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
