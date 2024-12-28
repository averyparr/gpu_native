use core::ops::{Index, IndexMut};

use crate::{
    assert_universal,
    cuda_atomic::{AtomicF32, AtomicF64},
};

#[derive(Clone, Copy)]
pub struct UniqueId(pub usize);

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

#[repr(C)]
pub struct CUDASlice<'a, T>(&'a [T]);

#[repr(C)]
pub struct CUDASliceMut<'a, T>(&'a mut [T]);

impl<'a, T> CUDASlice<'a, T> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }
}

impl<'a, T> CUDASliceMut<'a, T> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }
}

pub trait IntoAtomic {
    type Atomic;
    unsafe fn into_atomic<'a>(ptr: *mut Self) -> &'a Self::Atomic;
}

impl IntoAtomic for f32 {
    type Atomic = AtomicF32;
    unsafe fn into_atomic<'a>(ptr: *mut Self) -> &'a Self::Atomic {
        unsafe { AtomicF32::from_ptr(ptr) }
    }
}

impl IntoAtomic for f64 {
    type Atomic = AtomicF64;
    unsafe fn into_atomic<'a>(ptr: *mut Self) -> &'a Self::Atomic {
        unsafe { AtomicF64::from_ptr(ptr) }
    }
}

impl<'a, T: IntoAtomic> CUDASliceMut<'a, T> {
    // TODO I need to think more about lifetimes here
    #[inline(always)]
    pub fn atomic_ref(&mut self, idx: usize) -> &'a T::Atomic {
        crate::assert_universal!(idx < self.0.len(), "Out of bounds for atomic ref!");
        let offset_ptr = &mut self.0[idx] as *mut _;
        unsafe { T::into_atomic(offset_ptr) }
    }
}

impl<'a, T> From<&'a [T]> for CUDASlice<'a, T> {
    #[inline(always)]
    fn from(value: &'a [T]) -> Self {
        Self(value)
    }
}

impl<'a, T> From<&'a mut [T]> for CUDASlice<'a, T> {
    #[inline(always)]
    fn from(value: &'a mut [T]) -> Self {
        Self(value)
    }
}

impl<'a, T> From<&'a mut [T]> for CUDASliceMut<'a, T> {
    #[inline(always)]
    fn from(value: &'a mut [T]) -> Self {
        Self(value)
    }
}

impl<'a, T> Index<usize> for CUDASlice<'a, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        assert_universal!(index < self.0.len());
        &self.0[index]
    }
}

impl<'a, T> Index<usize> for CUDASliceMut<'a, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        assert_universal!(index < self.0.len());
        &self.0[index]
    }
}

impl<'a, T> Index<UniqueId> for CUDASlice<'a, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: UniqueId) -> &Self::Output {
        assert_universal!(index.0 < self.0.len());
        &self.0[index.0]
    }
}

impl<'a, T> Index<UniqueId> for CUDASliceMut<'a, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: UniqueId) -> &Self::Output {
        assert_universal!(index.0 < self.0.len());
        &self.0[index.0]
    }
}

impl<'a, T> IndexMut<UniqueId> for CUDASliceMut<'a, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: UniqueId) -> &mut Self::Output {
        assert_universal!(index.0 < self.0.len());
        &mut self.0[index.0]
    }
}
