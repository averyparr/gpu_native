use core::{cell::UnsafeCell, sync::atomic::Ordering};

#[allow(unused, non_camel_case_types)]
type s32 = i32;
#[allow(unused, non_camel_case_types)]
type s64 = i32;
#[allow(unused, non_camel_case_types)]
type b32 = u32;
#[allow(unused, non_camel_case_types)]
type b64 = u64;

macro_rules! impl_atomic_op {
    ($name:ident, $op:ident, $type:ty, $reg_ty: ident) => {
        #[cfg(target_arch = "nvptx64")]
        #[inline(always)]
        #[allow(unused)]
        pub unsafe fn $name(ptr: *mut $type, val: $type, order: Ordering) -> $type {
            let old_val: $type;
            match order {
                Ordering::Relaxed => unsafe {
                    core::arch::asm!(
                        concat!("atom.relaxed.global.", stringify!($op), ".", stringify!($type), " {0}, [{1}], {2};"),
                        out($reg_ty) old_val,
                        in(reg64) ptr,
                        in($reg_ty) val,
                    )
                },
                Ordering::Acquire => unsafe {
                    core::arch::asm!(
                        concat!("atom.acquire.global.", stringify!($op), ".", stringify!($type), " {0}, [{1}], {2};"),
                        out($reg_ty) old_val,
                        in(reg64) ptr,
                        in($reg_ty) val,
                    )
                },
                Ordering::Release => unsafe {
                    core::arch::asm!(
                        concat!("atom.release.global.", stringify!($op), ".", stringify!($type), " {0}, [{1}], {2};"),
                        out($reg_ty) old_val,
                        in(reg64) ptr,
                        in($reg_ty) val,
                    )
                },
                Ordering::AcqRel => unsafe {
                    core::arch::asm!(
                        concat!("atom.acq_rel.global.", stringify!($op), ".", stringify!($type), " {0}, [{1}], {2};"),
                        out($reg_ty) old_val,
                        in(reg64) ptr,
                        in($reg_ty) val,
                    )
                },
                Ordering::SeqCst => {
                    crate::cu_panic!("SeqCst is not supported on CUDA devices.");
                },
                _ => {
                    crate::cu_panic!("Unsupported ordering!");
                },
            };
            old_val
        }
    };
}

impl_atomic_op!(atomic_add_f64, add, f64, reg64);
impl_atomic_op!(atomic_add_f32, add, f32, reg32);

impl_atomic_op!(atomic_min_f32, min, f32, reg32);
impl_atomic_op!(atomic_min_f64, min, f64, reg64);

impl_atomic_op!(atomic_max_f32, max, f32, reg32);
impl_atomic_op!(atomic_max_f64, max, f64, reg64);

#[repr(C, align(8))]
pub struct AtomicF64 {
    v: UnsafeCell<f64>,
}

#[repr(C, align(4))]
pub struct AtomicF32 {
    v: UnsafeCell<f32>,
}

trait UnsignedFloatStorage: Copy {
    type StorageT: Copy;
    type AtomicStorageT;
    fn shim_to_bits(&self) -> Self::StorageT;
    fn shim_from_bits(bit_rep: Self::StorageT) -> Self;
    unsafe fn shim_from_ptr<'a>(ptr: *mut Self::StorageT) -> &'a Self::AtomicStorageT;
    fn shim_compare_exchange_weak(
        param: &Self::AtomicStorageT,
        current: Self::StorageT,
        new: Self::StorageT,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::StorageT, Self::StorageT>;
}

#[cfg(not(any(target_arch = "nvptx64")))]
impl UnsignedFloatStorage for f32 {
    type StorageT = u32;
    type AtomicStorageT = core::sync::atomic::AtomicU32;
    fn shim_from_bits(bit_rep: Self::StorageT) -> Self {
        Self::from_bits(bit_rep)
    }
    fn shim_to_bits(&self) -> Self::StorageT {
        self.to_bits()
    }
    unsafe fn shim_from_ptr<'a>(ptr: *mut Self::StorageT) -> &'a Self::AtomicStorageT {
        unsafe { Self::AtomicStorageT::from_ptr(ptr) }
    }
    fn shim_compare_exchange_weak(
        param: &Self::AtomicStorageT,
        current: Self::StorageT,
        new: Self::StorageT,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::StorageT, Self::StorageT> {
        param.compare_exchange_weak(current, new, success, failure)
    }
}

#[cfg(not(any(target_arch = "nvptx64")))]
impl UnsignedFloatStorage for f64 {
    type StorageT = u64;
    type AtomicStorageT = core::sync::atomic::AtomicU64;
    fn shim_from_bits(bit_rep: Self::StorageT) -> Self {
        Self::from_bits(bit_rep)
    }
    fn shim_to_bits(&self) -> Self::StorageT {
        self.to_bits()
    }
    unsafe fn shim_from_ptr<'a>(ptr: *mut Self::StorageT) -> &'a Self::AtomicStorageT {
        unsafe { Self::AtomicStorageT::from_ptr(ptr) }
    }
    fn shim_compare_exchange_weak(
        param: &Self::AtomicStorageT,
        current: Self::StorageT,
        new: Self::StorageT,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::StorageT, Self::StorageT> {
        param.compare_exchange_weak(current, new, success, failure)
    }
}

/// Generic backing implementation for an AtomicT(UnsafeCell<T>)
/// with a binary `op` taking `(orig, other)` and returning the value
/// we would like to be stored to the address should things work out.
///
/// We expect this to be substantially slower than hardware-supported
/// operations should they exist.
#[allow(unused)]
fn backing_float_impl<T: UnsignedFloatStorage>(
    cell_rep: &UnsafeCell<T>,
    val: T,
    op: impl Fn(T, T) -> T,
    order: Ordering,
) -> T {
    {
        #[cfg(target_arch = "nvptx64")]
        crate::assert_universal!(
            order != Ordering::SeqCst,
            "SeqCst not supported on CUDA devices!"
        );
        let failure = match order {
            Ordering::AcqRel | Ordering::Acquire => Ordering::Acquire,
            Ordering::Release | Ordering::Relaxed => Ordering::Relaxed,
            Ordering::SeqCst => Ordering::SeqCst,
            _ => {
                super::panic_universal!("Unsupported atomic ordering!");
            }
        };
        let v = unsafe { <T as UnsignedFloatStorage>::shim_from_ptr(cell_rep.get() as *mut _) };
        let mut f_bits: T::StorageT = unsafe { *cell_rep.get() }.shim_to_bits();
        loop {
            let f_old = T::shim_from_bits(f_bits);
            let f_new = op(f_old, val).shim_to_bits();
            let store_res = T::shim_compare_exchange_weak(v, f_bits, f_new, order, failure);
            match store_res {
                Ok(_) => break f_old,
                Err(new_old) => {
                    f_bits = new_old;
                }
            }
        }
    }
}

impl AtomicF64 {
    #[inline(always)]
    #[allow(unused)]
    pub fn new(v: f64) -> Self {
        Self {
            v: UnsafeCell::new(v),
        }
    }

    /// # Safety
    ///
    /// * `ptr` must be aligned to align_of::<AtomicF64>
    /// * `ptr` must be [valid] for both reads and writes for the whole lifetime `'a`.
    /// * You must adhere to the [Memory model for atomic accesses]. In particular, it is not
    ///   allowed to mix atomic and non-atomic accesses, or atomic accesses of different sizes,
    ///   without synchronization.
    ///
    /// [Memory model for atomic accesses]: https://doc.rust-lang.org/std/sync/atomic/#memory-model-for-atomic-accesses
    #[inline(always)]
    #[allow(unused)]
    pub const unsafe fn from_ptr<'a>(ptr: *mut f64) -> &'a Self {
        // SAFETY: guaranteed by the caller
        unsafe { &*ptr.cast() }
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn get_mut(&mut self) -> &mut f64 {
        self.v.get_mut()
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn from_mut(v: &mut f64) -> &mut Self {
        let [] = [(); align_of::<Self>() - align_of::<f64>()];
        let [] = [(); align_of::<f64>() - align_of::<Self>()];
        // SAFETY:
        //  - the mutable reference guarantees unique ownership.
        //  - the alignment of `f64` and `Self` is the
        //    same, as verified above
        unsafe { &mut *(v as *mut f64 as *mut Self) }
    }

    #[inline(always)]
    #[allow(unused)]
    pub const fn into_inner(self) -> f64 {
        self.v.into_inner()
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn fetch_add(&self, val: f64, order: Ordering) -> f64 {
        #[cfg(target_arch = "nvptx64")]
        unsafe {
            return atomic_add_f64(self.v.get(), val, order);
        }
        #[cfg(not(any(target_arch = "nvptx64")))]
        backing_float_impl(&self.v, val, |x, y| x + y, order)
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn fetch_sub(&self, val: f64, order: Ordering) -> f64 {
        self.fetch_add(-val, order)
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn fetch_min(&self, val: f64, order: Ordering) -> f64 {
        #[cfg(target_arch = "nvptx64")]
        unsafe {
            return atomic_min_f64(self.v.get(), val, order);
        }
        #[cfg(not(any(target_arch = "nvptx64")))]
        backing_float_impl(&self.v, val, |x, y| x.min(y), order)
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn fetch_max(&self, val: f64, order: Ordering) -> f64 {
        #[cfg(target_arch = "nvptx64")]
        unsafe {
            return atomic_max_f64(self.v.get(), val, order);
        }
        #[cfg(not(any(target_arch = "nvptx64")))]
        backing_float_impl(&self.v, val, |x, y| x.max(y), order)
    }
}

impl AtomicF32 {
    #[inline(always)]
    #[allow(unused)]
    pub fn new(v: f32) -> Self {
        Self {
            v: UnsafeCell::new(v),
        }
    }

    /// # Safety
    ///
    /// * `ptr` must be aligned to align_of::<AtomicF32>
    /// * `ptr` must be [valid] for both reads and writes for the whole lifetime `'a`.
    /// * You must adhere to the [Memory model for atomic accesses]. In particular, it is not
    ///   allowed to mix atomic and non-atomic accesses, or atomic accesses of different sizes,
    ///   without synchronization.
    ///
    /// [Memory model for atomic accesses]: https://doc.rust-lang.org/std/sync/atomic/#memory-model-for-atomic-accesses
    #[inline(always)]
    #[allow(unused)]
    pub const unsafe fn from_ptr<'a>(ptr: *mut f32) -> &'a Self {
        // SAFETY: guaranteed by the caller
        unsafe { &*ptr.cast() }
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn get_mut(&mut self) -> &mut f32 {
        self.v.get_mut()
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn from_mut(v: &mut f32) -> &mut Self {
        let [] = [(); align_of::<Self>() - align_of::<f32>()];
        let [] = [(); align_of::<f32>() - align_of::<Self>()];
        // SAFETY:
        //  - the mutable reference guarantees unique ownership.
        //  - the alignment of `f32` and `Self` is the
        //    same, as verified above
        unsafe { &mut *(v as *mut f32 as *mut Self) }
    }

    #[inline(always)]
    #[allow(unused)]
    pub const fn into_inner(self) -> f32 {
        self.v.into_inner()
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn fetch_add(&self, val: f32, order: Ordering) -> f32 {
        #[cfg(target_arch = "nvptx64")]
        unsafe {
            return atomic_add_f32(self.v.get(), val, order);
        }
        #[cfg(not(any(target_arch = "nvptx64")))]
        backing_float_impl(&self.v, val, |x, y| x + y, order)
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn fetch_sub(&self, val: f32, order: Ordering) -> f32 {
        self.fetch_add(-val, order)
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn fetch_min(&self, val: f32, order: Ordering) -> f32 {
        #[cfg(target_arch = "nvptx64")]
        unsafe {
            return atomic_min_f32(self.v.get(), val, order);
        }
        #[cfg(not(any(target_arch = "nvptx64")))]
        backing_float_impl(&self.v, val, |x, y| x.min(y), order)
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn fetch_max(&self, val: f32, order: Ordering) -> f32 {
        #[cfg(target_arch = "nvptx64")]
        unsafe {
            return atomic_max_f32(self.v.get(), val, order);
        }
        #[cfg(not(any(target_arch = "nvptx64")))]
        backing_float_impl(&self.v, val, |x, y| x.max(y), order)
    }
}
