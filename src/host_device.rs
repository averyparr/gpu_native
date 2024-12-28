#[macro_export]
macro_rules! assert_universal {
    ($expr: expr) => {
        #[cfg(target_arch = "nvptx64")]
        crate::cu_assert!($expr);
        #[cfg(not(target_arch = "nvptx64"))]
        assert!($expr);

        unsafe { core::hint::assert_unchecked($expr)};
    };
    ($expr: expr, $($arg:tt)+) => {
        #[cfg(target_arch = "nvptx64")]
        crate::cu_assert!($expr, $($arg)+);
        #[cfg(not(target_arch = "nvptx64"))]
        assert!($expr, $($arg)+);

        unsafe { core::hint::assert_unchecked($expr)};
    };
}

#[macro_export]
macro_rules! panic_universal {
    ($($arg:tt)*) => {
        #[cfg(target_arch = "nvptx64")]
        {
            crate::cu_panic!($($arg)*);
        }

        #[cfg(not(target_arch = "nvptx64"))]
        {
            panic!($($arg)*);
        }
    };
}
