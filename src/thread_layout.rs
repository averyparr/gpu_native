use crate::slices::UniqueId;

#[allow(unused)]
pub unsafe trait ThreadLayout {
    #[cfg(target_arch = "nvptx64")]
    fn uid(&self) -> UniqueId;

    #[cfg(not(any(target_arch = "nvptx64")))]
    fn uid(&self) -> UniqueId {
        panic!("Attempting to call ThreadLayout::uid() from host code!");
    }
}
