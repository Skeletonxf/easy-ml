use crate::tensors::Dimension;
use crate::tensors::views::{DataLayout, TensorRef};
use std::marker::PhantomData;

/**
 * A combination of a mapping function and a tensor.
 *
 * The provided function lazily transforms the data in the tensor for the TensorRef implementation.
 */
// TODO: Is there any way this can be extened to work for TensorMut too without requiring user
// to provide two nearly identical tranformation functions that differ only on & vs &mut inputs
// and outputs?
#[derive(Clone, Debug)]
pub(crate) struct TensorMap<T, U, S, F, const D: usize> {
    source: S,
    f: F,
    _from: PhantomData<T>,
    _to: PhantomData<U>,
}

impl<T, U, S, F, const D: usize> TensorMap<T, U, S, F, D>
where
    S: TensorRef<T, D>,
    F: Fn(&T) -> &U,
{
    /**
     * Creates a TensorMap from a source and a function to lazily transform the data with.
     */
    #[track_caller]
    pub fn from(source: S, f: F) -> TensorMap<T, U, S, F, D> {
        TensorMap {
            source,
            f,
            _from: PhantomData,
            _to: PhantomData,
        }
    }

    /**
     * Consumes the TensorMap, yielding the source it was created from.
     */
    #[allow(dead_code)]
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Gives a reference to the TensorMap's source (in which the data is not transformed).
     */
    #[allow(dead_code)]
    pub fn source_ref(&self) -> &S {
        &self.source
    }

    /**
     * Gives a mutable reference to the TensorMap's source (in which the data is not transformed).
     */
    #[allow(dead_code)]
    pub fn source_ref_mut(&mut self) -> &mut S {
        &mut self.source
    }
}

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorRef
// correctly as well.
/**
 * A TensorRename implements TensorRef, with the dimension names the TensorRename was created
 * with overriding the dimension names in the original source.
 */
unsafe impl<T, U, S, F, const D: usize> TensorRef<U, D> for TensorMap<T, U, S, F, D>
where
    S: TensorRef<T, D>,
    F: Fn(&T) -> &U,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&U> {
        Some((self.f)(self.source.get_reference(indexes)?))
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.source.view_shape()
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &U {
        unsafe { (self.f)(self.source.get_reference_unchecked(indexes)) }
    }

    fn data_layout(&self) -> DataLayout<D> {
        self.source.data_layout()
    }
}
