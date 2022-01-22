use crate::tensors::views::{TensorRef, TensorMut};
use crate::tensors::Dimension;

/**
 * If some type implements TensorRef, then a reference to it implements TensorRef as well
 */
unsafe impl<'source, T, S, const D: usize> TensorRef<T, D> for &'source S
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        TensorRef::get_reference(*self, indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        TensorRef::view_shape(*self)
    }
}

/**
 * If some type implements TensorRef, then an exclusive reference to it implements TensorRef
 * as well
 */
unsafe impl<'source, T, S, const D: usize> TensorRef<T, D> for &'source mut S
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        TensorRef::get_reference(*self, indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        TensorRef::view_shape(*self)
    }
}

/**
 * If some type implements TensorMut, then an exclusive reference to it implements TensorMut
 * as well
 */
unsafe impl<'source, T, S, const D: usize> TensorMut<T, D> for &'source mut S
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        TensorMut::get_reference_mut(*self, indexes)
    }
}

// TODO: Boxed values
