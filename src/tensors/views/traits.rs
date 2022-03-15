use crate::tensors::views::{TensorMut, TensorRef};
use crate::tensors::Dimension;

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorRef
// correctly as well.
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

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        TensorRef::get_reference_unchecked(*self, indexes)
    }
}

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorRef
// correctly as well.
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

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        TensorRef::get_reference_unchecked(*self, indexes)
    }
}

// # Safety
//
// The type implementing TensorMut must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorMut
// correctly as well.
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

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        TensorMut::get_reference_unchecked_mut(*self, indexes)
    }
}

// TODO: Boxed values
