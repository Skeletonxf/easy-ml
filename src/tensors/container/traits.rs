/*!
 * Trait implementations for [TensorlikeRef](TensorlikeRef) and [TensorlikeMut](TensorlikeMut).
 *
 * These implementations are written here but Rust docs will display them on the
 * traits' pages.
 */

use crate::tensors::container::{TensorlikeRef, TensorlikeMut};
use crate::tensors::views::{DataLayout, TensorMut, TensorRef};
use crate::tensors::Dimension;

/**
 * Any type that implements TensorRef or TensorMut can be wrapped by this type to implement
 * TensorlikeRef and TensorlikeMut respectively.
 */
struct Container<C> {
    pub container: C,
}

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorlikeRef
// correctly as well.
/**
 * All wrapped TensorRef types implement TensorlikeRef where the value returned by indexing is &T
 */
unsafe impl<'a, T, C, const D: usize> TensorlikeRef<'a, T, D, &'a T> for Container<C>
where
    C: TensorRef<T, D>,
{
    fn get_value(&'a self, indexes: [usize; D]) -> Option<&'a T> {
        TensorRef::get_reference(&self.container, indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        TensorRef::view_shape(&self.container)
    }

    unsafe fn get_value_unchecked(&'a self, indexes: [usize; D]) -> &'a T {
        TensorRef::get_reference_unchecked(&self.container, indexes)
    }

    fn data_layout(&self) -> DataLayout<D> {
        TensorRef::data_layout(&self.container)
    }
}

// # Safety
//
// The type implementing TensorMut must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorlikeMut
// correctly as well.
/**
 * All wrapped TensorMut types implement TensorlikeMut where the values returned by indexing are
 * &T and &mut T.
 */
unsafe impl<'a, T, C, const D: usize> TensorlikeMut<'a, T, D, &'a T, &'a mut T> for Container<C>
where
    C: TensorMut<T, D>,
{
    fn get_value_mut(&'a mut self, indexes: [usize; D]) -> Option<&'a mut T> {
        TensorMut::get_reference_mut(&mut self.container, indexes)
    }

    unsafe fn get_value_unchecked_mut(&'a mut self, indexes: [usize; D]) -> &'a mut T {
        TensorMut::get_reference_unchecked_mut(&mut self.container, indexes)
    }
}

// # Safety
//
// The type implementing TensorlikeRef must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorlikeRef
// correctly as well.
/**
 * If some type implements TensorlikeRef, then a reference to it implements TensorlikeRef as well
 */
unsafe impl<'a, 'source, T, S, const D: usize, Ref> TensorlikeRef<'a, T, D, Ref> for &'source S
where
    S: TensorlikeRef<'a, T, D, Ref>,
{
    fn get_value(&'a self, indexes: [usize; D]) -> Option<Ref> {
        TensorlikeRef::get_value(*self, indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        TensorlikeRef::view_shape(*self)
    }

    unsafe fn get_value_unchecked(&'a self, indexes: [usize; D]) -> Ref {
        TensorlikeRef::get_value_unchecked(*self, indexes)
    }

    fn data_layout(&self) -> DataLayout<D> {
        TensorlikeRef::data_layout(*self)
    }
}

// # Safety
//
// The type implementing TensorlikeRef must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorlikeRef
// correctly as well.
/**
 * If some type implements TensorlikeRef, then an exclusive reference to it implements
 * TensorlikeRef as well
 */
unsafe impl<'a, 'source, T, S, const D: usize, Ref> TensorlikeRef<'a, T, D, Ref> for &'source mut S
where
    S: TensorlikeRef<'a, T, D, Ref>,
{
    fn get_value(&'a self, indexes: [usize; D]) -> Option<Ref> {
        TensorlikeRef::get_value(*self, indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        TensorlikeRef::view_shape(*self)
    }

    unsafe fn get_value_unchecked(&'a self, indexes: [usize; D]) -> Ref {
        TensorlikeRef::get_value_unchecked(*self, indexes)
    }

    fn data_layout(&self) -> DataLayout<D> {
        TensorlikeRef::data_layout(*self)
    }
}

// # Safety
//
// The type implementing TensorlikeMut must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorlikeMut
// correctly as well.
/**
 * If some type implements TensorlikeMut, then an exclusive reference to it implements
 * TensorlikeMut as well
 */
unsafe impl<'a, 'source, T, S, const D: usize, Ref, Mut> TensorlikeMut<'a, T, D, Ref, Mut> for &'source mut S
where
    S: TensorlikeMut<'a, T, D, Ref, Mut>,
{
    fn get_value_mut(&'a mut self, indexes: [usize; D]) -> Option<Mut> {
        TensorlikeMut::get_value_mut(*self, indexes)
    }

    unsafe fn get_value_unchecked_mut(&'a mut self, indexes: [usize; D]) -> Mut {
        TensorlikeMut::get_value_unchecked_mut(*self, indexes)
    }
}

// TODO: Box blanket impls too
