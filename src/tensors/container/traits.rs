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
 *
 * TODO: Consider if TensorlikeRef should just be a supertype of TensorRef from introduction
 * and aim for releasing version 2.0
 *
 * Argument towards no is TensorlikeRef took me hours just to define correctly so might add
 * too much cognitive burden to the API to be literally everywhere and mandatory to understand
 * Tensors in general.
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
unsafe impl<T, C, const D: usize> TensorlikeRef<T, D> for Container<C>
where
    C: TensorRef<T, D>,
{
    type Ref<'a> = &'a T where Self: 'a, T: 'a;

    fn get_value<'a>(&'a self, indexes: [usize; D]) -> Option<&'a T> {
        TensorRef::get_reference(&self.container, indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        TensorRef::view_shape(&self.container)
    }

    unsafe fn get_value_unchecked<'a>(&'a self, indexes: [usize; D]) -> &'a T {
        TensorRef::get_reference_unchecked(&self.container, indexes)
    }

    fn data_layout(&self) -> DataLayout<D> {
        TensorRef::data_layout(&self.container)
    }
}

// Sanity checking we'll be able to do impls for RecordContainer later
unsafe impl<T> TensorlikeRef<T, 0> for (T, usize) {
    type Ref<'a> = (&'a T, usize) where Self: 'a, T: 'a;

    fn get_value<'a>(&'a self, _indexes: [usize; 0]) -> Option<(&'a T, usize)> {
        Some((&self.0, self.1))
    }

    fn view_shape(&self) -> [(Dimension, usize); 0] {
        []
    }

    unsafe fn get_value_unchecked<'a>(&'a self, _indexes: [usize; 0]) -> (&'a T, usize) {
        (&self.0, self.1)
    }

    fn data_layout(&self) -> DataLayout<0> {
        DataLayout::Other
    }
}

// Sanity checking we'll be able to do impls for RecordContainer later
unsafe impl<T> TensorlikeRef<T, 0> for (T, usize, usize) {
    type Ref<'a> = (usize, usize) where Self: 'a, T: 'a;

    fn get_value<'a>(&'a self, _indexes: [usize; 0]) -> Option<(usize, usize)> {
        Some((self.1, self.2))
    }

    fn view_shape(&self) -> [(Dimension, usize); 0] {
        []
    }

    unsafe fn get_value_unchecked<'a>(&'a self, _indexes: [usize; 0]) -> (usize, usize) {
        (self.1, self.2)
    }

    fn data_layout(&self) -> DataLayout<0> {
        DataLayout::Other
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
unsafe impl<T, C, const D: usize> TensorlikeMut<T, D> for Container<C>
where
    C: TensorMut<T, D>,
{
    type Mut<'a> = &'a mut T where Self: 'a, T: 'a;

    fn get_value_mut<'a>(&'a mut self, indexes: [usize; D]) -> Option<&'a mut T> {
        TensorMut::get_reference_mut(&mut self.container, indexes)
    }

    unsafe fn get_value_unchecked_mut<'a>(&'a mut self, indexes: [usize; D]) -> &'a mut T {
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
unsafe impl<'source, T, S, const D: usize> TensorlikeRef<T, D> for &'source S
where
    S: TensorlikeRef<T, D>,
{
    type Ref<'a> = S::Ref<'a> where Self: 'a, T: 'a;

    fn get_value<'a>(&'a self, indexes: [usize; D]) -> Option<S::Ref<'a>> {
        TensorlikeRef::get_value(*self, indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        TensorlikeRef::view_shape(*self)
    }

    unsafe fn get_value_unchecked<'a>(&'a self, indexes: [usize; D]) -> S::Ref<'a> {
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
unsafe impl<'source, T, S, const D: usize> TensorlikeRef<T, D> for &'source mut S
where
    S: TensorlikeRef<T, D>,
{
    type Ref<'a> = S::Ref<'a> where Self: 'a, T: 'a;

    fn get_value<'a>(&'a self, indexes: [usize; D]) -> Option<S::Ref<'a>> {
        TensorlikeRef::get_value(*self, indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        TensorlikeRef::view_shape(*self)
    }

    unsafe fn get_value_unchecked<'a>(&'a self, indexes: [usize; D]) -> S::Ref<'a> {
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
unsafe impl<'source, T, S, const D: usize> TensorlikeMut<T, D> for &'source mut S
where
    S: TensorlikeMut<T, D>,
{
    type Mut<'a> = S::Mut<'a> where Self: 'a, T: 'a;

    fn get_value_mut<'a>(&'a mut self, indexes: [usize; D]) -> Option<S::Mut<'a>> {
        TensorlikeMut::get_value_mut(*self, indexes)
    }

    unsafe fn get_value_unchecked_mut<'a>(&'a mut self, indexes: [usize; D]) -> S::Mut<'a> {
        TensorlikeMut::get_value_unchecked_mut(*self, indexes)
    }
}

// TODO: Box blanket impls too
