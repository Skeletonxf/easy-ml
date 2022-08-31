/*!
 * Trait implementations for [TensorRef](TensorRef) and [TensorMut](TensorMut).
 *
 * These implementations are written here but Rust docs will display them on the
 * traits' pages.
 *
 * An owned or referenced [Tensor](Tensor) is a TensorRef, and a TensorMut if not a shared
 * reference, Therefore, you can pass a Tensor to any function which takes a TensorRef.
 *
 * Boxed TensorRef and TensorMut values also implement TensorRef and TensorMut respectively.
 */

use crate::tensors::views::{TensorMut, TensorRef, DataLayout};
use crate::tensors::Dimension;
#[allow(unused_imports)] // used in doc links
use crate::tensors::Tensor;

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

    fn data_layout(&self) -> DataLayout<D> {
        TensorRef::data_layout(*self)
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

    fn data_layout(&self) -> DataLayout<D> {
        TensorRef::data_layout(*self)
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

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorRef
// correctly as well.
/**
 * A box of a TensorRef also implements TensorRef.
 */
unsafe impl<T, S, const D: usize> TensorRef<T, D> for Box<S>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.as_ref().get_reference(indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.as_ref().view_shape()
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        self.as_ref().get_reference_unchecked(indexes)
    }

    fn data_layout(&self) -> DataLayout<D> {
        self.as_ref().data_layout()
    }
}

// # Safety
//
// The type implementing TensorMut must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorMut
// correctly as well.
/**
 * A box of a TensorMut also implements TensorMut.
 */
unsafe impl<T, S, const D: usize> TensorMut<T, D> for Box<S>
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.as_mut().get_reference_mut(indexes)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        self.as_mut().get_reference_unchecked_mut(indexes)
    }
}

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorRef
// correctly as well.
/**
 * A box of a dynamic TensorRef also implements TensorRef.
 */
unsafe impl<T, const D: usize> TensorRef<T, D> for Box<dyn TensorRef<T, D>> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.as_ref().get_reference(indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.as_ref().view_shape()
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        self.as_ref().get_reference_unchecked(indexes)
    }

    fn data_layout(&self) -> DataLayout<D> {
        self.as_ref().data_layout()
    }
}

// # Safety
//
// The type implementing TensorMut must implement TensorRef correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorRef
// correctly as well.
/**
 * A box of a dynamic TensorMut also implements TensorRef.
 */
unsafe impl<T, const D: usize> TensorRef<T, D> for Box<dyn TensorMut<T, D>> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.as_ref().get_reference(indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.as_ref().view_shape()
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        self.as_ref().get_reference_unchecked(indexes)
    }

    fn data_layout(&self) -> DataLayout<D> {
        self.as_ref().data_layout()
    }
}

// # Safety
//
// The type implementing TensorMut must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorMut
// correctly as well.
/**
 * A box of a dynamic TensorMut also implements TensorMut.
 */
unsafe impl<T, const D: usize> TensorMut<T, D> for Box<dyn TensorMut<T, D>> {
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.as_mut().get_reference_mut(indexes)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        self.as_mut().get_reference_unchecked_mut(indexes)
    }
}
