
use crate::matrices::views::NoInteriorMutability;
use crate::tensors::Tensor;

// # Safety
//
// We promise to never implement interior mutability for Tensor.
/**
 * A shared reference to a Tensor implements NoInteriorMutability.
 */
unsafe impl<'source, T, const D: usize> NoInteriorMutability for &'source Tensor<T, D> {}

// # Safety
//
// We promise to never implement interior mutability for Tensor.
/**
 * An exclusive reference to a Tensor implements NoInteriorMutability.
 */
unsafe impl<'source, T, const D: usize> NoInteriorMutability for &'source mut Tensor<T, D> {}

// # Safety
//
// We promise to never implement interior mutability for Tensor.
/**
 * An owned Tensor implements NoInteriorMutability.
 */
unsafe impl<T, const D: usize> NoInteriorMutability for Tensor<T, D> {}

// TODO: Boxed values
