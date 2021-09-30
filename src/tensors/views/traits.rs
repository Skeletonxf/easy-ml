use crate::tensors::{Tensor, Dimension};
use crate::tensors::views::TensorRef;
//use crate::matrices::views::NoInteriorMutability;

unsafe impl <'source, T, const D: usize> TensorRef<T, D> for &'source Tensor<T, D> {
    fn try_get_reference(&self, dimensions: [(Dimension, usize); D]) -> Option<&T> {
        Tensor::_try_get_reference(self, dimensions)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        Tensor::shape(self)
    }
}

unsafe impl <'source, T, const D: usize> TensorRef<T, D> for &'source mut Tensor<T, D> {
    fn try_get_reference(&self, dimensions: [(Dimension, usize); D]) -> Option<&T> {
        Tensor::_try_get_reference(self, dimensions)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        Tensor::shape(self)
    }
}

unsafe impl <T, const D: usize> TensorRef<T, D> for Tensor<T, D> {
    fn try_get_reference(&self, dimensions: [(Dimension, usize); D]) -> Option<&T> {
        Tensor::_try_get_reference(self, dimensions)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        Tensor::shape(self)
    }
}
