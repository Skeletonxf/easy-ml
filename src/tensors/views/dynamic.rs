use crate::tensors::views::{Dimension, TensorRef, TensorView};
use std::marker::PhantomData;

/**
 * A wrapper around a tensor with the const generic parameter type erased.
 *
 * This is rarely useful as the API surface becomes much more difficult to use,
 * but when you need to store tensors of different ranks in the same structure
 * this can be necessary.
 */
pub struct DynamicTensor<T> {
    rank: usize,
    _type: PhantomData<T>,
}

impl<T> DynamicTensor<T> {
    /**
     * Creates the DynamicTensor from any source type that can be converted to a
     * TensorView, which includes Tensor, &Tensor, &mut Tensor and references
     * to a TensorView.
     */
    pub fn from<S, I, const D: usize>(tensor: I) -> Self
    where
        I: Into<TensorView<T, S, D>>,
        S: TensorRef<T, D>,
    {
        let view = tensor.into();
        DynamicTensor {
            rank: view.shape().len(),
            _type: PhantomData,
        }
        // TODO: We're going to need to use some unsafe here to create
        // our own VTable and pointers
        // Should be similar to: https://lucumr.pocoo.org/2024/5/16/macro-vtable-magic/
        // except we're not using Arc
    }

    /**
     * Returns the number of dimensions, also known as rank, of this tensor.
     */
    pub fn dimensions_len(&self) -> usize {
        self.rank
    }

    /**
     * Gets the nth dimension name and length of this tensor, if there is one.
     *
     * Dimensions are indexed from 0 inclusive to the number of dimensions exclusive.
     */
    fn get_dimension(&self, index: usize) -> Option<(Dimension, usize)> {
        unimplemented!()
    }

    /**
     * Gets a this tensor's shape if the const parameter matches the number of dimensions
     * in this tensor. Otherwise returns None.
     */
    fn view_shape_dynamic_const<const D: usize>(&self) -> Option<[(Dimension, usize); D]> {
        unimplemented!()
    }

    /**
     * Gets a reference to the value at the index if the index is in range and of a matching
     * length to the number of dimensions in this tensor. Otherwise returns None.
     */
    fn get_dynamic_reference(&self, indexes: &[usize]) -> Option<&T> {
        unimplemented!()
    }

    /**
     * Gets a reference to the value at the index if the index is in range and of a matching
     * length to the number of dimensions in this tensor. Otherwise returns None.
     */
    fn get_dynamic_const_reference<const D: usize>(&self, indexes: [usize; D]) -> Option<&T> {
        unimplemented!()
    }
}
