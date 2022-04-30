use crate::tensors::views::{TensorMut, TensorRef, TensorAccess};
use crate::tensors::Dimension;

/**
 * A TensorRef/TensorMut type providing access to the data in a Tensor with a particular order
 * of dimension indexing.
 *
 * This is a thin wrapper around TensorAccess. If you just need to index a tensor, you can
 * use the TensorAccess directly instead of constructing this.
 *
 * See the [indexing module level documentation](crate::tensors::indexing) for more information.
 */
#[derive(Clone, Debug)]
pub struct TensorOrder<T, S, const D: usize> {
    source: TensorAccess<T, S, D>,
}

impl<T, S, const D: usize> TensorOrder<T, S, D>
where
    S: TensorRef<T, D>,
{
    /**
     * Creates a TensorOrder from a [TensorAccess](TensorAccess), exposing the dimension order
     * of the TensorAccess as a TensorRef/TensorMut implementing type, so it can be used in
     * a [TensorView](crate::tensors::views::TensorView).
     */
    pub fn from(tensor_access: TensorAccess<T, S, D>) -> TensorOrder<T, S, D> {
        TensorOrder {
            source: tensor_access,
        }
    }

    /**
     * Consumes the TensorOrder, yielding the source it was created from.
     */
    pub fn source(self) -> TensorAccess<T, S, D> {
        self.source
    }

    // Does it make sense to have source_ref and source_ref_mut methods here?
    // Could there be any self integrity problems from giving those out?
}

// # Safety
//
// The type implementing TensorRef inside the TensorAccess must implement it correctly, so by
// delegating to it without changing anything other than the order we index it, we implement
// TensorRef correctly as well.
/**
 * A TensorOrder implements TensorRef, with the dimension order and indexing matching that of the
 * TensorAccess shape.
 */
unsafe impl<T, S, const D: usize> TensorRef<T, D> for TensorOrder<T, S, D>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.source.try_get_reference(indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.source.shape()
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        self.source.get_reference_unchecked(indexes)
    }
}

// # Safety
//
// The type implementing TensorMut inside the TensorAccess must implement it correctly, so by
// delegating to it without changing anything other than the order we index it, we implement
// TensorMut correctly as well.
/**
 * A TensorOrder implements TensorMut, with the dimension order and indexing matching that of the
 * TensorAccess shape.
 */
unsafe impl<T, S, const D: usize> TensorMut<T, D> for TensorOrder<T, S, D>
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.source.try_get_reference_mut(indexes)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        self.source.get_reference_unchecked_mut(indexes)
    }
}
