use std::marker::PhantomData;

use crate::tensors::indexing::TensorAccess;
use crate::tensors::{Dimension, Tensor};

mod indexes;
pub(crate) mod transposition;
pub mod traits;

pub use indexes::*;

// TODO: Document NoInteriorMutability as part of TensorRef (can't use it as a supertrait because then blanket impls would have to be breaking or not useful)
pub unsafe trait TensorRef<T, const D: usize> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T>;

    fn view_shape(&self) -> [(Dimension, usize); D];
}

pub unsafe trait TensorMut<T, const D: usize>: TensorRef<T, D> {
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T>;
}

pub struct TensorView<T, S, const D: usize> {
    source: S,
    _type: PhantomData<T>,
}

impl<T, S, const D: usize> TensorView<T, S, D>
where
    S: TensorRef<T, D>,
{
    pub fn from(source: S) -> TensorView<T, S, D> {
        TensorView {
            source,
            _type: PhantomData,
        }
    }

    pub fn source(self) -> S {
        self.source
    }

    pub fn source_ref(&self) -> &S {
        &self.source
    }

    pub fn source_ref_mut(&mut self) -> &mut S {
        &mut self.source
    }

    #[track_caller]
    pub fn get(
        &self,
        dimensions: [Dimension; D],
    ) -> TensorAccess<T, &S, D> {
        TensorAccess::from(&self.source, dimensions)
    }

    #[track_caller]
    pub fn get_mut(
        &mut self,
        dimensions: [Dimension; D],
    ) -> TensorAccess<T, &mut S, D> {
        TensorAccess::from(&mut self.source, dimensions)
    }

    #[track_caller]
    pub fn get_owned(self, dimensions: [Dimension; D]) -> TensorAccess<T, S, D> {
        TensorAccess::from(self.source, dimensions)
    }
}

impl<T, S, const D: usize> TensorView<T, S, D>
where
    S: TensorMut<T, D>
{
}

impl<'a, T, S, const D: usize> TensorView<T, S, D>
where
    T: Clone,
    S: TensorRef<T, D>,
{
    /**
     * Returns a new Tensor which has the same data as this tensor, but with the order of the
     * dimensions and corresponding order of data changed.
     *
     * For example, with a `[("row", x), ("column", y)]` tensor you could call
     * `transpose(["y", "x"])` which would return a new tensor where every (y,x) of its data
     * corresponds to (x,y) in the original.
     *
     * This method need not shift *all* the dimensions though, you could also swap the width
     * and height of images in a tensor with a shape of
     * `[("batch", b), ("h", h), ("w", w), ("c", c)]` via `transpose(["batch", "w", "h", "c"])`
     * which would return a new tensor where every (b,w,h,c) of its data corresponds to (b,h,w,c)
     * in the original.
     *
     * # Panics
     *
     * If the set of dimensions in the tensor does not match the set of dimensions provided. The
     * order need not match (and if the order does match, this function is just an expensive
     * clone).
     */
    #[track_caller]
    pub fn transpose(&self, dimensions: [Dimension; D]) -> Tensor<T, D> {
        crate::tensors::views::transposition::transpose(&self.source, dimensions)
    }
}
