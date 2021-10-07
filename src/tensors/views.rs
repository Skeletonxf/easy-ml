use std::marker::PhantomData;

use crate::tensors::Dimension;
use crate::tensors::indexing::TensorAccess;

pub mod traits;
mod indexes;

pub use indexes::*;

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

impl <T, S, const D: usize> TensorView<T, S, D>
where
    S: TensorRef<T, D>
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

    // TODO: Should create a TensorAccess from a source type which doesn't force a move here
    #[track_caller]
    pub fn get(self, dimensions: [Dimension; D]) -> TensorAccess<T, S, D> {
        TensorAccess::from(self.source, dimensions)
    }
}

impl <T, S, const D: usize> TensorView<T, S, D>
where
    S: TensorMut<T, D>
{
    // TODO: Should create a TensorAccess from a source type which doesn't force a move here
    #[track_caller]
    pub fn get_mut(self, dimensions: [Dimension; D]) -> TensorAccess<T, S, D> {
        TensorAccess::from(self.source, dimensions)
    }
}



// impl <T, S> TensorView<T, S, 2>
// where
//     S: TensorRef<T, 2>,
// {
//     pub fn select(self, index: [(Dimension, usize); 1]) -> TensorView<T, TensorIndex<T, S, 2, 1>, 1> {
//         TensorView::from(TensorIndex::from(self.source, index))
//     }
// }
//
// impl <T, S> TensorView<T, S, 3>
// where
//     S: TensorRef<T, 3>,
// {
//     pub fn select(self, index: [(Dimension, usize); 1]) -> TensorView<T, TensorIndex<T, S, 3, 1>, 2> {
//         TensorView::from(TensorIndex::from(self.source, index))
//     }
// }
