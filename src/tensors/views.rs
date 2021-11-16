use std::marker::PhantomData;

use crate::tensors::indexing::TensorAccess;
use crate::tensors::Dimension;

mod indexes;
pub mod traits;

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
    ) -> TensorAccess<T, TensorViewSourceRef<'_, T, S, D>, D> {
        TensorAccess::from(
            TensorViewSourceRef {
                source: &self.source,
                _type: PhantomData,
            },
            dimensions,
        )
    }

    #[track_caller]
    pub fn get_mut(
        &mut self,
        dimensions: [Dimension; D],
    ) -> TensorAccess<T, TensorViewSourceMut<'_, T, S, D>, D> {
        TensorAccess::from(
            TensorViewSourceMut {
                source: &mut self.source,
                _type: PhantomData,
            },
            dimensions,
        )
    }

    #[track_caller]
    pub fn get_owned(self, dimensions: [Dimension; D]) -> TensorAccess<T, S, D> {
        TensorAccess::from(self.source, dimensions)
    }
}

impl<T, S, const D: usize> TensorView<T, S, D> where S: TensorMut<T, D> {}

pub struct TensorViewSourceRef<'s, T, S, const D: usize> {
    source: &'s S,
    _type: PhantomData<T>,
}

pub struct TensorViewSourceMut<'s, T, S, const D: usize> {
    source: &'s mut S,
    _type: PhantomData<T>,
}

unsafe impl<'a, T, S, const D: usize> TensorRef<T, D> for TensorViewSourceRef<'a, T, S, D>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.source.get_reference(indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.source.view_shape()
    }
}

unsafe impl<'a, T, S, const D: usize> TensorRef<T, D> for TensorViewSourceMut<'a, T, S, D>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.source.get_reference(indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.source.view_shape()
    }
}

unsafe impl<'a, T, S, const D: usize> TensorMut<T, D> for TensorViewSourceMut<'a, T, S, D>
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.source.get_reference_mut(indexes)
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
