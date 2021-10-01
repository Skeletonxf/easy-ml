use std::marker::PhantomData;

use crate::tensors::Dimension;

pub mod traits;
mod indexes;

pub use indexes::*;

pub unsafe trait TensorRef<T, const D: usize> {
    type Accessor: TensorRefAccess<T, D>;

    // TODO: This really should be a result so we can have proper error handling
    fn get_references(self, dimensions: [Dimension; D]) -> Option<Self::Accessor>;

    fn view_shape(&self) -> [(Dimension, usize); D];

    // TODO: Need to express dimensionality access, reductions, elementwise operations in a vectorised way
    // perhaps can now do this on TensorView since we have vectorised reads
}

pub unsafe trait TensorRefAccess<T, const D: usize> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T>;
}

// pub unsafe trait TensorMut<T, const D: usize>: TensorRef<T, D> {
//     fn try_get_reference_mut(&mut self, dimensions: [(Dimension, usize); D]) -> Option<&mut T>;
// }

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

    pub fn shape(&self) -> [(Dimension, usize); D] {
        self.source.view_shape()
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
    //
    // pub fn try_get_reference(&self, dimensions: [(Dimension, usize); D]) -> Option<&T> {
    //     self.source.try_get_reference(dimensions)
    // }
    //
    // #[track_caller]
    // pub fn get_reference(&self, dimensions: [(Dimension, usize); D]) -> &T {
    //     match self.source.try_get_reference(dimensions) {
    //         Some(reference) => reference,
    //         None => panic!(
    //             "Unable to index with {:?}, TensorView dimensions are {:?}.",
    //             dimensions, self.shape()
    //         )
    //     }
    // }
}
//
// impl <T, S, const D: usize> TensorView<T, S, D>
// where
//     S: TensorRef<T, D>,
//     T: Clone,
// {
//     #[track_caller]
//     pub fn get(&self, dimensions: [(Dimension, usize); D]) -> T {
//         self.get_reference(dimensions).clone()
//     }
// }
//
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
