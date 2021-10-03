use std::marker::PhantomData;

use crate::tensors::Dimension;

pub mod traits;
mod indexes;

pub use indexes::*;

// also need to document that interior mutability is completely banned since batch reference
// reading relies on view_shape being fixed for entire batch of reads
// need to document that you will typically want to implement this for &T and &mut T rather than
// just T.
// need to then abstract over this as much as possible to have a get_references(&self) method on TensorView
// since ergonomics of taking by value by accident are pretty bad
pub unsafe trait TensorRef<T, const D: usize> {
    type Accessor: TensorRefAccess<T, D>;

    fn get_references(self, dimensions: [Dimension; D]) -> Option<Self::Accessor>;

    fn view_shape(&self) -> [(Dimension, usize); D];

    // TODO: Need to express dimensionality access, reductions, elementwise operations in a vectorised way
    // perhaps can now do this on TensorView since we have vectorised reads
}

pub unsafe trait TensorRefAccess<T, const D: usize> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T>;
}

pub unsafe trait TensorMut<T, const D: usize>: TensorRef<T, D> {
    fn get_references_mut(self, dimensions: [Dimension; D]) -> Option<Self::Accessor>;
}

pub unsafe trait TensorMutAccess<T, const D: usize> {
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T>;
}

pub struct TensorView<T, S, const D: usize> {
    source: S,
    _type: PhantomData<T>,
}

/**
 * Access to the data in a Tensor with a particular type of dimension indexing.
 */
pub struct TensorAccess<T, S, const D: usize> {
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
}

impl <T, S, const D: usize> TensorView<T, S, D>
where
    for<'a> &'a S: TensorRef<T, D>
{
    #[track_caller]
    pub fn get(&self, dimensions: [Dimension; D]) -> TensorAccess<T, <&S as TensorRef<T, D>>::Accessor, D> {
        match self.source.get_references(dimensions) {
            Some(access) => TensorAccess::from(access),
            None => panic!(
                "Unable to index with {:?}, TensorView dimensions are {:?}.",
                dimensions, self.shape()
            ),
        }
    }

    pub fn shape(&self) -> [(Dimension, usize); D] {
        (&self.source).view_shape()
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

impl <T, S, const D: usize> TensorView<T, S, D>
where
    for<'a> &'a mut S: TensorMut<T, D>
{
    #[track_caller]
    pub fn get_mut(&mut self, dimensions: [Dimension; D]) -> TensorAccess<T, <&mut S as TensorRef<T, D>>::Accessor, D> {
        let shape = (&mut self.source).view_shape();
        match self.source.get_references_mut(dimensions) {
            Some(access) => TensorAccess::from(access),
            None => panic!(
                "Unable to index with {:?}, TensorView dimensions are {:?}.",
                dimensions, shape
            ),
        }
    }
}

impl <T, S, const D: usize> TensorView<T, S, D>
where
    S: TensorMut<T, D>
{
    #[track_caller]
    pub fn get_owned(self, dimensions: [Dimension; D]) -> TensorAccess<T, <S as TensorRef<T, D>>::Accessor, D> {
        let shape = self.source.view_shape();
        match self.source.get_references_mut(dimensions) {
            Some(access) => TensorAccess::from(access),
            None => panic!(
                "Unable to index with {:?}, TensorView dimensions are {:?}.",
                dimensions, shape
            ),
        }
    }
}

impl <T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorRefAccess<T, D>
{
    pub fn from(source: S) -> TensorAccess<T, S, D> {
        TensorAccess {
            source,
            _type: PhantomData,
        }
    }

    pub fn try_get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.source.get_reference(indexes)
    }

    // FIXME: Need to allow querying view_shape from Accessors
    #[track_caller]
    pub fn get_reference(&self, indexes: [usize; D]) -> &T {
        match self.source.get_reference(indexes) {
            Some(reference) => reference,
            None => panic!(
                "Unable to index with {:?}",
                indexes,
            ),
        }
    }
}

impl <T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorRefAccess<T, D>,
    T: Clone,
{
    // FIXME: Need to allow querying view_shape from Accessors
    #[track_caller]
    pub fn get(&self, indexes: [usize; D]) -> T {
        match self.source.get_reference(indexes) {
            Some(reference) => reference.clone(),
            None => panic!(
                "Unable to index with {:?}",
                indexes,
            ),
        }
    }
}

impl <T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorMutAccess<T, D>
{
    pub fn try_get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.source.get_reference_mut(indexes)
    }

    // FIXME: Need to allow querying view_shape from Accessors
    #[track_caller]
    pub fn get_reference_mut(&mut self, indexes: [usize; D]) -> &mut T {
        match self.source.get_reference_mut(indexes) {
            Some(reference) => reference,
            None => panic!(
                "Unable to index with {:?}",
                indexes,
            ),
        }
    }
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
