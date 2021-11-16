use crate::tensors::views::{TensorMut, TensorRef};
use crate::tensors::Dimension;

use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

/**
 * Access to the data in a Tensor with a particular order of dimension indexing.
 */
#[derive(Clone, Debug)]
pub struct TensorAccess<T, S, const D: usize> {
    source: S,
    dimension_mapping: [usize; D],
    _type: PhantomData<T>,
}

impl<T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorRef<T, D>,
{
    #[track_caller]
    pub fn from(source: S, dimensions: [Dimension; D]) -> TensorAccess<T, S, D> {
        match TensorAccess::try_from(source, dimensions) {
            Err(error) => panic!("{}", error),
            Ok(success) => success,
        }
    }

    pub fn try_from(
        source: S,
        dimensions: [Dimension; D],
    ) -> Result<TensorAccess<T, S, D>, InvalidDimensionsError<D>> {
        Ok(TensorAccess {
            dimension_mapping: dimension_mapping(&source.view_shape(), &dimensions).ok_or_else(
                || InvalidDimensionsError {
                    actual: source.view_shape(),
                    requested: dimensions,
                },
            )?,
            source,
            _type: PhantomData,
        })
    }

    pub fn shape(&self) -> [(Dimension, usize); D] {
        let memory_shape = self.source.view_shape();
        let mut shape = memory_shape.clone();
        for d in 0..D {
            shape[d] = memory_shape[self.dimension_mapping[d]];
        }
        shape
    }
}

/**
 * An error indicating failure to create a TensorAccess because the requested dimension order
 * does not match the set of dimensions in the source data.
 */
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct InvalidDimensionsError<const D: usize> {
    actual: [(Dimension, usize); D],
    requested: [Dimension; D],
}

impl<const D: usize> Error for InvalidDimensionsError<D> {}

impl<const D: usize> fmt::Display for InvalidDimensionsError<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Requested dimension order: {:?} does not match the set of dimensions in the source: {:?}",
            &self.actual, &self.requested
        )
    }
}

#[test]
fn test_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<InvalidDimensionsError<3>>();
}

#[test]
fn test_send() {
    fn assert_send<T: Send>() {}
    assert_send::<InvalidDimensionsError<3>>();
}

// Computes a mapping from a set of dimensions in memory order to a matching set of
// dimensions in an arbitary order.
// Returns a list where each dimension in the memory order is mapped to the requested order,
// such that if the memory order is x,y,z but the requested order is z,y,x then the mapping
// is [2,1,0] as this maps the first dimension x to the third dimension x, the second dimension y
// to the second dimension y, and the third dimension z to to the first dimension z.
fn dimension_mapping<const D: usize>(
    memory: &[(Dimension, usize); D],
    requested: &[Dimension; D],
) -> Option<[usize; D]> {
    let mut mapping = [0; D];
    for d in 0..D {
        let dimension = memory[d].0;
        // happy path, requested dimension is in the same order as in memory
        let order = if requested[d] == dimension {
            d
        } else {
            // If dimensions are in a different order, find the requested dimension with the
            // matching dimension name.
            // Since both lists are the same length and we know our memory order won't contain
            // duplicates this also ensures the two lists have exactly the same set of names
            // as otherwise one of these `find`s will fail.
            let (n, _) = requested
                .iter()
                .enumerate()
                .find(|(_, d)| **d == dimension)?;
            n
        };
        mapping[d] = order;
    }
    Some(mapping)
}

// Reorders some indexes according to the dimension_mapping to yield indexes in memory order.
#[inline]
fn map_dimensions<const D: usize>(
    dimension_mapping: &[usize; D],
    indexes: &[usize; D],
) -> [usize; D] {
    let mut lookup = [0; D];
    for d in 0..D {
        lookup[d] = indexes[dimension_mapping[d]];
    }
    lookup
}

impl<T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorRef<T, D>,
{
    pub fn try_get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.source
            .get_reference(map_dimensions(&self.dimension_mapping, &indexes))
    }

    #[track_caller]
    pub fn get_reference(&self, indexes: [usize; D]) -> &T {
        match self.try_get_reference(indexes) {
            Some(reference) => reference,
            None => panic!(
                "Unable to index with {:?}, Tensor dimensions are {:?}.",
                indexes,
                self.shape()
            ),
        }
    }
}

impl<T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorRef<T, D>,
    T: Clone,
{
    #[track_caller]
    pub fn get(&self, indexes: [usize; D]) -> T {
        match self.try_get_reference(indexes) {
            Some(reference) => reference.clone(),
            None => panic!(
                "Unable to index with {:?}, Tensor dimensions are {:?}.",
                indexes,
                self.shape()
            ),
        }
    }
}

impl<T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorMut<T, D>,
{
    pub fn try_get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.source
            .get_reference_mut(map_dimensions(&self.dimension_mapping, &indexes))
    }

    #[track_caller]
    pub fn get_reference_mut(&mut self, indexes: [usize; D]) -> &mut T {
        match self.try_get_reference_mut(indexes) {
            Some(reference) => reference,
            // can't provide a better error because the borrow checker insists that returning
            // a reference in the Some branch means our mutable borrow prevents us calling
            // self.shape() and a bad error is better than cloning self.shape() on every call
            None => panic!("Unable to index with {:?}", indexes),
        }
    }
}

#[test]
fn test_dimension_mapping() {
    use crate::tensors::{dimension, of};
    let mapping = dimension_mapping(
        &[of("x", 0), of("y", 0), of("z", 0)],
        &[dimension("x"), dimension("y"), dimension("z")],
    );
    assert_eq!([0, 1, 2], mapping.unwrap());
    let mapping = dimension_mapping(
        &[of("x", 0), of("y", 0), of("z", 0)],
        &[dimension("z"), dimension("y"), dimension("x")],
    );
    assert_eq!([2, 1, 0], mapping.unwrap());
}
