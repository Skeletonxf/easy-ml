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

    /**
     * Creates a TensorAccess which maps each dimension name to an index in the order
     * of the dimensions reported by [`view_shape()`](TensorRef::view_shape).
     *
     * Hence if you create a TensorAccess directly from a Tensor, then use `from_source_order`,
     * this uses the order the dimensions were laid out in memory with.
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * use easy_ml::tensors::indexing::TensorAccess;
     * use easy_ml::tensors::{dimension, of};
     * let tensor = Tensor::new(vec![
     *     1, 2,
     *     3, 4,
     *
     *     5, 6,
     *     7, 8
     * ], [of("x", 2), of("y", 2), of("z", 2)]);
     * let xyz = tensor.get([dimension("x"), dimension("y"), dimension("z")]);
     * let also_xyz = TensorAccess::from_source_order(&tensor);
     * ```
     */
    pub fn from_source_order(source: S) -> TensorAccess<T, S, D> {
        let mut no_op_mapping = [0; D];
        for d in 0..D {
            no_op_mapping[d] = d;
        }
        TensorAccess {
            dimension_mapping: no_op_mapping,
            source,
            _type: PhantomData,
        }
    }

    pub fn shape(&self) -> [(Dimension, usize); D] {
        let memory_shape = self.source.view_shape();
        #[allow(clippy::clone_on_copy)]
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

    fn index_is_valid(&self, indexes: [usize; D]) -> bool {
        self.try_get_reference(indexes).is_some()
    }

    pub fn index_reference_iter(&self) -> IndexOrderIterator<T, S, D> {
        IndexOrderIterator::from(self)
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

/**
 * An iterator over references to all values in a tensor.
 *
 * First the all 0 index is iterated, then each iteration increments the rightmost index.
 * If the tensor access is created with the same dimension order as the data in the tensor,
 * this will take a single step in memory on each iteration, akin to iterating through the
 * flattened data of the tensor.
 *
 * If the tensor access is created with a different dimension order, this iterator will still
 * iterate the rightmost index of the index order defined by the tensor access but the tensor
 * access will map those dimensions to the order the data in the tensor is stored, allowing
 * iteration through dimensions in a different order to how they are stored, but no longer
 * taking a single step in memory on each iteration.
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * use easy_ml::tensors::of;
 * use easy_ml::tensors::dimension as d;
 *
 * let tensor_0 = Tensor::from_scalar(1);
 * let tensor_1 = Tensor::new(vec![ 1, 2, 3, 4, 5, 6, 7 ], [of("a", 7)]);
 * let tensor_2 = Tensor::new(
 *     // two rows, three columns
 *     vec![ 1, 2, 3,
 *           4, 5, 6 ],
 *     [of("a", 2), of("b", 3)]
 * );
 * let tensor_3 = Tensor::new(
 *     // two rows each a single column, stacked on top of each other
 *     vec![ 1,
 *           2,
 *
 *           3,
 *           4 ],
 *     [of("a", 2), of("b", 1), of("c", 2)]
 * );
 * let tensor_access_0 = tensor_0.get([]);
 * let tensor_access_1 = tensor_1.get([d("a")]);
 * let tensor_access_2 = tensor_2.get([d("a"), d("b")]);
 * let tensor_access_2_rev = tensor_2.get([d("b"), d("a")]);
 * let tensor_access_3 = tensor_3.get([d("a"), d("b"), d("c")]);
 * let tensor_access_3_rev = tensor_3.get([d("c"), d("b"), d("a")]);
 * assert_eq!(
 *     tensor_access_0.index_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1]
 * );
 * assert_eq!(
 *     tensor_access_1.index_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6, 7]
 * );
 * assert_eq!(
 *     tensor_access_2.index_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6]
 * );
 * assert_eq!(
 *     tensor_access_2_rev.index_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1, 4, 2, 5, 3, 6]
 * );
 * assert_eq!(
 *     tensor_access_3.index_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4]
 * );
 * assert_eq!(
 *     tensor_access_3_rev.index_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1, 3, 2, 4]
 * );
 * ```
 */
pub struct IndexOrderIterator<'a, T, S, const D: usize> {
    tensor: &'a TensorAccess<T, S, D>,
    shape: [(Dimension, usize); D],
    indexes: [usize; D],
    finished: bool,
}

impl<'a, T, S, const D: usize> IndexOrderIterator<'a, T, S, D>
where
    S: TensorRef<T, D>,
{
    pub fn from(tensor_access: &TensorAccess<T, S, D>) -> IndexOrderIterator<T, S, D> {
        IndexOrderIterator {
            finished: !tensor_access.index_is_valid([0; D]),
            shape: tensor_access.shape(),
            tensor: tensor_access,
            indexes: [0; D],
        }
    }
}

impl<'a, T, S, const D: usize> Iterator for IndexOrderIterator<'a, T, S, D>
where
    S: TensorRef<T, D>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        index_order_iter(&mut self.finished, &mut self.indexes, &self.shape).map(|indexes| {
            self.tensor.get_reference(indexes)
        })
    }
}

// Common index order iterator logic
fn index_order_iter<const D: usize>(
    finished: &mut bool,
    indexes: &mut [usize; D],
    shape: &[(Dimension, usize); D],
) -> Option<[usize; D]> {
    if *finished {
        return None;
    }

    let value = Some(*indexes);

    if D > 0 {
        // Increment index of final dimension. In the 2D case, we iterate through a row by
        // incrementing through every column index.
        indexes[D - 1] += 1;
        for d in (1..D).rev() {
            if indexes[d] == shape[d].1 {
                // ran to end of this dimension with our index
                // In the 2D case, we finished indexing through every column in the row,
                // and it's now time to move onto the next row.
                indexes[d] = 0;
                indexes[d - 1] += 1;
            }
        }
        // Check if we ran past the final index
        if indexes[0] == shape[0].1 {
            *finished = true;
        }
    } else {
        *finished = true;
    }

    value
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
