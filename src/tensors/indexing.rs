/*!
 * # Indexing
 *
 * Many libraries represent tensors as n dimensional arrays, however there is often some semantic
 * meaning to each dimension. You may have a batch of 2000 images, each 100 pixels wide and high,
 * with each pixel representing 3 numbers for rgb values. This can be represented as a
 * 2000 x 100 x 100 x 3 tensor, but a 4 dimensional array does not track the semantic meaning
 * of each dimension and associated index.
 *
 * 6 months later you could come back to the code and forget which order the dimensions were
 * created in, at best getting the indexes out of bounds and causing a crash in your application,
 * and at worst silently reading the wrong data without realising. *Was it width then height or
 * height then width?*...
 *
 * Easy ML moves the n dimensional array to an implementation detail, and most of its APIs work
 * on the names of each dimension in a tensor instead of just the order. Instead of a
 * 2000 x 100 x 100 x 3 tensor in which the last element is at [1999, 99, 99, 2], Easy ML tracks
 * the names of the dimensions, so you have a
 * `[("batch", 2000), ("width", 100), ("height", 100), ("rgb", 3)]` shaped tensor.
 *
 * This can't stop you from getting the math wrong, but confusion over which dimension
 * means what is reduced, tensors carry around their pairs of dimension name and length
 * so adding a `[("batch", 2000), ("width", 100), ("height", 100), ("rgb", 3)]` shaped tensor
 * to a `[("batch", 2000), ("height", 100), ("width", 100), ("rgb", 3)]` will fail unless you
 * transpose one first, and you could access an element as
 * [batch: 1999, width: 0, height: 99, rgb: 3] or [batch: 1999, height: 99, width: 0, rgb: 3]
 * and read the same data, because you index into dimensions based on their name, not the order
 * they are stored in memory.
 *
 * Even with a name for each dimension, at some point you still need to say what order you want
 * to index each dimension with, and this is where `[TensorAccess]`(TensorAccess) comes in. It
 * creates a mapping from the dimension name order you want to access elements with to the order
 * the dimensions are stored as.
 */

use crate::tensors::views::{TensorMut, TensorRef};
use crate::tensors::{Dimension, Tensor};

use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

pub use crate::matrices::iterators::WithIndex;

// TODO: Unchecked indexing APIs
// TODO: Iterators should use unchecked indexing once fully stress tested.

/**
 * Access to the data in a Tensor with a particular order of dimension indexing.
 *
 * See the [module level documentation](crate::tensors::indexing) for more information.
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
            dimension_mapping: crate::tensors::dimensions::dimension_mapping(
                &source.view_shape(),
                &dimensions,
            )
            .ok_or_else(|| InvalidDimensionsError {
                actual: source.view_shape(),
                requested: dimensions,
            })?,
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
     * let tensor = Tensor::from([("x", 2), ("y", 2), ("z", 2)], vec![
     *     1, 2,
     *     3, 4,
     *
     *     5, 6,
     *     7, 8
     * ]);
     * let xyz = tensor.get(["x", "y", "z"]);
     * let also_xyz = TensorAccess::from_source_order(&tensor);
     * let also_xyz = tensor.source_order();
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
        crate::tensors::dimensions::dimension_mapping_shape(
            &self.source.view_shape(),
            &self.dimension_mapping,
        )
    }
}

/**
 * An error indicating failure to create a TensorAccess because the requested dimension order
 * does not match the set of dimensions in the source data.
 */
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct InvalidDimensionsError<const D: usize> {
    pub actual: [(Dimension, usize); D],
    pub requested: [Dimension; D],
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

impl<T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorRef<T, D>,
{
    pub fn try_get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.source
            .get_reference(crate::tensors::dimensions::map_dimensions(
                &self.dimension_mapping,
                &indexes,
            ))
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

    pub fn index_order_reference_iter(&self) -> IndexOrderReferenceIterator<T, S, D> {
        IndexOrderReferenceIterator::from(self)
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

    /**
     * Creates and returns a new tensor with all values from the original with the
     * function applied to each.
     */
    pub fn map<U>(&self, mapping_function: impl Fn(T) -> U) -> Tensor<U, D> {
        let mapped = self
            .index_order_iter()
            .map(mapping_function)
            .collect();
        Tensor::from(self.shape(), mapped)
    }

    /**
     * Creates and returns a new tensor with all values from the original and
     * the index of each value mapped by a function. The indexes passed to the mapping
     * function always increment the rightmost index, starting at all 0s, using the dimension
     * order that the TensorAccess is indexed by, not neccessarily the index order the
     * original source uses.
     */
    pub fn map_with_index<U>(&self, mapping_function: impl Fn([usize; D], T) -> U) -> Tensor<U, D> {
        let mapped = self
            .index_order_iter()
            .with_index()
            .map(|(i, x)| mapping_function(i, x))
            .collect();
        Tensor::from(self.shape(), mapped)
    }

    pub fn index_order_iter(&self) -> IndexOrderIterator<T, S, D> {
        IndexOrderIterator::from(self)
    }
}

impl<T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorMut<T, D>,
{
    pub fn try_get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.source
            .get_reference_mut(crate::tensors::dimensions::map_dimensions(
                &self.dimension_mapping,
                &indexes,
            ))
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

    pub fn index_order_reference_mut_iter(&mut self) -> IndexOrderReferenceMutIterator<T, S, D> {
        IndexOrderReferenceMutIterator::from(self)
    }
}

impl<T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorMut<T, D>,
    T: Clone,
{
    /**
     * Applies a function to all values in the tensor, modifying
     * the tensor in place.
     */
    pub fn map_mut(&mut self, mapping_function: impl Fn(T) -> T) {
        self.index_order_reference_mut_iter()
            .for_each(|x| *x = mapping_function(x.clone()));
    }

    /**
     * Applies a function to all values and each value's index in the tensor, modifying
     * the tensor in place. The indexes passed to the mapping function always increment
     * the rightmost index, starting at all 0s, using the dimension order that the
     * TensorAccess is indexed by, not neccessarily the index order the original source uses.
     */
    pub fn map_mut_with_index(&mut self, mapping_function: impl Fn([usize; D], T) -> T) {
        self
            .index_order_reference_mut_iter()
            .with_index()
            .for_each(|(i, x)| *x = mapping_function(i, x.clone()));
    }
}

// Common index order iterator logic
pub(crate) fn index_order_iter<const D: usize>(
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

/**
 * An iterator over copies of all values in a tensor.
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
 * let tensor_0 = Tensor::from_scalar(1);
 * let tensor_1 = Tensor::from([("a", 7)], vec![ 1, 2, 3, 4, 5, 6, 7 ]);
 * let tensor_2 = Tensor::from([("a", 2), ("b", 3)], vec![
 *    // two rows, three columns
 *    1, 2, 3,
 *    4, 5, 6
 * ]);
 * let tensor_3 = Tensor::from([("a", 2), ("b", 1), ("c", 2)], vec![
 *     // two rows each a single column, stacked on top of each other
 *     1,
 *     2,
 *
 *     3,
 *     4
 * ]);
 * let tensor_access_0 = tensor_0.get([]);
 * let tensor_access_1 = tensor_1.get(["a"]);
 * let tensor_access_2 = tensor_2.get(["a", "b"]);
 * let tensor_access_2_rev = tensor_2.get(["b", "a"]);
 * let tensor_access_3 = tensor_3.get(["a", "b", "c"]);
 * let tensor_access_3_rev = tensor_3.get(["c", "b", "a"]);
 * assert_eq!(
 *     tensor_access_0.index_order_iter().collect::<Vec<i32>>(),
 *     vec![1]
 * );
 * assert_eq!(
 *     tensor_access_1.index_order_iter().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6, 7]
 * );
 * assert_eq!(
 *     tensor_access_2.index_order_iter().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6]
 * );
 * assert_eq!(
 *     tensor_access_2_rev.index_order_iter().collect::<Vec<i32>>(),
 *     vec![1, 4, 2, 5, 3, 6]
 * );
 * assert_eq!(
 *     tensor_access_3.index_order_iter().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4]
 * );
 * assert_eq!(
 *     tensor_access_3_rev.index_order_iter().collect::<Vec<i32>>(),
 *     vec![1, 3, 2, 4]
 * );
 * ```
 */
#[derive(Debug)]
pub struct IndexOrderIterator<'a, T, S, const D: usize> {
    tensor: &'a TensorAccess<T, S, D>,
    shape: [(Dimension, usize); D],
    indexes: [usize; D],
    finished: bool,
}

impl<'a, T, S, const D: usize> IndexOrderIterator<'a, T, S, D>
where
    T: Clone,
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

    /**
     * Constructors an iterator which also yields the indexes of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T, S, const D: usize> Iterator for IndexOrderIterator<'a, T, S, D>
where
    T: Clone,
    S: TensorRef<T, D>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        index_order_iter(&mut self.finished, &mut self.indexes, &self.shape)
            .map(|indexes| self.tensor.get(indexes))
    }
}

impl<'a, T, S, const D: usize> Iterator for WithIndex<IndexOrderIterator<'a, T, S, D>>
where
    T: Clone,
    S: TensorRef<T, D>,
{
    type Item = ([usize; D], T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.iterator.indexes;
        self.iterator.next().map(|x| (index, x))
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
 * let tensor_0 = Tensor::from_scalar(1);
 * let tensor_1 = Tensor::from([("a", 7)], vec![ 1, 2, 3, 4, 5, 6, 7 ]);
 * let tensor_2 = Tensor::from([("a", 2), ("b", 3)], vec![
 *    // two rows, three columns
 *    1, 2, 3,
 *    4, 5, 6
 * ]);
 * let tensor_3 = Tensor::from([("a", 2), ("b", 1), ("c", 2)], vec![
 *     // two rows each a single column, stacked on top of each other
 *     1,
 *     2,
 *
 *     3,
 *     4
 * ]);
 * let tensor_access_0 = tensor_0.get([]);
 * let tensor_access_1 = tensor_1.get(["a"]);
 * let tensor_access_2 = tensor_2.get(["a", "b"]);
 * let tensor_access_2_rev = tensor_2.get(["b", "a"]);
 * let tensor_access_3 = tensor_3.get(["a", "b", "c"]);
 * let tensor_access_3_rev = tensor_3.get(["c", "b", "a"]);
 * assert_eq!(
 *     tensor_access_0.index_order_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1]
 * );
 * assert_eq!(
 *     tensor_access_1.index_order_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6, 7]
 * );
 * assert_eq!(
 *     tensor_access_2.index_order_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6]
 * );
 * assert_eq!(
 *     tensor_access_2_rev.index_order_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1, 4, 2, 5, 3, 6]
 * );
 * assert_eq!(
 *     tensor_access_3.index_order_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4]
 * );
 * assert_eq!(
 *     tensor_access_3_rev.index_order_reference_iter().cloned().collect::<Vec<i32>>(),
 *     vec![1, 3, 2, 4]
 * );
 * ```
 */
#[derive(Debug)]
pub struct IndexOrderReferenceIterator<'a, T, S, const D: usize> {
    tensor: &'a TensorAccess<T, S, D>,
    shape: [(Dimension, usize); D],
    indexes: [usize; D],
    finished: bool,
}

impl<'a, T, S, const D: usize> IndexOrderReferenceIterator<'a, T, S, D>
where
    S: TensorRef<T, D>,
{
    pub fn from(tensor_access: &TensorAccess<T, S, D>) -> IndexOrderReferenceIterator<T, S, D> {
        IndexOrderReferenceIterator {
            finished: !tensor_access.index_is_valid([0; D]),
            shape: tensor_access.shape(),
            tensor: tensor_access,
            indexes: [0; D],
        }
    }

    /**
     * Constructors an iterator which also yields the indexes of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T, S, const D: usize> Iterator for IndexOrderReferenceIterator<'a, T, S, D>
where
    S: TensorRef<T, D>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        index_order_iter(&mut self.finished, &mut self.indexes, &self.shape)
            .map(|indexes| self.tensor.get_reference(indexes))
    }
}

impl<'a, T, S, const D: usize> Iterator for WithIndex<IndexOrderReferenceIterator<'a, T, S, D>>
where
    S: TensorRef<T, D>,
{
    type Item = ([usize; D], &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.iterator.indexes;
        self.iterator.next().map(|x| (index, x))
    }
}

#[derive(Debug)]
pub struct IndexOrderReferenceMutIterator<'a, T, S, const D: usize> {
    tensor: &'a mut TensorAccess<T, S, D>,
    shape: [(Dimension, usize); D],
    indexes: [usize; D],
    finished: bool,
}

impl<'a, T, S, const D: usize> IndexOrderReferenceMutIterator<'a, T, S, D>
where
    S: TensorMut<T, D>,
{
    pub fn from(
        tensor_access: &mut TensorAccess<T, S, D>,
    ) -> IndexOrderReferenceMutIterator<T, S, D> {
        IndexOrderReferenceMutIterator {
            finished: !tensor_access.index_is_valid([0; D]),
            shape: tensor_access.shape(),
            tensor: tensor_access,
            indexes: [0; D],
        }
    }

    /**
     * Constructors an iterator which also yields the indexes of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T, S, const D: usize> Iterator for IndexOrderReferenceMutIterator<'a, T, S, D>
where
    S: TensorMut<T, D>,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        index_order_iter(&mut self.finished, &mut self.indexes, &self.shape).map(|indexes| {
            unsafe {
                // Safety: We are not allowed to give out overlapping mutable references,
                // but since we will always increment the counter on every call to next()
                // and stop when we reach the end no references will overlap.
                // The compiler doesn't know this, so transmute the lifetime for it.
                std::mem::transmute(self.tensor.get_reference_mut(indexes))
            }
        })
    }
}

impl<'a, T, S, const D: usize> Iterator for WithIndex<IndexOrderReferenceMutIterator<'a, T, S, D>>
where
    S: TensorMut<T, D>,
{
    type Item = ([usize; D], &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.iterator.indexes;
        self.iterator.next().map(|x| (index, x))
    }
}
