/*!
 * # Indexing
 *
 * Many libraries represent tensors as N dimensional arrays, however there is often some semantic
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
 * Easy ML moves the N dimensional array to an implementation detail, and most of its APIs work
 * on the names of each dimension in a tensor instead of just the order. Instead of a
 * 2000 x 100 x 100 x 3 tensor in which the last element is at [1999, 99, 99, 2], Easy ML tracks
 * the names of the dimensions, so you have a
 * `[("batch", 2000), ("width", 100), ("height", 100), ("rgb", 3)]` shaped tensor.
 *
 * This can't stop you from getting the math wrong, but confusion over which dimension
 * means what is reduced. Tensors carry around their pairs of dimension name and length
 * so adding a `[("batch", 2000), ("width", 100), ("height", 100), ("rgb", 3)]` shaped tensor
 * to a `[("batch", 2000), ("height", 100), ("width", 100), ("rgb", 3)]` will fail unless you
 * reorder one first, and you could access an element as
 * `tensor.index_by(["batch", "width", "height", "rgb"]).get([1999, 0, 99, 3])` or
 * `tensor.index_by(["batch", "height", "width", "rgb"]).get([1999, 99, 0, 3])` and read the same data,
 * because you index into dimensions based on their name, not just the order they are stored in
 * memory.
 *
 * Even with a name for each dimension, at some point you still need to say what order you want
 * to index each dimension with, and this is where [`TensorAccess`] comes in. It
 * creates a mapping from the dimension name order you want to access elements with to the order
 * the dimensions are stored as.
 */

use crate::differentiation::{Index, Primitive, Record, RecordTensor};
use crate::numeric::Numeric;
use crate::tensors::dimensions;
use crate::tensors::dimensions::DimensionMappings;
use crate::tensors::views::{DataLayout, TensorMut, TensorRef};
use crate::tensors::{Dimension, Tensor};

use std::error::Error;
use std::fmt;
use std::iter::{ExactSizeIterator, FusedIterator};
use std::marker::PhantomData;

pub use crate::matrices::iterators::WithIndex;

// TODO: Iterators should use unchecked indexing once fully stress tested.

/**
 * Access to the data in a Tensor with a particular order of dimension indexing. The order
 * affects the shape of the TensorAccess as well as the order of indexes you supply to read
 * or write values to the tensor.
 *
 * See the [module level documentation](crate::tensors::indexing) for more information.
 */
#[derive(Clone, Debug)]
pub struct TensorAccess<T, S, const D: usize> {
    source: S,
    dimension_mapping: DimensionMappings<D>,
    _type: PhantomData<T>,
}

impl<T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorRef<T, D>,
{
    /**
     * Creates a TensorAccess which can be indexed in the order of the supplied dimensions
     * to read or write values from this tensor.
     *
     * # Panics
     *
     * If the set of dimensions supplied do not match the set of dimensions in this tensor's shape.
     */
    #[track_caller]
    pub fn from(source: S, dimensions: [Dimension; D]) -> TensorAccess<T, S, D> {
        match TensorAccess::try_from(source, dimensions) {
            Err(error) => panic!("{}", error),
            Ok(success) => success,
        }
    }

    /**
     * Creates a TensorAccess which can be indexed in the order of the supplied dimensions
     * to read or write values from this tensor.
     *
     * Returns Err if the set of dimensions supplied do not match the set of dimensions in this
     * tensor's shape.
     */
    pub fn try_from(
        source: S,
        dimensions: [Dimension; D],
    ) -> Result<TensorAccess<T, S, D>, InvalidDimensionsError<D>> {
        Ok(TensorAccess {
            dimension_mapping: DimensionMappings::new(&source.view_shape(), &dimensions)
                .ok_or_else(|| InvalidDimensionsError {
                    actual: source.view_shape(),
                    requested: dimensions,
                })?,
            source,
            _type: PhantomData,
        })
    }

    /**
     * Creates a TensorAccess which is indexed in the same order as the dimensions in the view
     * shape of the tensor it is created from.
     *
     * Hence if you create a TensorAccess directly from a Tensor by `from_source_order`
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
     * let xyz = tensor.index_by(["x", "y", "z"]);
     * let also_xyz = TensorAccess::from_source_order(&tensor);
     * let also_xyz = tensor.index();
     * ```
     */
    pub fn from_source_order(source: S) -> TensorAccess<T, S, D> {
        TensorAccess {
            dimension_mapping: DimensionMappings::no_op_mapping(),
            source,
            _type: PhantomData,
        }
    }

    /**
     * Creates a TensorAccess which is indexed in the same order as the linear data layout
     * dimensions in the tensor it is created from, or None if the source data layout
     * is not linear.
     *
     * Hence if you use `from_memory_order` on a source that was originally big endian like
     * [Tensor] this uses the order for efficient iteration through each step in memory
     * when [iterating](TensorIterator).
     */
    pub fn from_memory_order(source: S) -> Option<TensorAccess<T, S, D>> {
        let data_layout = match source.data_layout() {
            DataLayout::Linear(order) => order,
            _ => return None,
        };
        let shape = source.view_shape();
        Some(TensorAccess::try_from(source, data_layout).unwrap_or_else(|_| panic!(
            "Source implementation contained dimensions {:?} in data_layout that were not the same set as in the view_shape {:?} which breaks the contract of TensorRef",
             data_layout, shape
        )))
    }

    /**
     * The shape this TensorAccess has with the dimensions mapped to the order the TensorAccess
     * was created with, not necessarily the same order as in the underlying tensor.
     */
    pub fn shape(&self) -> [(Dimension, usize); D] {
        self.dimension_mapping
            .map_shape_to_requested(&self.source.view_shape())
    }

    pub fn source(self) -> S {
        self.source
    }

    // # Safety
    //
    // Giving out a mutable reference to our source could allow it to be changed out from under us
    // and make our dimmension mapping invalid. However, since the source implements TensorRef
    // interior mutability is not allowed, so we can give out shared references without breaking
    // our own integrity.
    pub fn source_ref(&self) -> &S {
        &self.source
    }
}

/**
 * An error indicating failure to create a TensorAccess because the requested dimension order
 * does not match the shape in the source data.
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
            "Requested dimension order: {:?} does not match the shape in the source: {:?}",
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
    /**
     * Using the dimension ordering of the TensorAccess, gets a reference to the value at the
     * index if the index is in range. Otherwise returns None.
     */
    pub fn try_get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.source
            .get_reference(self.dimension_mapping.map_dimensions_to_source(&indexes))
    }

    /**
     * Using the dimension ordering of the TensorAccess, gets a reference to the value at the
     * index if the index is in range, panicking if the index is out of range.
     */
    // NOTE: Ideally `get_reference` would be used here for consistency, but that opens the
    // minefield of TensorRef::get_reference and TensorAccess::get_ref being different signatures
    // but the same name.
    #[track_caller]
    pub fn get_ref(&self, indexes: [usize; D]) -> &T {
        match self.try_get_reference(indexes) {
            Some(reference) => reference,
            None => panic!(
                "Unable to index with {:?}, Tensor dimensions are {:?}.",
                indexes,
                self.shape()
            ),
        }
    }

    /**
     * Using the dimension ordering of the TensorAccess, gets a reference to the value at the
     * index wihout any bounds checking.
     *
     * # Safety
     *
     * Calling this method with an out-of-bounds index is *[undefined behavior]* even if the
     * resulting reference is not used. Valid indexes are defined as in [TensorRef]. Note that
     * the order of the indexes needed here must match with
     * [`TensorAccess::shape`](TensorAccess::shape) which may not neccessarily be the same
     * as the `view_shape` of the `TensorRef` implementation this TensorAccess was created from).
     *
     * [undefined behavior]: <https://doc.rust-lang.org/reference/behavior-considered-undefined.html>
     * [TensorRef]: TensorRef
     */
    // NOTE: This aliases with TensorRef::get_reference_unchecked but the TensorRef impl
    // just calls this and the signatures match anyway, so there are no potential issues.
    #[allow(clippy::missing_safety_doc)] // it's not missing
    pub unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        unsafe {
            self.source
                .get_reference_unchecked(self.dimension_mapping.map_dimensions_to_source(&indexes))
        }
    }

    /**
     * Returns an iterator over references to the data in this TensorAccess, in the order of
     * the TensorAccess shape.
     */
    pub fn iter_reference(&self) -> TensorReferenceIterator<T, TensorAccess<T, S, D>, D> {
        TensorReferenceIterator::from(self)
    }
}

impl<T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorRef<T, D>,
    T: Clone,
{
    /**
     * Using the dimension ordering of the TensorAccess, gets a copy of the value at the
     * index if the index is in range, panicking if the index is out of range.
     *
     * For a non panicking API see [`try_get_reference`](TensorAccess::try_get_reference)
     */
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
     * Gets a copy of the first value in this tensor.
     * For 0 dimensional tensors this is the only index `[]`, for 1 dimensional tensors this
     * is `[0]`, for 2 dimensional tensors `[0,0]`, etcetera.
     */
    pub fn first(&self) -> T {
        self.iter()
            .next()
            .expect("Tensors always have at least 1 element")
    }

    /**
     * Creates and returns a new tensor with all values from the original with the
     * function applied to each.
     *
     * Note: mapping methods are defined on [Tensor] and
     * [TensorView](crate::tensors::views::TensorView) directly so you don't need to create a
     * TensorAccess unless you want to do the mapping with a different dimension order.
     */
    pub fn map<U>(&self, mapping_function: impl Fn(T) -> U) -> Tensor<U, D> {
        let mapped = self.iter().map(mapping_function).collect();
        Tensor::from(self.shape(), mapped)
    }

    /**
     * Creates and returns a new tensor with all values from the original and
     * the index of each value mapped by a function. The indexes passed to the mapping
     * function always increment the rightmost index, starting at all 0s, using the dimension
     * order that the TensorAccess is indexed by, not neccessarily the index order the
     * original source uses.
     *
     * Note: mapping methods are defined on [Tensor] and
     * [TensorView](crate::tensors::views::TensorView) directly so you don't need to create a
     * TensorAccess unless you want to do the mapping with a different dimension order.
     */
    pub fn map_with_index<U>(&self, mapping_function: impl Fn([usize; D], T) -> U) -> Tensor<U, D> {
        let mapped = self
            .iter()
            .with_index()
            .map(|(i, x)| mapping_function(i, x))
            .collect();
        Tensor::from(self.shape(), mapped)
    }

    /**
     * Returns an iterator over copies of the data in this TensorAccess, in the order of
     * the TensorAccess shape.
     */
    pub fn iter(&self) -> TensorIterator<T, TensorAccess<T, S, D>, D> {
        TensorIterator::from(self)
    }
}

impl<T, S, const D: usize> TensorAccess<T, S, D>
where
    S: TensorMut<T, D>,
{
    /**
     * Using the dimension ordering of the TensorAccess, gets a mutable reference to the value at
     * the index if the index is in range. Otherwise returns None.
     */
    pub fn try_get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.source
            .get_reference_mut(self.dimension_mapping.map_dimensions_to_source(&indexes))
    }

    /**
     * Using the dimension ordering of the TensorAccess, gets a mutable reference to the value at
     * the index if the index is in range, panicking if the index is out of range.
     */
    // NOTE: Ideally `get_reference_mut` would be used here for consistency, but that opens the
    // minefield of TensorMut::get_reference_mut and TensorAccess::get_ref_mut being different
    // signatures but the same name.
    #[track_caller]
    pub fn get_ref_mut(&mut self, indexes: [usize; D]) -> &mut T {
        match self.try_get_reference_mut(indexes) {
            Some(reference) => reference,
            // can't provide a better error because the borrow checker insists that returning
            // a reference in the Some branch means our mutable borrow prevents us calling
            // self.shape() and a bad error is better than cloning self.shape() on every call
            None => panic!("Unable to index with {:?}", indexes),
        }
    }

    /**
     * Using the dimension ordering of the TensorAccess, gets a mutable reference to the value at
     * the index wihout any bounds checking.
     *
     * # Safety
     *
     * Calling this method with an out-of-bounds index is *[undefined behavior]* even if the
     * resulting reference is not used. Valid indexes are defined as in [TensorRef]. Note that
     * the order of the indexes needed here must match with
     * [`TensorAccess::shape`](TensorAccess::shape) which may not neccessarily be the same
     * as the `view_shape` of the `TensorRef` implementation this TensorAccess was created from).
     *
     * [undefined behavior]: <https://doc.rust-lang.org/reference/behavior-considered-undefined.html>
     * [TensorRef]: TensorRef
     */
    // NOTE: This aliases with TensorRef::get_reference_unchecked_mut but the TensorMut impl
    // just calls this and the signatures match anyway, so there are no potential issues.
    #[allow(clippy::missing_safety_doc)] // it's not missing
    pub unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        unsafe {
            self.source.get_reference_unchecked_mut(
                self.dimension_mapping.map_dimensions_to_source(&indexes),
            )
        }
    }

    /**
     * Returns an iterator over mutable references to the data in this TensorAccess, in the order
     * of the TensorAccess shape.
     */
    pub fn iter_reference_mut(
        &mut self,
    ) -> TensorReferenceMutIterator<T, TensorAccess<T, S, D>, D> {
        TensorReferenceMutIterator::from(self)
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
        self.iter_reference_mut()
            .for_each(|x| *x = mapping_function(x.clone()));
    }

    /**
     * Applies a function to all values and each value's index in the tensor, modifying
     * the tensor in place. The indexes passed to the mapping function always increment
     * the rightmost index, starting at all 0s, using the dimension order that the
     * TensorAccess is indexed by, not neccessarily the index order the original source uses.
     */
    pub fn map_mut_with_index(&mut self, mapping_function: impl Fn([usize; D], T) -> T) {
        self.iter_reference_mut()
            .with_index()
            .for_each(|(i, x)| *x = mapping_function(i, x.clone()));
    }
}

impl<'a, T, S, const D: usize> TensorAccess<(T, Index), &RecordTensor<'a, T, S, D>, D>
where
    T: Numeric + Primitive,
    S: TensorRef<(T, Index), D>,
{
    /**
     * Using the dimension ordering of the TensorAccess, returns a copy of the data at the index
     * as a Record if the index is in range, panicking if the index is out of range.
     *
     * If you need to access all the data as records instead of just a specific index you should
     * probably use one of the iterator APIs instead.
     *
     * See also: [iter_as_records](RecordTensor::iter_as_records)
     *
     * # Panics
     *
     * If the index is out of range.
     *
     * For a non panicking API see [try_get_as_record](TensorAccess::try_get_as_record)
     *
     * ```
     * use easy_ml::differentiation::RecordTensor;
     * use easy_ml::differentiation::WengertList;
     * use easy_ml::tensors::Tensor;
     *
     * let list = WengertList::new();
     * let X = RecordTensor::variables(
     *     &list,
     *     Tensor::from(
     *         [("r", 2), ("c", 3)],
     *         vec![
     *             3.0, 4.0, 5.0,
     *             1.0, 4.0, 9.0,
     *         ]
     *     )
     * );
     * let x = X.index_by(["c", "r"]).get_as_record([2, 0]);
     * assert_eq!(x.number, 5.0);
     * ```
     */
    #[track_caller]
    pub fn get_as_record(&self, indexes: [usize; D]) -> Record<'a, T> {
        Record::from_existing(self.get(indexes), self.source.history())
    }

    /**
     * Using the dimension ordering of the TensorAccess, returns a copy of the data at the index
     * as a Record if the index is in range. Otherwise returns None.
     *
     * If you need to access all the data as records instead of just a specific index you should
     * probably use one of the iterator APIs instead.
     *
     * See also: [iter_as_records](RecordTensor::iter_as_records)
     */
    pub fn try_get_as_record(&self, indexes: [usize; D]) -> Option<Record<'a, T>> {
        self.try_get_reference(indexes)
            .map(|r| Record::from_existing(r.clone(), self.source.history()))
    }
}

impl<'a, T, S, const D: usize> TensorAccess<(T, Index), RecordTensor<'a, T, S, D>, D>
where
    T: Numeric + Primitive,
    S: TensorRef<(T, Index), D>,
{
    /**
     * Using the dimension ordering of the TensorAccess, returns a copy of the data at the index
     * as a Record if the index is in range, panicking if the index is out of range.
     *
     * If you need to access all the data as records instead of just a specific index you should
     * probably use one of the iterator APIs instead.
     *
     * See also: [iter_as_records](RecordTensor::iter_as_records)
     *
     * # Panics
     *
     * If the index is out of range.
     *
     * For a non panicking API see [try_get_as_record](TensorAccess::try_get_as_record)
     */
    #[track_caller]
    pub fn get_as_record(&self, indexes: [usize; D]) -> Record<'a, T> {
        Record::from_existing(self.get(indexes), self.source.history())
    }

    /**
     * Using the dimension ordering of the TensorAccess, returns a copy of the data at the index
     * as a Record if the index is in range. Otherwise returns None.
     *
     * If you need to access all the data as records instead of just a specific index you should
     * probably use one of the iterator APIs instead.
     *
     * See also: [iter_as_records](RecordTensor::iter_as_records)
     */
    pub fn try_get_as_record(&self, indexes: [usize; D]) -> Option<Record<'a, T>> {
        self.try_get_reference(indexes)
            .map(|r| Record::from_existing(r.clone(), self.source.history()))
    }
}

impl<'a, T, S, const D: usize> TensorAccess<(T, Index), &mut RecordTensor<'a, T, S, D>, D>
where
    T: Numeric + Primitive,
    S: TensorRef<(T, Index), D>,
{
    /**
     * Using the dimension ordering of the TensorAccess, returns a copy of the data at the index
     * as a Record if the index is in range, panicking if the index is out of range.
     *
     * If you need to access all the data as records instead of just a specific index you should
     * probably use one of the iterator APIs instead.
     *
     * See also: [iter_as_records](RecordTensor::iter_as_records)
     *
     * # Panics
     *
     * If the index is out of range.
     *
     * For a non panicking API see [try_get_as_record](TensorAccess::try_get_as_record)
     */
    #[track_caller]
    pub fn get_as_record(&self, indexes: [usize; D]) -> Record<'a, T> {
        Record::from_existing(self.get(indexes), self.source.history())
    }

    /**
     * Using the dimension ordering of the TensorAccess, returns a copy of the data at the index
     * as a Record if the index is in range. Otherwise returns None.
     *
     * If you need to access all the data as records instead of just a specific index you should
     * probably use one of the iterator APIs instead.
     *
     * See also: [iter_as_records](RecordTensor::iter_as_records)
     */
    pub fn try_get_as_record(&self, indexes: [usize; D]) -> Option<Record<'a, T>> {
        self.try_get_reference(indexes)
            .map(|r| Record::from_existing(r.clone(), self.source.history()))
    }
}

// # Safety
//
// The type implementing TensorRef inside the TensorAccess must implement it correctly, so by
// delegating to it without changing anything other than the order we index it, we implement
// TensorRef correctly as well.
/**
 * A TensorAccess implements TensorRef, with the dimension order and indexing matching that of the
 * TensorAccess shape.
 */
unsafe impl<T, S, const D: usize> TensorRef<T, D> for TensorAccess<T, S, D>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.try_get_reference(indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.shape()
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        unsafe { self.get_reference_unchecked(indexes) }
    }

    fn data_layout(&self) -> DataLayout<D> {
        match self.source.data_layout() {
            // We might have reordered the view_shape but we didn't rearrange the memory or change
            // what each dimension name refers to in memory, so the data layout remains as is.
            DataLayout::Linear(order) => DataLayout::Linear(order),
            DataLayout::NonLinear => DataLayout::NonLinear,
            DataLayout::Other => DataLayout::Other,
        }
    }
}

// # Safety
//
// The type implementing TensorMut inside the TensorAccess must implement it correctly, so by
// delegating to it without changing anything other than the order we index it, we implement
// TensorMut correctly as well.
/**
 * A TensorAccess implements TensorMut, with the dimension order and indexing matching that of the
 * TensorAccess shape.
 */
unsafe impl<T, S, const D: usize> TensorMut<T, D> for TensorAccess<T, S, D>
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.try_get_reference_mut(indexes)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        unsafe { self.get_reference_unchecked_mut(indexes) }
    }
}

/**
 * Any tensor access of a Displayable type implements Display
 *
 * You can control the precision of the formatting using format arguments, i.e.
 * `format!("{:.3}", tensor)`
 */
impl<T: std::fmt::Display, S, const D: usize> std::fmt::Display for TensorAccess<T, S, D>
where
    T: std::fmt::Display,
    S: TensorRef<T, D>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(&self, f)?;
        writeln!(f)?;
        write!(f, "Data Layout = {:?}", self.data_layout())
    }
}

/**
 * An iterator over all indexes in a shape.
 *
 * First the all 0 index is iterated, then each iteration increments the rightmost index.
 * For a shape of `[("a", 2), ("b", 2), ("c", 2)]` this will yield indexes in order of: `[0,0,0]`,
 * `[0,0,1]`, `[0,1,0]`, `[0,1,1]`, `[1,0,0]`, `[1,0,1]`, `[1,1,0]`, `[1,1,1]`,
 *
 * You don't typically need to use this directly, as tensors have iterators that iterate over
 * them and return values to you (using this under the hood), but `ShapeIterator` can be useful
 * if you need to hold a mutable reference to a tensor while iterating as `ShapeIterator` does
 * not borrow the tensor. NB: if you do index into a tensor you're mutably borrowing using
 * `ShapeIterator` directly, take care to ensure you don't accidentally reshape the tensor and
 * continue to use indexes from `ShapeIterator` as they would then be invalid.
 */
#[derive(Clone, Debug)]
pub struct ShapeIterator<const D: usize> {
    shape: [(Dimension, usize); D],
    indexes: [usize; D],
    finished: bool,
}

impl<const D: usize> ShapeIterator<D> {
    /**
     * Constructs a ShapeIterator for a shape.
     *
     * If the shape has any dimensions with a length of zero, the iterator will immediately
     * return None on [`next()`](Iterator::next).
     */
    pub fn from(shape: [(Dimension, usize); D]) -> ShapeIterator<D> {
        // If we're given an invalid shape (shape input is not neccessarily going to meet the no
        // 0 lengths contract of TensorRef because that's not actually required here), return
        // a finished iterator
        // Since this is an iterator over an owned shape, it's not going to become invalid later
        // when we start iterating so this is the only check we need.
        let starting_index_valid = shape.iter().all(|(_, l)| *l > 0);
        ShapeIterator {
            shape,
            indexes: [0; D],
            finished: !starting_index_valid,
        }
    }
}

impl<const D: usize> Iterator for ShapeIterator<D> {
    type Item = [usize; D];

    fn next(&mut self) -> Option<Self::Item> {
        iter(&mut self.finished, &mut self.indexes, &self.shape)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        size_hint(self.finished, &self.indexes, &self.shape)
    }
}

// Once we hit the end we mark ourselves as finished so we're always Fused.
impl<const D: usize> FusedIterator for ShapeIterator<D> {}
// We can always calculate the exact number of steps remaining because the shape and indexes are
// private fields that are only mutated by `next` to count up.
impl<const D: usize> ExactSizeIterator for ShapeIterator<D> {}

/// Common index order iterator logic
fn iter<const D: usize>(
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

/// Common size hint logic
fn size_hint<const D: usize>(
    finished: bool,
    indexes: &[usize; D],
    shape: &[(Dimension, usize); D],
) -> (usize, Option<usize>) {
    if finished {
        return (0, Some(0));
    }

    let remaining = if D > 0 {
        let total = dimensions::elements(shape);
        let strides = crate::tensors::compute_strides(shape);
        let seen = crate::tensors::get_index_direct_unchecked(indexes, &strides);
        total - seen
    } else {
        1
        // If D == 0 and we're not finished we've not returned the sole index yet so there's
        // exactly 1 left
    };

    (remaining, Some(remaining))
}

/**
 * An iterator over copies of all values in a tensor.
 *
 * First the all 0 index is iterated, then each iteration increments the rightmost index.
 * For [Tensor] or [TensorRef]s which do not reorder the underlying Tensor
 * this will take a single step in memory on each iteration, akin to iterating through the
 * flattened data of the tensor.
 *
 * If the TensorRef reorders the tensor data (e.g. [TensorAccess]) this iterator
 * will still iterate the rightmost index allowing iteration through dimensions in a different
 * order to how they are stored, but no longer taking a single step in memory on each
 * iteration (which may be less cache friendly for the CPU).
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
 * let tensor_access_0 = tensor_0.index_by([]);
 * let tensor_access_1 = tensor_1.index_by(["a"]);
 * let tensor_access_2 = tensor_2.index_by(["a", "b"]);
 * let tensor_access_2_rev = tensor_2.index_by(["b", "a"]);
 * let tensor_access_3 = tensor_3.index_by(["a", "b", "c"]);
 * let tensor_access_3_rev = tensor_3.index_by(["c", "b", "a"]);
 * assert_eq!(
 *     tensor_0.iter().collect::<Vec<i32>>(),
 *     vec![1]
 * );
 * assert_eq!(
 *     tensor_access_0.iter().collect::<Vec<i32>>(),
 *     vec![1]
 * );
 * assert_eq!(
 *     tensor_1.iter().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6, 7]
 * );
 * assert_eq!(
 *     tensor_access_1.iter().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6, 7]
 * );
 * assert_eq!(
 *     tensor_2.iter().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6]
 * );
 * assert_eq!(
 *     tensor_access_2.iter().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6]
 * );
 * assert_eq!(
 *     tensor_access_2_rev.iter().collect::<Vec<i32>>(),
 *     vec![1, 4, 2, 5, 3, 6]
 * );
 * assert_eq!(
 *     tensor_3.iter().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4]
 * );
 * assert_eq!(
 *     tensor_access_3.iter().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4]
 * );
 * assert_eq!(
 *     tensor_access_3_rev.iter().collect::<Vec<i32>>(),
 *     vec![1, 3, 2, 4]
 * );
 * ```
 */
#[derive(Debug)]
pub struct TensorIterator<'a, T, S, const D: usize> {
    shape_iterator: ShapeIterator<D>,
    source: &'a S,
    _type: PhantomData<T>,
}

impl<'a, T, S, const D: usize> TensorIterator<'a, T, S, D>
where
    T: Clone,
    S: TensorRef<T, D>,
{
    pub fn from(source: &S) -> TensorIterator<T, S, D> {
        TensorIterator {
            shape_iterator: ShapeIterator::from(source.view_shape()),
            source,
            _type: PhantomData,
        }
    }

    /**
     * Constructs an iterator which also yields the indexes of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T, S, const D: usize> From<TensorIterator<'a, T, S, D>>
    for WithIndex<TensorIterator<'a, T, S, D>>
where
    T: Clone,
    S: TensorRef<T, D>,
{
    fn from(iterator: TensorIterator<'a, T, S, D>) -> Self {
        iterator.with_index()
    }
}

impl<'a, T, S, const D: usize> Iterator for TensorIterator<'a, T, S, D>
where
    T: Clone,
    S: TensorRef<T, D>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // Safety: ShapeIterator only iterates over the correct indexes into our tensor's shape as
        // defined by TensorRef. Since TensorRef promises no interior mutability and we hold an
        // immutable reference to our tensor source, it can't be resized which ensures
        // ShapeIterator can always yield valid indexes for our iteration.
        self.shape_iterator
            .next()
            .map(|indexes| unsafe { self.source.get_reference_unchecked(indexes) }.clone())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.shape_iterator.size_hint()
    }
}

impl<'a, T, S, const D: usize> FusedIterator for TensorIterator<'a, T, S, D>
where
    T: Clone,
    S: TensorRef<T, D>,
{
}

impl<'a, T, S, const D: usize> ExactSizeIterator for TensorIterator<'a, T, S, D>
where
    T: Clone,
    S: TensorRef<T, D>,
{
}

impl<'a, T, S, const D: usize> Iterator for WithIndex<TensorIterator<'a, T, S, D>>
where
    T: Clone,
    S: TensorRef<T, D>,
{
    type Item = ([usize; D], T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.iterator.shape_iterator.indexes;
        self.iterator.next().map(|x| (index, x))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iterator.size_hint()
    }
}

impl<'a, T, S, const D: usize> FusedIterator for WithIndex<TensorIterator<'a, T, S, D>>
where
    T: Clone,
    S: TensorRef<T, D>,
{
}

impl<'a, T, S, const D: usize> ExactSizeIterator for WithIndex<TensorIterator<'a, T, S, D>>
where
    T: Clone,
    S: TensorRef<T, D>,
{
}

/**
 * An iterator over references to all values in a tensor.
 *
 * First the all 0 index is iterated, then each iteration increments the rightmost index.
 * For [Tensor] or [TensorRef]s which do not reorder the underlying Tensor
 * this will take a single step in memory on each iteration, akin to iterating through the
 * flattened data of the tensor.
 *
 * If the TensorRef reorders the tensor data (e.g. [TensorAccess]) this iterator
 * will still iterate the rightmost index allowing iteration through dimensions in a different
 * order to how they are stored, but no longer taking a single step in memory on each
 * iteration (which may be less cache friendly for the CPU).
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
 * let tensor_access_0 = tensor_0.index_by([]);
 * let tensor_access_1 = tensor_1.index_by(["a"]);
 * let tensor_access_2 = tensor_2.index_by(["a", "b"]);
 * let tensor_access_2_rev = tensor_2.index_by(["b", "a"]);
 * let tensor_access_3 = tensor_3.index_by(["a", "b", "c"]);
 * let tensor_access_3_rev = tensor_3.index_by(["c", "b", "a"]);
 * assert_eq!(
 *     tensor_0.iter_reference().cloned().collect::<Vec<i32>>(),
 *     vec![1]
 * );
 * assert_eq!(
 *     tensor_access_0.iter_reference().cloned().collect::<Vec<i32>>(),
 *     vec![1]
 * );
 * assert_eq!(
 *     tensor_1.iter_reference().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6, 7]
 * );
 * assert_eq!(
 *     tensor_access_1.iter_reference().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6, 7]
 * );
 * assert_eq!(
 *     tensor_2.iter_reference().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6]
 * );
 * assert_eq!(
 *     tensor_access_2.iter_reference().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4, 5, 6]
 * );
 * assert_eq!(
 *     tensor_access_2_rev.iter_reference().cloned().collect::<Vec<i32>>(),
 *     vec![1, 4, 2, 5, 3, 6]
 * );
 * assert_eq!(
 *     tensor_3.iter_reference().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4]
 * );
 * assert_eq!(
 *     tensor_access_3.iter_reference().cloned().collect::<Vec<i32>>(),
 *     vec![1, 2, 3, 4]
 * );
 * assert_eq!(
 *     tensor_access_3_rev.iter_reference().cloned().collect::<Vec<i32>>(),
 *     vec![1, 3, 2, 4]
 * );
 * ```
 */
#[derive(Debug)]
pub struct TensorReferenceIterator<'a, T, S, const D: usize> {
    shape_iterator: ShapeIterator<D>,
    source: &'a S,
    _type: PhantomData<&'a T>,
}

impl<'a, T, S, const D: usize> TensorReferenceIterator<'a, T, S, D>
where
    S: TensorRef<T, D>,
{
    pub fn from(source: &S) -> TensorReferenceIterator<T, S, D> {
        TensorReferenceIterator {
            shape_iterator: ShapeIterator::from(source.view_shape()),
            source,
            _type: PhantomData,
        }
    }

    /**
     * Constructs an iterator which also yields the indexes of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T, S, const D: usize> From<TensorReferenceIterator<'a, T, S, D>>
    for WithIndex<TensorReferenceIterator<'a, T, S, D>>
where
    S: TensorRef<T, D>,
{
    fn from(iterator: TensorReferenceIterator<'a, T, S, D>) -> Self {
        iterator.with_index()
    }
}

impl<'a, T, S, const D: usize> Iterator for TensorReferenceIterator<'a, T, S, D>
where
    S: TensorRef<T, D>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        // Safety: ShapeIterator only iterates over the correct indexes into our tensor's shape as
        // defined by TensorRef. Since TensorRef promises no interior mutability and we hold an
        // immutable reference to our tensor source, it can't be resized which ensures
        // ShapeIterator can always yield valid indexes for our iteration.
        self.shape_iterator
            .next()
            .map(|indexes| unsafe { self.source.get_reference_unchecked(indexes) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.shape_iterator.size_hint()
    }
}

impl<'a, T, S, const D: usize> FusedIterator for TensorReferenceIterator<'a, T, S, D> where
    S: TensorRef<T, D>
{
}

impl<'a, T, S, const D: usize> ExactSizeIterator for TensorReferenceIterator<'a, T, S, D> where
    S: TensorRef<T, D>
{
}

impl<'a, T, S, const D: usize> Iterator for WithIndex<TensorReferenceIterator<'a, T, S, D>>
where
    S: TensorRef<T, D>,
{
    type Item = ([usize; D], &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.iterator.shape_iterator.indexes;
        self.iterator.next().map(|x| (index, x))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iterator.size_hint()
    }
}

impl<'a, T, S, const D: usize> FusedIterator for WithIndex<TensorReferenceIterator<'a, T, S, D>> where
    S: TensorRef<T, D>
{
}

impl<'a, T, S, const D: usize> ExactSizeIterator for WithIndex<TensorReferenceIterator<'a, T, S, D>> where
    S: TensorRef<T, D>
{
}

/**
 * An iterator over mutable references to all values in a tensor.
 *
 * First the all 0 index is iterated, then each iteration increments the rightmost index.
 * For [Tensor] or [TensorRef]s which do not reorder the underlying Tensor
 * this will take a single step in memory on each iteration, akin to iterating through the
 * flattened data of the tensor.
 *
 * If the TensorRef reorders the tensor data (e.g. [TensorAccess]) this iterator
 * will still iterate the rightmost index allowing iteration through dimensions in a different
 * order to how they are stored, but no longer taking a single step in memory on each
 * iteration (which may be less cache friendly for the CPU).
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * let mut tensor = Tensor::from([("a", 7)], vec![ 1, 2, 3, 4, 5, 6, 7 ]);
 * let doubled = tensor.map(|x| 2 * x);
 * // mutating a tensor in place can also be done with Tensor::map_mut and
 * // Tensor::map_mut_with_index
 * for elem in tensor.iter_reference_mut() {
 *    *elem = 2 * *elem;
 * }
 * assert_eq!(
 *     tensor,
 *     doubled,
 * );
 * ```
 */
#[derive(Debug)]
pub struct TensorReferenceMutIterator<'a, T, S, const D: usize> {
    shape_iterator: ShapeIterator<D>,
    source: &'a mut S,
    _type: PhantomData<&'a mut T>,
}

impl<'a, T, S, const D: usize> TensorReferenceMutIterator<'a, T, S, D>
where
    S: TensorMut<T, D>,
{
    pub fn from(source: &mut S) -> TensorReferenceMutIterator<T, S, D> {
        TensorReferenceMutIterator {
            shape_iterator: ShapeIterator::from(source.view_shape()),
            source,
            _type: PhantomData,
        }
    }

    /**
     * Constructs an iterator which also yields the indexes of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T, S, const D: usize> From<TensorReferenceMutIterator<'a, T, S, D>>
    for WithIndex<TensorReferenceMutIterator<'a, T, S, D>>
where
    S: TensorMut<T, D>,
{
    fn from(iterator: TensorReferenceMutIterator<'a, T, S, D>) -> Self {
        iterator.with_index()
    }
}

impl<'a, T, S, const D: usize> Iterator for TensorReferenceMutIterator<'a, T, S, D>
where
    S: TensorMut<T, D>,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.shape_iterator.next().map(|indexes| {
            unsafe {
                // Safety: We are not allowed to give out overlapping mutable references,
                // but since we will always increment the counter on every call to next()
                // and stop when we reach the end no references will overlap.
                // The compiler doesn't know this, so transmute the lifetime for it.
                // Safety: ShapeIterator only iterates over the correct indexes into our
                // tensor's shape as defined by TensorRef. Since TensorRef promises no interior
                // mutability and we hold an exclusive reference to our tensor source, it can't
                // be resized (except by us - and we don't) which ensures ShapeIterator can always
                // yield valid indexes for our iteration.
                std::mem::transmute(self.source.get_reference_unchecked_mut(indexes))
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.shape_iterator.size_hint()
    }
}

impl<'a, T, S, const D: usize> FusedIterator for TensorReferenceMutIterator<'a, T, S, D> where
    S: TensorMut<T, D>
{
}

impl<'a, T, S, const D: usize> ExactSizeIterator for TensorReferenceMutIterator<'a, T, S, D> where
    S: TensorMut<T, D>
{
}

impl<'a, T, S, const D: usize> Iterator for WithIndex<TensorReferenceMutIterator<'a, T, S, D>>
where
    S: TensorMut<T, D>,
{
    type Item = ([usize; D], &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.iterator.shape_iterator.indexes;
        self.iterator.next().map(|x| (index, x))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iterator.size_hint()
    }
}

impl<'a, T, S, const D: usize> FusedIterator for WithIndex<TensorReferenceMutIterator<'a, T, S, D>> where
    S: TensorMut<T, D>
{
}

impl<'a, T, S, const D: usize> ExactSizeIterator
    for WithIndex<TensorReferenceMutIterator<'a, T, S, D>>
where
    S: TensorMut<T, D>,
{
}

/**
 * An iterator over all values in an owned tensor.
 *
 * This iterator does not clone the values, it returns the actual values stored in the tensor.
 * There is no such method to return `T` by value from a [TensorRef]/[TensorMut], to do
 * this it [replaces](std::mem::replace) the values with dummy values. Hence it can only be
 * created for types that implement [Default] or [ZeroOne](crate::numeric::ZeroOne)
 * from [Numeric](crate::numeric) which provide a means to create dummy values.
 *
 * First the all 0 index is iterated, then each iteration increments the rightmost index.
 * For [Tensor] or [TensorRef]s which do not reorder the underlying Tensor
 * this will take a single step in memory on each iteration, akin to iterating through the
 * flattened data of the tensor.
 *
 * If the TensorRef reorders the tensor data (e.g. [TensorAccess]) this iterator
 * will still iterate the rightmost index allowing iteration through dimensions in a different
 * order to how they are stored, but no longer taking a single step in memory on each
 * iteration (which may be less cache friendly for the CPU).
 *
 * ```
 * use easy_ml::tensors::Tensor;
 *
 * #[derive(Debug, Default, Eq, PartialEq)]
 * struct NoClone(i32);
 *
 * let tensor = Tensor::from([("a", 3)], vec![ NoClone(1), NoClone(2), NoClone(3) ]);
 * let values = tensor.iter_owned(); // will use T::default() for dummy values
 * assert_eq!(vec![ NoClone(1), NoClone(2), NoClone(3) ], values.collect::<Vec<NoClone>>());
 * ```
 */
#[derive(Debug)]
pub struct TensorOwnedIterator<T, S, const D: usize> {
    shape_iterator: ShapeIterator<D>,
    source: S,
    producer: fn() -> T,
}

impl<T, S, const D: usize> TensorOwnedIterator<T, S, D>
where
    S: TensorMut<T, D>,
{
    /**
     * Creates the TensorOwnedIterator from a source where the default values will be provided
     * by [Default::default]. This constructor is also used by the convenience
     * methods on [Tensor::iter_owned](Tensor::iter_owned) and
     * [TensorView::iter_owned](crate::tensors::views::TensorView::iter_owned).
     */
    pub fn from(source: S) -> TensorOwnedIterator<T, S, D>
    where
        T: Default,
    {
        TensorOwnedIterator {
            shape_iterator: ShapeIterator::from(source.view_shape()),
            source,
            producer: || T::default(),
        }
    }

    /**
     * Creates the TensorOwnedIterator from a source where the default values will be provided
     * by [ZeroOne::zero](crate::numeric::ZeroOne::zero).
     */
    pub fn from_numeric(source: S) -> TensorOwnedIterator<T, S, D>
    where
        T: crate::numeric::ZeroOne,
    {
        TensorOwnedIterator {
            shape_iterator: ShapeIterator::from(source.view_shape()),
            source,
            producer: || T::zero(),
        }
    }

    /**
     * Constructs an iterator which also yields the indexes of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<T, S, const D: usize> From<TensorOwnedIterator<T, S, D>>
    for WithIndex<TensorOwnedIterator<T, S, D>>
where
    S: TensorMut<T, D>,
{
    fn from(iterator: TensorOwnedIterator<T, S, D>) -> Self {
        iterator.with_index()
    }
}

impl<T, S, const D: usize> Iterator for TensorOwnedIterator<T, S, D>
where
    S: TensorMut<T, D>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.shape_iterator.next().map(|indexes| {
            let producer = self.producer;
            let dummy = producer();
            // Safety: ShapeIterator only iterates over the correct indexes into our
            // tensor's shape as defined by TensorRef. Since TensorRef promises no interior
            // mutability and we hold our tensor source by value, it can't be resized (except by
            // us - and we don't) which ensures ShapeIterator can always yield valid indexes for
            // our iteration.
            let value = std::mem::replace(
                unsafe { self.source.get_reference_unchecked_mut(indexes) },
                dummy,
            );
            value
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.shape_iterator.size_hint()
    }
}

impl<T, S, const D: usize> FusedIterator for TensorOwnedIterator<T, S, D> where S: TensorMut<T, D> {}

impl<T, S, const D: usize> ExactSizeIterator for TensorOwnedIterator<T, S, D> where
    S: TensorMut<T, D>
{
}

impl<T, S, const D: usize> Iterator for WithIndex<TensorOwnedIterator<T, S, D>>
where
    S: TensorMut<T, D>,
{
    type Item = ([usize; D], T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.iterator.shape_iterator.indexes;
        self.iterator.next().map(|x| (index, x))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iterator.size_hint()
    }
}

impl<T, S, const D: usize> FusedIterator for WithIndex<TensorOwnedIterator<T, S, D>> where
    S: TensorMut<T, D>
{
}

impl<T, S, const D: usize> ExactSizeIterator for WithIndex<TensorOwnedIterator<T, S, D>> where
    S: TensorMut<T, D>
{
}

/**
 * A TensorTranspose makes the data in the tensor it is created from appear to be in a different
 * order, swapping the lengths of each named dimension to match the new order but leaving the
 * dimension name order unchanged.
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * use easy_ml::tensors::indexing::TensorTranspose;
 * use easy_ml::tensors::views::TensorView;
 * let tensor = Tensor::from([("batch", 2), ("rows", 3), ("columns", 2)], vec![
 *     1, 2,
 *     3, 4,
 *     5, 6,
 *
 *     7, 8,
 *     9, 0,
 *     1, 2
 * ]);
 * let transposed = TensorView::from(TensorTranspose::from(&tensor, ["batch", "columns", "rows"]));
 * assert_eq!(
 *     transposed,
 *     Tensor::from([("batch", 2), ("rows", 2), ("columns", 3)], vec![
 *         1, 3, 5,
 *         2, 4, 6,
 *
 *         7, 9, 1,
 *         8, 0, 2
 *     ])
 * );
 * let also_transposed = tensor.transpose_view(["batch", "columns", "rows"]);
 * ```
 */
#[derive(Clone)]
pub struct TensorTranspose<T, S, const D: usize> {
    access: TensorAccess<T, S, D>,
}

impl<T: fmt::Debug, S: fmt::Debug, const D: usize> fmt::Debug for TensorTranspose<T, S, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorTranspose")
            .field("source", &self.access.source)
            .field("dimension_mapping", &self.access.dimension_mapping)
            .field("_type", &self.access._type)
            .finish()
    }
}

impl<T, S, const D: usize> TensorTranspose<T, S, D>
where
    S: TensorRef<T, D>,
{
    /**
     * Creates a TensorTranspose which makes the data appear in the order of the
     * supplied dimensions. The order of the dimension names is unchanged, although their lengths
     * may swap.
     *
     * # Panics
     *
     * If the set of dimensions in the tensor does not match the set of dimensions provided. The
     * order need not match.
     */
    #[track_caller]
    pub fn from(source: S, dimensions: [Dimension; D]) -> TensorTranspose<T, S, D> {
        TensorTranspose {
            access: match TensorAccess::try_from(source, dimensions) {
                Err(error) => panic!("{}", error),
                Ok(success) => success,
            },
        }
    }

    /**
     * Creates a TensorTranspose which makes the data to appear in the order of the
     * supplied dimensions. The order of the dimension names is unchanged, although their lengths
     * may swap.
     *
     * Returns Err if the set of dimensions supplied do not match the set of dimensions in this
     * tensor's shape.
     */
    pub fn try_from(
        source: S,
        dimensions: [Dimension; D],
    ) -> Result<TensorTranspose<T, S, D>, InvalidDimensionsError<D>> {
        TensorAccess::try_from(source, dimensions).map(|access| TensorTranspose { access })
    }

    /**
     * The shape of this TensorTranspose appears to rearrange the data to the order of supplied
     * dimensions. The actual data in the underlying tensor and the order of the dimension names
     * on this TensorTranspose remains unchanged, although the lengths of the dimensions in this
     * shape of may swap compared to the source's shape.
     */
    pub fn shape(&self) -> [(Dimension, usize); D] {
        let names = self.access.source.view_shape();
        let order = self.access.shape();
        std::array::from_fn(|d| (names[d].0, order[d].1))
    }

    pub fn source(self) -> S {
        self.access.source
    }

    // # Safety
    //
    // Giving out a mutable reference to our source could allow it to be changed out from under us
    // and make our dimmension mapping invalid. However, since the source implements TensorRef
    // interior mutability is not allowed, so we can give out shared references without breaking
    // our own integrity.
    pub fn source_ref(&self) -> &S {
        &self.access.source
    }
}

// # Safety
//
// The TensorAccess must implement TensorRef correctly, so by delegating to it without changing
// anything other than the order of the dimension names we expose, we implement
// TensoTensorRefrMut correctly as well.
/**
 * A TensorTranspose implements TensorRef, with the dimension order and indexing matching that
 * of the TensorTranspose shape.
 */
unsafe impl<T, S, const D: usize> TensorRef<T, D> for TensorTranspose<T, S, D>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        // we didn't change the lengths of any dimension in our shape from the TensorAccess so we
        // can delegate to the tensor access for non named indexing here
        self.access.try_get_reference(indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.shape()
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        unsafe { self.access.get_reference_unchecked(indexes) }
    }

    fn data_layout(&self) -> DataLayout<D> {
        let data_layout = self.access.data_layout();
        match data_layout {
            DataLayout::Linear(order) => DataLayout::Linear(
                self.access
                    .dimension_mapping
                    .map_linear_data_layout_to_transposed(&order),
            ),
            _ => data_layout,
        }
    }
}

// # Safety
//
// The TensorAccess must implement TensorMut correctly, so so by delegating to it without changing
// anything other than the order of the dimension names we expose, we implement, we implement
// TensorMut correctly as well.
/**
 * A TensorTranspose implements TensorMut, with the dimension order and indexing matching that of
 * the TensorTranspose shape.
 */
unsafe impl<T, S, const D: usize> TensorMut<T, D> for TensorTranspose<T, S, D>
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.access.try_get_reference_mut(indexes)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        unsafe { self.access.get_reference_unchecked_mut(indexes) }
    }
}

/**
 * Any tensor transpose of a Displayable type implements Display
 *
 * You can control the precision of the formatting using format arguments, i.e.
 * `format!("{:.3}", tensor)`
 */
impl<T: std::fmt::Display, S, const D: usize> std::fmt::Display for TensorTranspose<T, S, D>
where
    T: std::fmt::Display,
    S: TensorRef<T, D>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(&self, f)?;
        writeln!(f)?;
        write!(f, "Data Layout = {:?}", self.data_layout())
    }
}
