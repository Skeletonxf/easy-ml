/*!
 * Generic N dimensional [named tensors](http://nlp.seas.harvard.edu/NamedTensor).
 *
 * **These APIs are still in alpha, use at your own risk! Breaking changes still expected.**
 *
 * Tensors are generic over some type `T` and some usize `D`. If `T` is [Numeric](super::numeric)
 * then the tensor can be used in a mathematical way. `D` is the number of dimensions in the tensor
 * and a compile time constant. Each tensor also carries `D` dimension name and length pairs.
 */
use crate::linear_algebra;
use crate::numeric::{Numeric, NumericRef};
use crate::tensors::indexing::{
    IndexOrderIterator, IndexOrderReferenceIterator, IndexOrderReferenceMutIterator, ShapeIterator,
    TensorAccess,
};
use crate::tensors::views::{
    IndexRange, IndexRangeValidationError, TensorExpansion, TensorIndex, TensorMask, TensorMut,
    TensorRange, TensorRef, TensorRename, TensorView,
};

use std::error::Error;
use std::fmt;

pub mod dimensions;
mod display;
pub mod indexing;
pub mod operations;
pub mod views;

/**
 * Dimension names are represented as static string references.
 *
 * This allows you to use string literals to refer to named dimensions, for example you might want
 * to construct a tensor with a shape of
 * `[("batch", 1000), ("height", 100), ("width", 100), ("rgba", 4)]`.
 *
 * Alternatively you can define the strings once as constants and refer to your dimension
 * names by the constant identifiers.
 *
 * ```
 * const BATCH: &'static str = "batch";
 * const HEIGHT: &'static str = "height";
 * const WIDTH: &'static str = "width";
 * const RGBA: &'static str = "rgba";
 * ```
 *
 * Although `Dimension` is interchangable with `&'static str` as it is just a type alias, Easy ML
 * uses `Dimension` whenever dimension names are expected to distinguish the types from just
 * strings.
 */
pub type Dimension = &'static str;

/**
 * An error indicating failure to do something with a Tensor because the requested shape
 * is not valid.
 */
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidShapeError<const D: usize> {
    shape: [(Dimension, usize); D],
}

impl<const D: usize> InvalidShapeError<D> {
    /**
     * Checks if this shape is valid. This is mainly for internal library use but may also be
     * useful for unit testing.
     */
    pub fn is_valid(&self) -> bool {
        !crate::tensors::dimensions::has_duplicates(&self.shape)
            && !self.shape.iter().any(|d| d.1 == 0)
    }

    /**
     * Constructs an InvalidShapeError for assistance with unit testing. Note that you can
     * construct an InvalidShapeError that *is* a valid shape in this way.
     */
    pub fn new(shape: [(Dimension, usize); D]) -> InvalidShapeError<D> {
        InvalidShapeError { shape }
    }

    pub fn shape(&self) -> [(Dimension, usize); D] {
        self.shape
    }

    pub fn shape_ref(&self) -> &[(Dimension, usize); D] {
        &self.shape
    }
}

impl<const D: usize> fmt::Display for InvalidShapeError<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dimensions must all be at least length 1 with unique names: {:?}",
            self.shape
        )
    }
}

impl<const D: usize> Error for InvalidShapeError<D> {}

/**
 * An error indicating failure to do something with a Tensor because the dimension names that
 * were provided did not match with the dimension names that were valid.
 *
 * Typically this would be due to the same dimension name being provided multiple times, or a
 * dimension name being provided that is not present in the shape of the Tensor in use.
 */
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidDimensionsError<const D: usize, const P: usize> {
    valid: [Dimension; D],
    provided: [Dimension; P],
}

impl<const D: usize, const P: usize> InvalidDimensionsError<D, P> {
    /**
     * Checks if the provided dimensions have duplicate names. This is mainly for internal library
     * use but may also be useful for unit testing.
     */
    pub fn has_duplicates(&self) -> bool {
        crate::tensors::dimensions::has_duplicates_names(&self.provided)
    }

    // TODO: method to check provided is a subset of valid

    /**
     * Constructs an InvalidDimensions for assistance with unit testing.
     */
    pub fn new(provided: [Dimension; P], valid: [Dimension; D]) -> InvalidDimensionsError<D, P> {
        InvalidDimensionsError { valid, provided }
    }

    pub fn provided_names(&self) -> [Dimension; P] {
        self.provided
    }

    pub fn provided_names_ref(&self) -> &[Dimension; P] {
        &self.provided
    }

    pub fn valid_names(&self) -> [Dimension; D] {
        self.valid
    }

    pub fn valid_names_ref(&self) -> &[Dimension; D] {
        &self.valid
    }
}

impl<const D: usize, const P: usize> fmt::Display for InvalidDimensionsError<D, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if P > 0 {
            write!(
                f,
                "Dimensions names {:?} were incorrect, valid dimensions in this context are: {:?}",
                self.provided, self.valid
            )
        } else {
            write!(f, "Dimensions names {:?} were incorrect", self.provided)
        }
    }
}

impl<const D: usize, const P: usize> Error for InvalidDimensionsError<D, P> {}

#[test]
fn test_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<InvalidShapeError<2>>();
    assert_sync::<InvalidDimensionsError<2, 2>>();
}

#[test]
fn test_send() {
    fn assert_send<T: Send>() {}
    assert_send::<InvalidShapeError<2>>();
    assert_send::<InvalidDimensionsError<2, 2>>();
}

/**
 * A [named tensor](http://nlp.seas.harvard.edu/NamedTensor).
 *
 * TODO: Summary
 *
 * See also:
 * - [indexing](indexing)
 */
#[derive(Debug)]
pub struct Tensor<T, const D: usize> {
    data: Vec<T>,
    dimensions: [(Dimension, usize); D],
    strides: [usize; D],
}

impl<T, const D: usize> Tensor<T, D> {
    /**
     * Creates a Tensor with a particular number of dimensions and lengths in each dimension.
     *
     * The product of the dimension lengths corresponds to the number of elements the Tensor
     * will store. Elements are stored in what would be row major order for a Matrix.
     * Each step in memory through the N dimensions corresponds to incrementing the rightmost
     * index, hence a shape of `[("row", 5), ("column", 5)]` would mean the first 6 elements
     * passed in the Vec would be for (0,0), (0,1), (0,2), (0,3), (0,4), (1,0) and so on to (4,4)
     * for the 25th and final element.
     *
     * # Panics
     *
     * - If the number of provided elements does not match the product of the dimension lengths.
     * - If a dimension name is not unique
     * - If any dimension has 0 elements
     *
     * Note that an empty list for dimensions is valid, and constructs a 0 dimensional tensor with
     * a single element (since the product of an empty list is 1).
     */
    #[track_caller]
    pub fn from(dimensions: [(Dimension, usize); D], data: Vec<T>) -> Self {
        validate_dimensions(&dimensions, data.len());
        let strides = compute_strides(&dimensions);
        Tensor {
            data,
            dimensions,
            strides,
        }
    }

    /**
     * The shape of this tensor. Since Tensors are named Tensors, their shape is not just a
     * list of length along each dimension, but instead a list of pairs of names and lengths.
     *
     * See also
     * - [dimensions](crate::tensors::dimensions)
     * - [indexing](crate::tensors::indexing)
     */
    pub fn shape(&self) -> [(Dimension, usize); D] {
        self.dimensions
    }

    // Unverified constructor for interal use when we know the dimensions/data/strides are
    // unchanged and don't need reverification
    pub(crate) fn direct_from(
        data: Vec<T>,
        dimensions: [(Dimension, usize); D],
        strides: [usize; D],
    ) -> Self {
        Tensor {
            data,
            dimensions,
            strides,
        }
    }
}

// TODO: Merge with InvalidShapeError type
// Panics if the dimensions are invalid for any reason with the appropriate error message.
#[track_caller]
#[inline]
fn validate_dimensions<const D: usize>(dimensions: &[(Dimension, usize); D], data_len: usize) {
    let elements = crate::tensors::dimensions::elements(dimensions);
    if data_len != elements {
        panic!(
            "Product of dimension lengths must match size of data. {} != {}",
            elements, data_len
        );
    }
    if crate::tensors::dimensions::has_duplicates(dimensions) {
        panic!("Dimension names must all be unique: {:?}", &dimensions);
    }
    if dimensions.iter().any(|d| d.1 == 0) {
        panic!("No dimension can have 0 elements: {:?}", &dimensions);
    }
}

impl<T> Tensor<T, 0> {
    /**
     * Creates a 0 dimensional tensor from some scalar
     */
    pub fn from_scalar(value: T) -> Tensor<T, 0> {
        Tensor {
            data: vec![value],
            dimensions: [],
            strides: [],
        }
    }

    /**
     * Returns the sole element of the 0 dimensional tensor.
     */
    pub fn into_scalar(self) -> T {
        self.data.into_iter().next().unwrap()
    }
}

impl<T> Tensor<T, 0>
where
    T: Clone,
{
    /**
     * Returns a copy of the sole element in the 0 dimensional tensor.
     */
    pub fn scalar(&self) -> T {
        self.data.get(0).unwrap().clone()
    }
}

impl<T> From<T> for Tensor<T, 0> {
    fn from(scalar: T) -> Tensor<T, 0> {
        Tensor::from_scalar(scalar)
    }
}
// TODO: See if we can find a way to write the reverse Tensor<T, 0> -> T conversion using From or Into (doesn't seem like we can?)

// # Safety
//
// We promise to never implement interior mutability for Tensor.
/**
 * A Tatrix implements TensorRef.
 */
unsafe impl<T, const D: usize> TensorRef<T, D> for Tensor<T, D> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions)?;
        self.data.get(i)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        Tensor::shape(self)
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        // The point of get_reference_unchecked is no bounds checking, and therefore
        // it does not make any sense to just use `unwrap` here. The trait documents that
        // it's undefind behaviour to call this method with an out of bounds index, so we
        // can assume the None case will never happen.
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions).unwrap_unchecked();
        self.data.get_unchecked(i)
    }
}

// # Safety
//
// We promise to never implement interior mutability for Tensor.
unsafe impl<T, const D: usize> TensorMut<T, D> for Tensor<T, D> {
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions)?;
        self.data.get_mut(i)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        // The point of get_reference_unchecked_mut is no bounds checking, and therefore
        // it does not make any sense to just use `unwrap` here. The trait documents that
        // it's undefind behaviour to call this method with an out of bounds index, so we
        // can assume the None case will never happen.
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions).unwrap_unchecked();
        self.data.get_unchecked_mut(i)
    }
}

/**
 * Any tensor of a Displayable type implements Display
 */
impl<T: std::fmt::Display, const D: usize> std::fmt::Display for Tensor<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(self, f)
    }
}

fn compute_strides<const D: usize>(dimensions: &[(Dimension, usize); D]) -> [usize; D] {
    let mut strides = [0; D];
    for d in 0..D {
        strides[d] = dimensions.iter().skip(d + 1).map(|d| d.1).product();
    }
    strides
}

// returns the 1 dimensional index to use to get the requested index into some tensor
#[inline]
fn get_index_direct<const D: usize>(
    // indexes to use
    indexes: &[usize; D],
    // strides for indexing into the tensor
    strides: &[usize; D],
    // dimensions of the tensor to index into
    dimensions: &[(Dimension, usize); D],
) -> Option<usize> {
    let mut index = 0;
    for d in 0..D {
        let n = indexes[d];
        if n >= dimensions[d].1 {
            return None;
        }
        index += n * strides[d];
    }
    Some(index)
}

impl<T, const D: usize> Tensor<T, D> {
    pub fn view(&self) -> TensorView<T, &Tensor<T, D>, D> {
        TensorView::from(self)
    }

    pub fn view_mut(&mut self) -> TensorView<T, &mut Tensor<T, D>, D> {
        TensorView::from(self)
    }

    pub fn view_owned(self) -> TensorView<T, Tensor<T, D>, D> {
        TensorView::from(self)
    }

    /**
     * Returns a TensorAccess which can be indexed in the order of the supplied dimensions
     * to read values from this tensor.
     *
     * # Panics
     *
     * If the set of dimensions supplied do not match the set of dimensions in this tensor's shape.
     */
    #[track_caller]
    pub fn get(&self, dimensions: [Dimension; D]) -> TensorAccess<T, &Tensor<T, D>, D> {
        TensorAccess::from(self, dimensions)
    }

    /**
     * Returns a TensorAccess which can be indexed in the order of the supplied dimensions
     * to read or write values from this tensor.
     *
     * # Panics
     *
     * If the set of dimensions supplied do not match the set of dimensions in this tensor's shape.
     */
    #[track_caller]
    pub fn get_mut(&mut self, dimensions: [Dimension; D]) -> TensorAccess<T, &mut Tensor<T, D>, D> {
        TensorAccess::from(self, dimensions)
    }

    /**
     * Returns a TensorAccess which can be indexed in the order of the supplied dimensions
     * to read or write values from this tensor.
     *
     * # Panics
     *
     * If the set of dimensions supplied do not match the set of dimensions in this tensor's shape.
     */
    #[track_caller]
    pub fn get_owned(self, dimensions: [Dimension; D]) -> TensorAccess<T, Tensor<T, D>, D> {
        TensorAccess::from(self, dimensions)
    }

    /**
     * Creates a TensorAccess which will index into the dimensions this Tensor was created with
     * in the same order as they were provided. See [TensorAccess::from_source_order].
     */
    pub fn source_order(&self) -> TensorAccess<T, &Tensor<T, D>, D> {
        TensorAccess::from_source_order(self)
    }

    /**
     * Creates a TensorAccess which will index into the dimensions this Tensor was
     * created with in the same order as they were provided. The TensorAccess mutably borrows
     * the Tensor, and can therefore mutate it. See [TensorAccess::from_source_order].
     */
    pub fn source_order_mut(&mut self) -> TensorAccess<T, &mut Tensor<T, D>, D> {
        TensorAccess::from_source_order(self)
    }

    /**
     * Creates a TensorAccess which will index into the dimensions this Tensor was
     * created with in the same order as they were provided. The TensorAccess takes ownership
     * of the Tensor, and can therefore mutate it. See [TensorAccess::from_source_order].
     */
    pub fn source_order_owned(self) -> TensorAccess<T, Tensor<T, D>, D> {
        TensorAccess::from_source_order(self)
    }

    /**
     * Returns an iterator over references to the data in this Tensor.
     */
    pub fn index_order_reference_iter(&self) -> IndexOrderReferenceIterator<T, Tensor<T, D>, D> {
        IndexOrderReferenceIterator::from(self)
    }

    /**
     * Returns an iterator over mutable references to the data in this Tensor.
     */
    pub fn index_order_reference_mut_iter(
        &mut self,
    ) -> IndexOrderReferenceMutIterator<T, Tensor<T, D>, D> {
        IndexOrderReferenceMutIterator::from(self)
    }

    // Non public index order reference iterator since we don't want to expose our implementation
    // details to public API since then we could never change them.
    pub(crate) fn direct_index_order_reference_iter(&self) -> std::slice::Iter<T> {
        self.data.iter()
    }

    /**
     * Renames the dimension names of the tensor without changing the lengths of the dimensions
     * in the tensor or moving any data around.
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * let mut tensor = Tensor::from([("x", 2), ("y", 3)], vec![1, 2, 3, 4, 5, 6]);
     * tensor.rename(["y", "z"]);
     * assert_eq!([("y", 2), ("z", 3)], tensor.shape());
     * ```
     *
     * # Panics
     *
     * - If a dimension name is not unique
     */
    #[track_caller]
    pub fn rename(&mut self, dimensions: [Dimension; D]) {
        if crate::tensors::dimensions::has_duplicates_names(&dimensions) {
            panic!("Dimension names must all be unique: {:?}", &dimensions);
        }
        for d in 0..D {
            self.dimensions[d].0 = dimensions[d];
        }
    }

    /**
     * Returns a TensorView with the dimension names of the shape renamed to the provided
     * dimensions. The data of this tensor and the dimension lengths and order remain unchanged.
     *
     * This is a shorthand for constructing the TensorView from this Tensor.
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * use easy_ml::tensors::views::{TensorView, TensorRename};
     * let abc = Tensor::from([("a", 3), ("b", 3), ("c", 3)], (0..27).collect());
     * let xyz = abc.rename_view(["x", "y", "z"]);
     * let also_xyz = TensorView::from(TensorRename::from(&abc, ["x", "y", "z"]));
     * assert_eq!(xyz, also_xyz);
     * assert_eq!(xyz, Tensor::from([("x", 3), ("y", 3), ("z", 3)], (0..27).collect()));
     * ```
     *
     * # Panics
     *
     * If a dimension name is not unique
     */
    #[track_caller]
    pub fn rename_view(
        &self,
        dimensions: [Dimension; D],
    ) -> TensorView<T, TensorRename<T, &Tensor<T, D>, D>, D> {
        TensorView::from(TensorRename::from(self, dimensions))
    }

    /**
     * Changes the shape of the tensor without changing the number of dimensions or moving any
     * data around.
     *
     * # Panics
     *
     * - If the number of provided elements in the new shape does not match the product of the
     * dimension lengths in the existing tensor's shape.
     * - If a dimension name is not unique
     * - If any dimension has 0 elements
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * let mut tensor = Tensor::from([("width", 2), ("height", 2)], vec![
     *     1, 2,
     *     3, 4
     * ]);
     * tensor.reshape_mut([("batch", 1), ("image", 4)]);
     * assert_eq!(tensor, Tensor::from([("batch", 1), ("image", 4)], vec![ 1, 2, 3, 4 ]));
     * ```
     */
    #[track_caller]
    pub fn reshape_mut(&mut self, dimensions: [(Dimension, usize); D]) {
        validate_dimensions(&dimensions, self.data.len());
        let strides = compute_strides(&dimensions);
        self.dimensions = dimensions;
        self.strides = strides;
    }

    /**
     * Consumes the tensor and changes the shape of the tensor without moving any
     * data around. The new Tensor may also have a different number of dimensions.
     *
     * # Panics
     *
     * - If the number of provided elements in the new shape does not match the product of the
     * dimension lengths in the existing tensor's shape.
     * - If a dimension name is not unique
     * - If any dimension has 0 elements
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * let tensor = Tensor::from([("width", 2), ("height", 2)], vec![
     *     1, 2,
     *     3, 4
     * ]);
     * let flattened = tensor.reshape_owned([("image", 4)]);
     * assert_eq!(flattened, Tensor::from([("image", 4)], vec![ 1, 2, 3, 4 ]));
     * ```
     */
    // TODO: View version
    #[track_caller]
    pub fn reshape_owned<const D2: usize>(
        self,
        dimensions: [(Dimension, usize); D2],
    ) -> Tensor<T, D2> {
        Tensor::from(dimensions, self.data)
    }

    /**
     * Returns a TensorView with a range taken in P dimensions, hiding the values **outside** the
     * range from view. Error cases are documented on [TensorRange](TensorRange).
     *
     * This is a shorthand for constructing the TensorView from this Tensor.
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * use easy_ml::tensors::views::{TensorView, TensorRange, IndexRange};
     * # use easy_ml::tensors::views::IndexRangeValidationError;
     * # fn main() -> Result<(), IndexRangeValidationError<3, 2>> {
     * let samples = Tensor::from([("batch", 5), ("x", 7), ("y", 7)], (0..(5 * 7 * 7)).collect());
     * let cropped = samples.range([("x", IndexRange::new(1, 5)), ("y", IndexRange::new(1, 5))])?;
     * let also_cropped = TensorView::from(
     *     TensorRange::from(&samples, [("x", 1..6), ("y", 1..6)])?
     * );
     * assert_eq!(cropped, also_cropped);
     * assert_eq!(
     *     cropped.select([("batch", 0)]),
     *     Tensor::from([("x", 5), ("y", 5)], vec![
     *          8,  9, 10, 11, 12,
     *         15, 16, 17, 18, 19,
     *         22, 23, 24, 25, 26,
     *         29, 30, 31, 32, 33,
     *         36, 37, 38, 39, 40
     *     ])
     * );
     * # Ok(())
     * # }
     * ```
     */
    pub fn range<R, const P: usize>(
        &self,
        ranges: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorRange<T, &Tensor<T, D>, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorRange::from(self, ranges).map(|range| TensorView::from(range))
    }

    /**
     * Returns a TensorView with a range taken in P dimensions, hiding the values **outside** the
     * range from view. Error cases are documented on [TensorRange](TensorRange). The TensorRange
     * mutably borrows this Tensor, and can therefore mutate it
     *
     * This is a shorthand for constructing the TensorView from this Tensor.
     */
    pub fn range_mut<R, const P: usize>(
        &mut self,
        ranges: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorRange<T, &mut Tensor<T, D>, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorRange::from(self, ranges).map(|range| TensorView::from(range))
    }

    /**
     * Returns a TensorView with a range taken in P dimensions, hiding the values **outside** the
     * range from view. Error cases are documented on [TensorRange](TensorRange). The TensorRange
     * takes ownership of this Tensor, and can therefore mutate it
     *
     * This is a shorthand for constructing the TensorView from this Tensor.
     */
    pub fn range_owned<R, const P: usize>(
        self,
        ranges: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorRange<T, Tensor<T, D>, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorRange::from(self, ranges).map(|range| TensorView::from(range))
    }

    /**
     * Returns a TensorView with a mask taken in P dimensions, hiding the values **inside** the
     * range from view. Error cases are documented on [TensorMask](TensorMask).
     *
     * This is a shorthand for constructing the TensorView from this Tensor.
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * use easy_ml::tensors::views::{TensorView, TensorMask, IndexRange};
     * # use easy_ml::tensors::views::IndexRangeValidationError;
     * # fn main() -> Result<(), IndexRangeValidationError<3, 2>> {
     * let samples = Tensor::from([("batch", 5), ("x", 7), ("y", 7)], (0..(5 * 7 * 7)).collect());
     * let corners = samples.mask([("x", IndexRange::new(3, 2)), ("y", IndexRange::new(3, 2))])?;
     * let also_corners = TensorView::from(
     *     TensorMask::from(&samples, [("x", 3..5), ("y", 3..5)])?
     * );
     * assert_eq!(corners, also_corners);
     * assert_eq!(
     *     corners.select([("batch", 0)]),
     *     Tensor::from([("x", 5), ("y", 5)], vec![
     *          0,  1,  2,    5, 6,
     *          7,  8,  9,   12, 13,
     *         14, 15, 16,   19, 20,
     *
     *         35, 36, 37,   40, 41,
     *         42, 43, 44,   47, 48
     *     ])
     * );
     * # Ok(())
     * # }
     * ```
     */
    pub fn mask<R, const P: usize>(
        &self,
        masks: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorMask<T, &Tensor<T, D>, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorMask::from(self, masks).map(|mask| TensorView::from(mask))
    }

    /**
     * Returns a TensorView with a mask taken in P dimensions, hiding the values **inside** the
     * range from view. Error cases are documented on [TensorMask](TensorMask). The TensorMask
     * mutably borrows this Tensor, and can therefore mutate it
     *
     * This is a shorthand for constructing the TensorView from this Tensor.
     */
    pub fn mask_mut<R, const P: usize>(
        &mut self,
        masks: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorMask<T, &mut Tensor<T, D>, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorMask::from(self, masks).map(|mask| TensorView::from(mask))
    }

    /**
     * Returns a TensorView with a mask taken in P dimensions, hiding the values **inside** the
     * range from view. Error cases are documented on [TensorMask](TensorMask). The TensorMask
     * takes ownership of this Tensor, and can therefore mutate it
     *
     * This is a shorthand for constructing the TensorView from this Tensor.
     */
    pub fn mask_owned<R, const P: usize>(
        self,
        masks: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorMask<T, Tensor<T, D>, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorMask::from(self, masks).map(|mask| TensorView::from(mask))
    }

    /**
     * Creates and returns a new tensor with all value pairs of two tensors with the same shape
     * mapped by a function. The value pairs are not copied for you, if you're using `Copy` types
     * or need to clone the values anyway, you can use
     * [`Tensor::elementwise`](Tensor::elementwise) instead.
     *
     * # Generics
     *
     * This method can be called with any right hand side that can be converted to a TensorView,
     * which includes `Tensor`, `&Tensor`, `&mut Tensor` as well as references to a `TensorView`.
     *
     * # Panics
     *
     * If the two tensors have different shapes.
     */
    pub fn elementwise_reference<S, I, M>(&self, rhs: I, mapping_function: M) -> Tensor<T, D>
    where
        I: Into<TensorView<T, S, D>>,
        S: TensorRef<T, D>,
        M: Fn(&T, &T) -> T,
    {
        self.elementwise_reference_less_generic(rhs.into(), mapping_function)
    }

    /**
     * Creates and returns a new tensor with all value pairs of two tensors with the same shape
     * mapped by a function. The mapping function also receives each index corresponding to the
     * value pairs. The value pairs are not copied for you, if you're using `Copy` types
     * or need to clone the values anyway, you can use
     * [`Tensor::elementwise_with_index`](Tensor::elementwise_with_index) instead.
     *
     * # Generics
     *
     * This method can be called with any right hand side that can be converted to a TensorView,
     * which includes `Tensor`, `&Tensor`, `&mut Tensor` as well as references to a `TensorView`.
     *
     * # Panics
     *
     * If the two tensors have different shapes.
     */
    pub fn elementwise_reference_with_index<S, I, M>(
        &self,
        rhs: I,
        mapping_function: M,
    ) -> Tensor<T, D>
    where
        I: Into<TensorView<T, S, D>>,
        S: TensorRef<T, D>,
        M: Fn([usize; D], &T, &T) -> T,
    {
        self.elementwise_reference_less_generic_with_index(rhs.into(), mapping_function)
    }

    fn elementwise_reference_less_generic<S, M>(
        &self,
        rhs: TensorView<T, S, D>,
        mapping_function: M,
    ) -> Tensor<T, D>
    where
        S: TensorRef<T, D>,
        M: Fn(&T, &T) -> T,
    {
        let left_shape = self.shape();
        let right_shape = rhs.shape();
        if left_shape != right_shape {
            panic!(
                "Dimensions of left and right tensors are not the same: (left: {:?}, right: {:?})",
                left_shape, right_shape
            );
        }
        let mapped = self
            .direct_index_order_reference_iter()
            .zip(rhs.index_order_reference_iter())
            .map(|(x, y)| mapping_function(x, y))
            .collect();
        // We're not changing the shape of the Tensor, so don't need to revalidate
        Tensor::direct_from(mapped, self.dimensions, self.strides)
    }

    fn elementwise_reference_less_generic_with_index<S, M>(
        &self,
        rhs: TensorView<T, S, D>,
        mapping_function: M,
    ) -> Tensor<T, D>
    where
        S: TensorRef<T, D>,
        M: Fn([usize; D], &T, &T) -> T,
    {
        let left_shape = self.shape();
        let right_shape = rhs.shape();
        if left_shape != right_shape {
            panic!(
                "Dimensions of left and right tensors are not the same: (left: {:?}, right: {:?})",
                left_shape, right_shape
            );
        }
        // we just checked both shapes were the same, so we don't need to propagate indexes
        // for both tensors because they'll be identical
        let mapped = self
            .direct_index_order_reference_iter()
            .zip(rhs.index_order_reference_iter().with_index())
            .map(|(x, (i, y))| mapping_function(i, x, y))
            .collect();
        // We're not changing the shape of the Tensor, so don't need to revalidate
        Tensor::direct_from(mapped, self.dimensions, self.strides)
    }
}

impl<T, const D: usize> Tensor<T, D>
where
    T: Clone,
{
    /**
     * Creates a tensor with a particular number of dimensions and length in each dimension
     * with all elements initialised to the provided value.
     *
     * # Panics
     *
     * - If a dimension name is not unique
     * - If any dimension has 0 elements
     */
    #[track_caller]
    pub fn empty(dimensions: [(Dimension, usize); D], value: T) -> Self {
        let elements = crate::tensors::dimensions::elements(&dimensions);
        Tensor::from(dimensions, vec![value; elements])
    }

    // TODO: View version (just a wrapper for TensorRename<TensorAccess>)
    /**
     * Returns a new Tensor which has the same data as this tensor, but with the order of data
     * changed. The order of the dimension names is unchanged, although their lengths may swap.
     *
     * For example, with a `[("x", x), ("y", y)]` tensor you could call
     * `transpose(["y", "x"])` which would return a new tensor with a shape of
     * `[("x", y), ("y", x)]` where every (x,y) of its data corresponds to (y,x) in the original.
     *
     * This method need not shift *all* the dimensions though, you could also swap the width
     * and height of images in a tensor with a shape of
     * `[("batch", b), ("h", h), ("w", w), ("c", c)]` via `transpose(["batch", "w", "h", "c"])`
     * which would return a new tensor where all the images have been swapped over the diagonal.
     *
     * See also: [TensorAccess](TensorAccess), [reorder](Tensor::reorder)
     *
     * # Panics
     *
     * If the set of dimensions in the tensor does not match the set of dimensions provided. The
     * order need not match (and if the order does match, this function is just an expensive
     * clone).
     */
    #[track_caller]
    pub fn transpose(&self, dimensions: [Dimension; D]) -> Tensor<T, D> {
        let shape = self.dimensions;
        let mut reordered = self.reorder(dimensions);
        // Transposition is essentially reordering, but we retain the dimension name ordering
        // of the original order, this means we may swap dimension lengths, but the dimensions
        // will not change order.
        for d in 0..D {
            reordered.dimensions[d].0 = shape[d].0;
        }
        reordered
    }

    /**
     * Modifies this tensor to have the same data as before, but with the order of data changed.
     * The order of the dimension names is unchanged, although their lengths may swap.
     *
     * For example, with a `[("x", x), ("y", y)]` tensor you could call
     * `transpose_mut(["y", "x"])` which would edit the tensor, updating its shape to
     * `[("x", y), ("y", x)]`, so every (x,y) of its data corresponds to (y,x) before the
     * transposition.
     *
     * The order swapping will try to be in place, but this is currently only supported for
     * square tensors with 2 dimensions. Other types of tensors will not be transposed in place.
     *
     * # Panics
     *
     * If the set of dimensions in the tensor does not match the set of dimensions provided. The
     * order need not match (and if the order does match, this function is just an expensive
     * clone).
     */
    #[track_caller]
    pub fn transpose_mut(&mut self, dimensions: [Dimension; D]) {
        let shape = self.dimensions;
        self.reorder_mut(dimensions);
        // Transposition is essentially reordering, but we retain the dimension name ordering
        // we had before, this means we may swap dimension lengths, but the dimensions
        // will not change order.
        for d in 0..D {
            self.dimensions[d].0 = shape[d].0;
        }
    }

    /**
     * Returns a new Tensor which has the same data as this tensor, but with the order of the
     * dimensions and corresponding order of data changed.
     *
     * For example, with a `[("x", x), ("y", y)]` tensor you could call
     * `reorder(["y", "x"])` which would return a new tensor with a shape of
     * `[("y", y), ("x", x)]` where every (y,x) of its data corresponds to (x,y) in the original.
     *
     * This method need not shift *all* the dimensions though, you could also swap the width
     * and height of images in a tensor with a shape of
     * `[("batch", b), ("h", h), ("w", w), ("c", c)]` via `reorder(["batch", "w", "h", "c"])`
     * which would return a new tensor where every (b,w,h,c) of its data corresponds to (b,h,w,c)
     * in the original.
     *
     * See also: [TensorAccess](TensorAccess), [transpose](Tensor::transpose)
     *
     * # Panics
     *
     * If the set of dimensions in the tensor does not match the set of dimensions provided. The
     * order need not match (and if the order does match, this function is just an expensive
     * clone).
     */
    #[track_caller]
    pub fn reorder(&self, dimensions: [Dimension; D]) -> Tensor<T, D> {
        let reorderd = match TensorAccess::try_from(&self, dimensions) {
            Ok(reordered) => reordered,
            Err(_error) => panic!(
                "Dimension names provided {:?} must be the same set of dimension names in the tensor: {:?}",
                dimensions,
                &self.dimensions,
            ),
        };
        let reorderd_shape = reorderd.shape();
        Tensor::from(
            reorderd_shape,
            reorderd.index_order_iter().collect(),
        )
    }

    /**
     * Modifies this tensor to have the same data as before, but with the order of the
     * dimensions and corresponding order of data changed.
     *
     * For example, with a `[("x", x), ("y", y)]` tensor you could call
     * `reorder_mut(["y", "x"])` which would edit the tensor, updating its shape to
     * `[("y", y), ("x", x)]`, so every (y,x) of its data corresponds to (x,y) before the
     * transposition.
     *
     * The order swapping will try to be in place, but this is currently only supported for
     * square tensors with 2 dimensions. Other types of tensors will not be reordered in place.
     *
     * # Panics
     *
     * If the set of dimensions in the tensor does not match the set of dimensions provided. The
     * order need not match (and if the order does match, this function is just an expensive
     * clone).
     */
    #[track_caller]
    pub fn reorder_mut(&mut self, dimensions: [Dimension; D]) {
        use crate::tensors::dimensions::{
            dimension_mapping, dimension_mapping_shape, map_dimensions,
        };
        if D == 2 && crate::tensors::dimensions::is_square(&self.dimensions) {
            let dimension_mapping = match dimension_mapping(&self.dimensions, &dimensions) {
                Some(dimension_mapping) => dimension_mapping,
                None => panic!(
                    "Dimension names provided {:?} must be the same set of dimension names in the tensor: {:?}",
                    dimensions,
                    &self.dimensions,
                ),
            };

            let shape = dimension_mapping_shape(&self.dimensions, &dimension_mapping);
            let shape_iterator = ShapeIterator::from(shape);

            for index in shape_iterator {
                let i = index[0];
                let j = index[1];
                if j >= i {
                    let mapped_index = map_dimensions(&dimension_mapping, &index);
                    // Swap elements from the upper triangle (using index order of the actual tensor's
                    // shape)
                    let temp = self.get_reference(index).unwrap().clone();
                    // tensor[i,j] becomes tensor[mapping(i,j)]
                    *self.get_reference_mut(index).unwrap() =
                        self.get_reference(mapped_index).unwrap().clone();
                    // tensor[mapping(i,j)] becomes tensor[i,j]
                    *self.get_reference_mut(mapped_index).unwrap() = temp;
                    // If the mapping is a noop we've assigned i,j to i,j
                    // If the mapping is i,j -> j,i we've assigned i,j to j,i and j,i to i,j
                }
            }

            // now update our shape and strides to match
            self.dimensions = shape;
            self.strides = compute_strides(&shape);
        } else {
            // fallback to allocating a new reordered tensor
            let reordered = self.reorder(dimensions);
            self.data = reordered.data;
            self.dimensions = reordered.dimensions;
            self.strides = reordered.strides;
        }
    }

    /**
     * Returns an iterator over copes of the data in this Tensor.
     */
    pub fn index_order_iter(&self) -> IndexOrderIterator<T, Tensor<T, D>, D> {
        IndexOrderIterator::from(self)
    }

    /**
     * Creates and returns a new tensor with all values from the original with the
     * function applied to each. This can be used to change the type of the tensor
     * such as creating a mask:
     * ```
     * use easy_ml::tensors::Tensor;
     * let x = Tensor::from([("a", 2), ("b", 2)], vec![
     *    0.0, 1.2,
     *    5.8, 6.9
     * ]);
     * let y = x.map(|element| element > 2.0);
     * let result = Tensor::from([("a", 2), ("b", 2)], vec![
     *    false, false,
     *    true, true
     * ]);
     * assert_eq!(&y, &result);
     * ```
     */
    pub fn map<U>(&self, mapping_function: impl Fn(T) -> U) -> Tensor<U, D> {
        let mapped = self
            .data
            .iter()
            .map(|x| mapping_function(x.clone()))
            .collect();
        // We're not changing the shape of the Tensor, so don't need to revalidate
        Tensor::direct_from(mapped, self.dimensions, self.strides)
    }

    /**
     * Creates and returns a new tensor with all values from the original and
     * the index of each value mapped by a function.
     */
    pub fn map_with_index<U>(&self, mapping_function: impl Fn([usize; D], T) -> U) -> Tensor<U, D> {
        self.source_order().map_with_index(mapping_function)
    }

    /**
     * Applies a function to all values in the tensor, modifying
     * the tensor in place.
     */
    pub fn map_mut(&mut self, mapping_function: impl Fn(T) -> T) {
        for value in self.data.iter_mut() {
            *value = mapping_function(value.clone());
        }
    }

    /**
     * Applies a function to all values and each value's index in the tensor, modifying
     * the tensor in place.
     */
    pub fn map_mut_with_index(&mut self, mapping_function: impl Fn([usize; D], T) -> T) {
        self.source_order_mut().map_mut_with_index(mapping_function);
    }

    /**
     * Creates and returns a new tensor with all value pairs of two tensors with the same shape
     * mapped by a function.
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * let lhs = Tensor::from([("a", 4)], vec![1, 2, 3, 4]);
     * let rhs = Tensor::from([("a", 4)], vec![0, 1, 2, 3]);
     * let multiplied = lhs.elementwise(&rhs, |l, r| l * r);
     * assert_eq!(
     *     multiplied,
     *     Tensor::from([("a", 4)], vec![0, 2, 6, 12])
     * );
     * ```
     *
     * # Generics
     *
     * This method can be called with any right hand side that can be converted to a TensorView,
     * which includes `Tensor`, `&Tensor`, `&mut Tensor` as well as references to a `TensorView`.
     *
     * # Panics
     *
     * If the two tensors have different shapes.
     */
    pub fn elementwise<S, I, M>(&self, rhs: I, mapping_function: M) -> Tensor<T, D>
    where
        I: Into<TensorView<T, S, D>>,
        S: TensorRef<T, D>,
        M: Fn(T, T) -> T,
    {
        self.elementwise_reference_less_generic(rhs.into(), |lhs, rhs| {
            mapping_function(lhs.clone(), rhs.clone())
        })
    }

    /**
     * Creates and returns a new tensor with all value pairs of two tensors with the same shape
     * mapped by a function. The mapping function also receives each index corresponding to the
     * value pairs.
     *
     * # Generics
     *
     * This method can be called with any right hand side that can be converted to a TensorView,
     * which includes `Tensor`, `&Tensor`, `&mut Tensor` as well as references to a `TensorView`.
     *
     * # Panics
     *
     * If the two tensors have different shapes.
     */
    pub fn elementwise_with_index<S, I, M>(&self, rhs: I, mapping_function: M) -> Tensor<T, D>
    where
        I: Into<TensorView<T, S, D>>,
        S: TensorRef<T, D>,
        M: Fn([usize; D], T, T) -> T,
    {
        self.elementwise_reference_less_generic_with_index(rhs.into(), |i, lhs, rhs| {
            mapping_function(i, lhs.clone(), rhs.clone())
        })
    }
}

impl<T> Tensor<T, 1>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
{
    /**
     * Computes the scalar product of two equal length vectors. For two vectors `[a,b,c]` and
     * `[d,e,f]`, returns `a*d + b*e + c*f`. This is also known as the dot product.
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * let tensor = Tensor::from([("sequence", 5)], vec![3, 4, 5, 6, 7]);
     * assert_eq!(tensor.scalar_product(&tensor), 3*3 + 4*4 + 5*5 + 6*6 + 7*7);
     * ```
     *
     * # Generics
     *
     * This method can be called with any right hand side that can be converted to a TensorView,
     * which includes `Tensor`, `&Tensor`, `&mut Tensor` as well as references to a `TensorView`.
     *
     * # Panics
     *
     * If the two vectors are not of equal length or their dimension names do not match.
     */
    // Would like this impl block to be in operations.rs too but then it would show first in the
    // Tensor docs which isn't ideal
    pub fn scalar_product<S, I>(&self, rhs: I) -> T
    where
        I: Into<TensorView<T, S, 1>>,
        S: TensorRef<T, 1>,
    {
        self.scalar_product_less_generic(rhs.into())
    }
}

impl<T> Tensor<T, 2>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
{
    /**
     * Returns the determinant of this square matrix, or None if the matrix
     * does not have a determinant. See [`linear_algebra`](super::linear_algebra::determinant_tensor())
     */
    pub fn determinant(&self) -> Option<T> {
        linear_algebra::determinant_tensor::<T, _, _>(self)
    }

    /**
     * Computes the inverse of a matrix provided that it exists. To have an inverse a
     * matrix must be square (same number of rows and columns) and it must also have a
     * non zero determinant. See [`linear_algebra`](super::linear_algebra::inverse_tensor())
     */
    pub fn inverse(&self) -> Option<Tensor<T, 2>> {
        linear_algebra::inverse_tensor::<T, _, _>(self)
    }

    /**
     * Computes the covariance matrix for this feature matrix along the specified feature
     * dimension in this matrix. See [`linear_algebra`](crate::linear_algebra::covariance()).
     */
    pub fn covariance(&self, feature_dimension: Dimension) -> Tensor<T, 2> {
        linear_algebra::covariance::<T, _, _>(self, feature_dimension)
    }
}

macro_rules! tensor_select_impl {
    (impl Tensor $d:literal 1) => {
        impl<T> Tensor<T, $d> {
            /**
             * Selects the provided dimension name and index pairs in this Tensor, returning a
             * TensorView which has fewer dimensions than this Tensor, with the removed dimensions
             * always indexed as the provided values.
             *
             * This is a shorthand for manually constructing the TensorView and
             * [TensorIndex](TensorIndex)
             *
             * Note: due to limitations in Rust's const generics support, this method is only
             * implemented for `provided_indexes` of length 1 and `D` from 1 to 6. You can fall
             * back to manual construction to create `TensorIndex`es with multiple provided
             * indexes if you need to reduce dimensionality by more than 1 dimension at a time.
             */
            #[track_caller]
            pub fn select(
                &self,
                provided_indexes: [(Dimension, usize); 1],
            ) -> TensorView<T, TensorIndex<T, &Tensor<T, $d>, $d, 1>, { $d - 1 }> {
                TensorView::from(TensorIndex::from(self, provided_indexes))
            }

            /**
             * Selects the provided dimension name and index pairs in this Tensor, returning a
             * TensorView which has fewer dimensions than this Tensor, with the removed dimensions
             * always indexed as the provided values. The TensorIndex mutably borrows this
             * Tensor, and can therefore mutate it
             *
             * See [select](Tensor::select)
             */
            #[track_caller]
            pub fn select_mut(
                &mut self,
                provided_indexes: [(Dimension, usize); 1],
            ) -> TensorView<T, TensorIndex<T, &mut Tensor<T, $d>, $d, 1>, { $d - 1 }> {
                TensorView::from(TensorIndex::from(self, provided_indexes))
            }

            /**
             * Selects the provided dimension name and index pairs in this Tensor, returning a
             * TensorView which has fewer dimensions than this Tensor, with the removed dimensions
             * always indexed as the provided values. The TensorIndex takes ownership of this
             * Tensor, and can therefore mutate it
             *
             * See [select](Tensor::select)
             */
            #[track_caller]
            pub fn select_owned(
                self,
                provided_indexes: [(Dimension, usize); 1],
            ) -> TensorView<T, TensorIndex<T, Tensor<T, $d>, $d, 1>, { $d - 1 }> {
                TensorView::from(TensorIndex::from(self, provided_indexes))
            }
        }
    };
}

tensor_select_impl!(impl Tensor 6 1);
tensor_select_impl!(impl Tensor 5 1);
tensor_select_impl!(impl Tensor 4 1);
tensor_select_impl!(impl Tensor 3 1);
tensor_select_impl!(impl Tensor 2 1);
tensor_select_impl!(impl Tensor 1 1);

macro_rules! tensor_expand_impl {
    (impl Tensor $d:literal 1) => {
        impl<T> Tensor<T, $d> {
            /**
             * Expands the dimensionality of this tensor by adding dimensions of length 1 at
             * a particular position within the shape, returning a TensorView which has more
             * dimensions than this Tensor.
             *
             * This is a shorthand for manually constructing the TensorView and
             * [TensorExpansion](TensorExpansion)
             *
             * Note: due to limitations in Rust's const generics support, this method is only
             * implemented for `extra_dimension_names` of length 1 and `D` from 0 to 5. You can
             * fall back to manual construction to create `TensorExpansion`s with multiple provided
             * indexes if you need to increase dimensionality by more than 1 dimension at a time.
             */
            #[track_caller]
            pub fn expand(
                &self,
                extra_dimension_names: [(usize, Dimension); 1],
            ) -> TensorView<T, TensorExpansion<T, &Tensor<T, $d>, $d, 1>, { $d + 1 }> {
                TensorView::from(TensorExpansion::from(self, extra_dimension_names))
            }

            /**
             * Expands the dimensionality of this tensor by adding dimensions of length 1 at
             * a particular position within the shape, returning a TensorView which has more
             * dimensions than this Tensor. The TensorIndex mutably borrows this
             * Tensor, and can therefore mutate it
             *
             * See [expand](Tensor::expand)
             */
            #[track_caller]
            pub fn expand_mut(
                &mut self,
                extra_dimension_names: [(usize, Dimension); 1],
            ) -> TensorView<T, TensorExpansion<T, &mut Tensor<T, $d>, $d, 1>, { $d + 1 }> {
                TensorView::from(TensorExpansion::from(self, extra_dimension_names))
            }

            /**
             * Expands the dimensionality of this tensor by adding dimensions of length 1 at
             * a particular position within the shape, returning a TensorView which has more
             * dimensions than this Tensor. The TensorIndex takes ownership of this
             * Tensor, and can therefore mutate it
             *
             * See [expand](Tensor::expand)
             */
            #[track_caller]
            pub fn expand_owned(
                self,
                extra_dimension_names: [(usize, Dimension); 1],
            ) -> TensorView<T, TensorExpansion<T, Tensor<T, $d>, $d, 1>, { $d + 1 }> {
                TensorView::from(TensorExpansion::from(self, extra_dimension_names))
            }
        }
    };
}

tensor_expand_impl!(impl Tensor 0 1);
tensor_expand_impl!(impl Tensor 1 1);
tensor_expand_impl!(impl Tensor 2 1);
tensor_expand_impl!(impl Tensor 3 1);
tensor_expand_impl!(impl Tensor 4 1);
tensor_expand_impl!(impl Tensor 5 1);
