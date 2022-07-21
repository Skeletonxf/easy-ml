/*!
 * Generic views into a tensor.
 *
 * The concept of a view into a tensor is built from the low level [TensorRef](TensorRef) and
 * [TensorMut](TensorMut) traits which define having read and read/write access to Tensor data
 * respectively, and the high level API implemented on the [TensorView](TensorView) struct.
 *
 * Since a Tensor is itself a TensorRef, the APIs for the traits are a little verbose to
 * avoid name clashes with methods defined on the Tensor and TensorView types. You should
 * typically use TensorRef and TensorMut implementations via the TensorView struct which provides
 * an API closely resembling Tensor.
 */

use std::marker::PhantomData;

use crate::linear_algebra;
use crate::numeric::{Numeric, NumericRef};
use crate::tensors::indexing::{
    IndexOrderIterator, IndexOrderReferenceIterator, IndexOrderReferenceMutIterator, TensorAccess,
};
use crate::tensors::{Dimension, Tensor};

mod indexes;
mod ranges;
mod renamed;
pub mod traits;

pub use indexes::*;
pub use ranges::*;
pub use renamed::*;

/**
* A shared/immutable reference to a tensor (or a portion of it) of some type and number of
* dimensions.
*
* # Indexing
*
* A TensorRef has a shape of type `[(Dimension, usize); D]`. This defines the valid indexes along
* each dimension name and length pair from 0 inclusive to the length exclusive. If the shape was
* `[("r", 2), ("c", 2)]` the indexes used would be `[0,0]`, `[0,1]`, `[1,0]` and `[1,1]`. Although
* the dimension name in each pair is used for many high level APIs, for TensorRef the order of
* dimensions is used, and the indexes (`[usize; D]`) these trait methods are called with must
* be in the same order as the shape.
*
* # Safety
*
* In order to support returning references without bounds checking in a useful way, the
* implementing type is required to uphold several invariants that cannot be checked by
* the compiler.
*
* 1 - Any valid index as described in Indexing will yield a safe reference when calling
* `get_reference_unchecked` and `get_reference_unchecked_mut`.
*
* 2 - The view shape that defines which indexes are valid may not be changed by a shared reference
* to the TensorRef implementation. ie, the tensor may not be resized while a mutable reference is
* held to it, except by that reference.
*
* 3 - All dimension names in the view_shape must be unique.
*
* 4 - All dimension lengths in the view_shape must be non zero.
*
* Essentially, interior mutability causes problems, since code looping through the range of valid
* indexes in a TensorRef needs to be able to rely on that range of valid indexes not changing.
* This is trivially the case by default since a [Tensor](Tensor) does not have any form of
* interior mutability, and therefore an iterator holding a shared reference to a Tensor prevents
* that tensor being resized. However, a type *wrongly* implementing TensorRef could introduce
* interior mutability by putting the Tensor in an `Arc<Mutex<>>` which would allow another thread
* to resize a tensor while an iterator was looping through previously valid indexes on a different
* thread. This is the same contract as
* [`NoInteriorMutability`](crate::matrices::views::NoInteriorMutability) used in the matrix APIs.
*
* Note that it is okay to be able to resize any TensorRef implementation if that always requires
* an exclusive reference to the TensorRef/Tensor, since the exclusivity prevents the above
* scenario.
*/
pub unsafe trait TensorRef<T, const D: usize> {
    /**
     * Gets a reference to the value at the index if the index is in range. Otherwise returns None.
     */
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T>;

    /**
     * The shape this tensor has. See [dimensions](crate::tensors::dimensions) for an overview.
     * The product of the lengths in the pairs define how many elements are in the tensor
     * (or the portion of it that is visible).
     */
    fn view_shape(&self) -> [(Dimension, usize); D];

    /**
     * Gets a reference to the value at the index without doing any bounds checking. For a safe
     * alternative see [get_reference](TensorRef::get_reference).
     *
     * # Safety
     *
     * Calling this method with an out-of-bounds index is *[undefined behavior]* even if the
     * resulting reference is not used. Valid indexes are defined as in [TensorRef].
     *
     * [undefined behavior]: <https://doc.rust-lang.org/reference/behavior-considered-undefined.html>
     * [TensorRef]: TensorRef
     */
    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T;
}

/**
 * A unique/mutable reference to a tensor (or a portion of it) of some type.
 *
 * # Safety
 *
 * See [TensorRef](TensorRef).
 */
pub unsafe trait TensorMut<T, const D: usize>: TensorRef<T, D> {
    /**
     * Gets a mutable reference to the value at the index, if the index is in range. Otherwise
     * returns None.
     */
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T>;

    /**
     * Gets a mutable reference to the value at the index without doing any bounds checking.
     * For a safe alternative see [get_reference_mut](TensorMut::get_reference_mut).
     *
     * # Safety
     *
     * Calling this method with an out-of-bounds index is *[undefined behavior]* even if the
     * resulting reference is not used. Valid indexes are defined as in [TensorRef].
     *
     * [undefined behavior]: <https://doc.rust-lang.org/reference/behavior-considered-undefined.html>
     * [TensorRef]: TensorRef
     */
    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T;
}

/**
 * A view into some or all of a tensor.
 *
 * A TensorView has a similar relationship to a [`Tensor`](Tensor) as a
 * `&str` has to a `String`, or an array slice to an array. A TensorView cannot resize
 * its source, and may span only a portion of the source Tensor in each dimension.
 *
 * However a TensorView is generic not only over the type of the data in the Tensor,
 * but also over the way the Tensor is 'sliced' and the two are orthogonal to each other.
 *
 * TensorView closely mirrors the API of Tensor.
 * Methods that create a new tensor do not return a TensorView, they return a Tensor.
 */
pub struct TensorView<T, S, const D: usize> {
    source: S,
    _type: PhantomData<T>,
}

/**
 * TensorView methods which require only read access via a [TensorRef](TensorRef) source.
 */
impl<T, S, const D: usize> TensorView<T, S, D>
where
    S: TensorRef<T, D>,
{
    /**
     * Creates a TensorView from a source of some type.
     *
     * The lifetime of the source determines the lifetime of the TensorView created. If the
     * TensorView is created from a reference to a Tensor, then the TensorView cannot live
     * longer than the Tensor referenced.
     */
    pub fn from(source: S) -> TensorView<T, S, D> {
        TensorView {
            source,
            _type: PhantomData,
        }
    }

    /**
     * Consumes the tensor view, yielding the source it was created from.
     */
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Gives a reference to the tensor view's source.
     */
    pub fn source_ref(&self) -> &S {
        &self.source
    }

    /**
     * Gives a mutable reference to the tensor view's source.
     */
    pub fn source_ref_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /**
     * The shape of this tensor view. Since Tensors are named Tensors, their shape is not just a
     * list of length along each dimension, but instead a list of pairs of names and lengths.
     *
     * Note that a TensorView may have a shape which is different than the Tensor it is providing
     * access to the data of. The TensorView might be [masking dimensions](TensorIndex) or
     * elements from the shape, or exposing [false ones](TensorExpansion).
     *
     * See also
     * - [dimensions](crate::tensors::dimensions)
     * - [indexing](crate::tensors::indexing)
     */
    pub fn shape(&self) -> [(Dimension, usize); D] {
        self.source.view_shape()
    }

    /**
     * Returns a TensorAccess which can be indexed in the order of the supplied dimensions
     * to read values from this tensor view.
     *
     * # Panics
     *
     * If the set of dimensions supplied do not match the set of dimensions in this tensor's shape.
     */
    #[track_caller]
    pub fn get(&self, dimensions: [Dimension; D]) -> TensorAccess<T, &S, D> {
        TensorAccess::from(&self.source, dimensions)
    }

    /**
     * Returns a TensorAccess which can be indexed in the order of the supplied dimensions
     * to read or write values from this tensor view.
     *
     * # Panics
     *
     * If the set of dimensions supplied do not match the set of dimensions in this tensor's shape.
     */
    #[track_caller]
    pub fn get_mut(&mut self, dimensions: [Dimension; D]) -> TensorAccess<T, &mut S, D> {
        TensorAccess::from(&mut self.source, dimensions)
    }

    /**
     * Returns a TensorAccess which can be indexed in the order of the supplied dimensions
     * to read or write values from this tensor view.
     *
     * # Panics
     *
     * If the set of dimensions supplied do not match the set of dimensions in this tensor's shape.
     */
    #[track_caller]
    pub fn get_owned(self, dimensions: [Dimension; D]) -> TensorAccess<T, S, D> {
        TensorAccess::from(self.source, dimensions)
    }

    /**
     * Creates a TensorAccess which will index into the dimensions of the source this TensorView
     * was created with in the same order as they were declared.
     * See [TensorAccess::from_source_order].
     */
    pub fn source_order(&self) -> TensorAccess<T, &S, D> {
        TensorAccess::from_source_order(&self.source)
    }

    /**
     * Creates a TensorAccess which will index into the dimensions of the source this TensorView
     * was created with in the same order as they were declared. The TensorAccess mutably borrows
     * the source, and can therefore mutate it if it implements TensorMut.
     * See [TensorAccess::from_source_order].
     */
    pub fn source_order_mut(&mut self) -> TensorAccess<T, &mut S, D> {
        TensorAccess::from_source_order(&mut self.source)
    }

    /**
     * Creates a TensorAccess which will index into the dimensions this Tensor was
     * created with in the same order as they were provided. The TensorAccess takes ownership
     * of the Tensor, and can therefore mutate it. The TensorAccess takes ownership of
     * the source, and can therefore mutate it if it implements TensorMut.
     * See [TensorAccess::from_source_order].
     */
    pub fn source_order_owned(self) -> TensorAccess<T, S, D> {
        TensorAccess::from_source_order(self.source)
    }

    /**
     * Returns an iterator over references to the data in this TensorView.
     */
    pub fn index_order_reference_iter(&self) -> IndexOrderReferenceIterator<T, S, D> {
        IndexOrderReferenceIterator::from(&self.source)
    }

    /**
     * Returns a TensorView with the dimension names of the shape renamed to the provided
     * dimensions. The data of this tensor and the dimension lengths and order remain unchanged.
     *
     * This is a shorthand for constructing the TensorView from this TensorView. See
     * [`Tensor::rename_view`](Tensor::rename_view).
     *
     * # Panics
     *
     * If a dimension name is not unique
     */
    #[track_caller]
    pub fn rename_view(
        &self,
        dimensions: [Dimension; D],
    ) -> TensorView<T, TensorRename<T, &S, D>, D> {
        TensorView::from(TensorRename::from(&self.source, dimensions))
    }

    /**
     * Returns a TensorView with a range taken in P dimensions, hiding the values **outside** the
     * range from view. Error cases are documented on [TensorRange](TensorRange).
     *
     * This is a shorthand for constructing the TensorView from this TensorView. See
     * [`Tensor::range`](Tensor::range).
     */
    pub fn range<R, const P: usize>(
        &self,
        ranges: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorRange<T, &S, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorRange::from(&self.source, ranges).map(|range| TensorView::from(range))
    }

    /**
     * Returns a TensorView with a range taken in P dimensions, hiding the values **outside** the
     * range from view. Error cases are documented on [TensorRange](TensorRange). The TensorRange
     * mutably borrows the source, and can therefore mutate it if it implements TensorMut.
     *
     * This is a shorthand for constructing the TensorView from this TensorView. See
     * [`Tensor::range`](Tensor::range).
     */
    pub fn range_mut<R, const P: usize>(
        &mut self,
        ranges: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorRange<T, &mut S, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorRange::from(&mut self.source, ranges).map(|range| TensorView::from(range))
    }

    /**
     * Returns a TensorView with a range taken in P dimensions, hiding the values **outside** the
     * range from view. Error cases are documented on [TensorRange](TensorRange). The TensorRange
     * takes ownership of the source, and can therefore mutate it if it implements TensorMut.
     *
     * This is a shorthand for constructing the TensorView from this TensorView. See
     * [`Tensor::range`](Tensor::range).
     */
    pub fn range_owned<R, const P: usize>(
        self,
        ranges: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorRange<T, S, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorRange::from(self.source, ranges).map(|range| TensorView::from(range))
    }

    /**
     * Returns a TensorView with a mask taken in P dimensions, hiding the values **inside** the
     * range from view. Error cases are documented on [TensorMask](TensorMask).
     *
     * This is a shorthand for constructing the TensorView from this TensorView. See
     * [`Tensor::mask`](Tensor::mask).
     */
    pub fn mask<R, const P: usize>(
        &self,
        masks: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorMask<T, &S, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorMask::from(&self.source, masks).map(|mask| TensorView::from(mask))
    }

    /**
     * Returns a TensorView with a mask taken in P dimensions, hiding the values **inside** the
     * range from view. Error cases are documented on [TensorMask](TensorMask). The TensorMask
     * mutably borrows the source, and can therefore mutate it if it implements TensorMut.
     *
     * This is a shorthand for constructing the TensorView from this TensorView. See
     * [`Tensor::mask`](Tensor::mask).
     */
    pub fn mask_mut<R, const P: usize>(
        &mut self,
        masks: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorMask<T, &mut S, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorMask::from(&mut self.source, masks).map(|mask| TensorView::from(mask))
    }

    /**
     * Returns a TensorView with a mask taken in P dimensions, hiding the values **inside** the
     * range from view. Error cases are documented on [TensorMask](TensorMask). The TensorMask
     * takes ownership of the source, and can therefore mutate it if it implements TensorMut.
     *
     * This is a shorthand for constructing the TensorView from this TensorView. See
     * [`Tensor::mask`](Tensor::mask).
     */
    pub fn mask_owned<R, const P: usize>(
        self,
        masks: [(Dimension, R); P],
    ) -> Result<TensorView<T, TensorMask<T, S, D>, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        TensorMask::from(self.source, masks).map(|mask| TensorView::from(mask))
    }

    /**
     * Creates and returns a new tensor with all value pairs of two tensors with the same shape
     * mapped by a function. The value pairs are not copied for you, if you're using `Copy` types
     * or need to clone the values anyway, you can use
     * [`TensorView::elementwise`](TensorView::elementwise) instead.
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
    pub fn elementwise_reference<S2, I, M>(&self, rhs: I, mapping_function: M) -> Tensor<T, D>
    where
        I: Into<TensorView<T, S2, D>>,
        S2: TensorRef<T, D>,
        M: Fn(&T, &T) -> T,
    {
        self.elementwise_reference_less_generic(rhs.into(), mapping_function)
    }

    /**
     * Creates and returns a new tensor with all value pairs of two tensors with the same shape
     * mapped by a function. The mapping function also receives each index corresponding to the
     * value pairs. The value pairs are not copied for you, if you're using `Copy` types
     * or need to clone the values anyway, you can use
     * [`TensorView::elementwise_with_index`](TensorView::elementwise_with_index) instead.
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
    pub fn elementwise_reference_with_index<S2, I, M>(
        &self,
        rhs: I,
        mapping_function: M,
    ) -> Tensor<T, D>
    where
        I: Into<TensorView<T, S2, D>>,
        S2: TensorRef<T, D>,
        M: Fn([usize; D], &T, &T) -> T,
    {
        self.elementwise_reference_less_generic_with_index(rhs.into(), mapping_function)
    }

    fn elementwise_reference_less_generic<S2, M>(
        &self,
        rhs: TensorView<T, S2, D>,
        mapping_function: M,
    ) -> Tensor<T, D>
    where
        S2: TensorRef<T, D>,
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
            .index_order_reference_iter()
            .zip(rhs.index_order_reference_iter())
            .map(|(x, y)| mapping_function(x, y))
            .collect();
        Tensor::from(left_shape, mapped)
    }

    fn elementwise_reference_less_generic_with_index<S2, M>(
        &self,
        rhs: TensorView<T, S2, D>,
        mapping_function: M,
    ) -> Tensor<T, D>
    where
        S2: TensorRef<T, D>,
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
            .index_order_reference_iter()
            .with_index()
            .zip(rhs.index_order_reference_iter())
            .map(|((i, x), y)| mapping_function(i, x, y))
            .collect();
        Tensor::from(left_shape, mapped)
    }
}

impl<T, S, const D: usize> TensorView<T, S, D>
where
    S: TensorMut<T, D>,
{
    /**
     * Returns an iterator over mutable references to the data in this TensorView.
     */
    pub fn index_order_reference_mut_iter(&mut self) -> IndexOrderReferenceMutIterator<T, S, D> {
        IndexOrderReferenceMutIterator::from(&mut self.source)
    }
}

/**
 * TensorView methods which require only read access via a [TensorRef](TensorRef) source
 * and a clonable type.
 */
impl<T, S, const D: usize> TensorView<T, S, D>
where
    T: Clone,
    S: TensorRef<T, D>,
{
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
     * See also: [TensorAccess](TensorAccess), [reorder](TensorView::reorder)
     *
     * # Panics
     *
     * If the set of dimensions in the tensor does not match the set of dimensions provided. The
     * order need not match (and if the order does match, this function is just an expensive
     * clone).
     */
    #[track_caller]
    pub fn transpose(&self, dimensions: [Dimension; D]) -> Tensor<T, D> {
        let shape = self.shape();
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
     * See also: [TensorAccess](TensorAccess), [transpose](TensorView::transpose)
     *
     * # Panics
     *
     * If the set of dimensions in the tensor does not match the set of dimensions provided. The
     * order need not match (and if the order does match, this function is just an expensive
     * clone).
     */
    #[track_caller]
    pub fn reorder(&self, dimensions: [Dimension; D]) -> Tensor<T, D> {
        let reorderd = match TensorAccess::try_from(&self.source, dimensions) {
            Ok(reordered) => reordered,
            Err(_error) => panic!(
                "Dimension names provided {:?} must be the same set of dimension names in the tensor: {:?}",
                dimensions,
                self.shape(),
            ),
        };
        let reorderd_shape = reorderd.shape();
        Tensor::from(
            reorderd_shape,
            reorderd.index_order_iter().collect(),
        )
    }

    /**
     * Creates and returns a new tensor with all values from the original with the
     * function applied to each. This can be used to change the type of the tensor
     * such as creating a mask:
     * ```
     * use easy_ml::tensors::Tensor;
     * use easy_ml::tensors::views::TensorView;
     * let x = TensorView::from(Tensor::from([("a", 2), ("b", 2)], vec![
     *    0.0, 1.2,
     *    5.8, 6.9
     * ]));
     * let y = x.map(|element| element > 2.0);
     * let result = Tensor::from([("a", 2), ("b", 2)], vec![
     *    false, false,
     *    true, true
     * ]);
     * assert_eq!(&y, &result);
     * ```
     */
    pub fn map<U>(&self, mapping_function: impl Fn(T) -> U) -> Tensor<U, D> {
        let mapped = self.index_order_iter().map(mapping_function).collect();
        Tensor::from(self.shape(), mapped)
    }

    /**
     * Creates and returns a new tensor with all values from the original and
     * the index of each value mapped by a function.
     */
    pub fn map_with_index<U>(&self, mapping_function: impl Fn([usize; D], T) -> U) -> Tensor<U, D> {
        let mapped = self
            .index_order_iter()
            .with_index()
            .map(|(i, x)| mapping_function(i, x))
            .collect();
        Tensor::from(self.shape(), mapped)
    }

    /**
     * Returns an iterator over copies of the data in this TensorView.
     */
    pub fn index_order_iter(&self) -> IndexOrderIterator<T, S, D> {
        IndexOrderIterator::from(&self.source)
    }

    /**
     * Creates and returns a new tensor with all value pairs of two tensors with the same shape
     * mapped by a function.
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * use easy_ml::tensors::views::TensorView;
     * let lhs = TensorView::from(Tensor::from([("a", 4)], vec![1, 2, 3, 4]));
     * let rhs = TensorView::from(Tensor::from([("a", 4)], vec![0, 1, 2, 3]));
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
    pub fn elementwise<S2, I, M>(&self, rhs: I, mapping_function: M) -> Tensor<T, D>
    where
        I: Into<TensorView<T, S2, D>>,
        S2: TensorRef<T, D>,
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
    pub fn elementwise_with_index<S2, I, M>(&self, rhs: I, mapping_function: M) -> Tensor<T, D>
    where
        I: Into<TensorView<T, S2, D>>,
        S2: TensorRef<T, D>,
        M: Fn([usize; D], T, T) -> T,
    {
        self.elementwise_reference_less_generic_with_index(rhs.into(), |i, lhs, rhs| {
            mapping_function(i, lhs.clone(), rhs.clone())
        })
    }
}

/**
 * TensorView methods which require only read access via a scalar [TensorRef](TensorRef) source
 * and a clonable type.
 */
impl<T, S> TensorView<T, S, 0>
where
    T: Clone,
    S: TensorRef<T, 0>,
{
    /**
     * Returns a copy of the sole element in the 0 dimensional tensor.
     */
    pub fn scalar(&self) -> T {
        self.source.get_reference([]).unwrap().clone()
    }
}

/**
 * TensorView methods which require mutable access via a [TensorMut](TensorMut) source.
 */
impl<T, S, const D: usize> TensorView<T, S, D>
where
    T: Clone,
    S: TensorMut<T, D>,
{
    /**
     * Applies a function to all values in the tensor view, modifying
     * the tensor in place.
     */
    pub fn map_mut(&mut self, mapping_function: impl Fn(T) -> T) {
        self.index_order_reference_mut_iter()
            .for_each(|x| *x = mapping_function(x.clone()));
    }

    /**
     * Applies a function to all values and each value's index in the tensor view, modifying
     * the tensor view in place.
     */
    pub fn map_mut_with_index(&mut self, mapping_function: impl Fn([usize; D], T) -> T) {
        self.index_order_reference_mut_iter()
            .with_index()
            .for_each(|(i, x)| *x = mapping_function(i, x.clone()));
    }
}

impl<T, S> TensorView<T, S, 1>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S: TensorRef<T, 1>,
{
    /**
     * Computes the scalar product of two equal length vectors. For two vectors `[a,b,c]` and
     * `[d,e,f]`, returns `a*d + b*e + c*f`. This is also known as the dot product.
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * use easy_ml::tensors::views::TensorView;
     * let tensor_view = TensorView::from(Tensor::from([("sequence", 5)], vec![3, 4, 5, 6, 7]));
     * assert_eq!(tensor_view.scalar_product(&tensor_view), 3*3 + 4*4 + 5*5 + 6*6 + 7*7);
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
    // TensorView docs which isn't ideal
    pub fn scalar_product<S2, I>(&self, rhs: I) -> T
    where
        I: Into<TensorView<T, S2, 1>>,
        S2: TensorRef<T, 1>,
    {
        self.scalar_product_less_generic(rhs.into())
    }
}

impl<T, S> TensorView<T, S, 2>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S: TensorRef<T, 2>,
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

macro_rules! tensor_view_select_impl {
    (impl TensorView $d:literal 1) => {
        impl<T, S> TensorView<T, S, $d>
        where
            S: TensorRef<T, $d>,
        {
            /**
             * Selects the provided dimension name and index pairs in this TensorView, returning a
             * TensorView which has fewer dimensions than this TensorView, with the removed dimensions
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
            ) -> TensorView<T, TensorIndex<T, &S, $d, 1>, { $d - 1 }> {
                TensorView::from(TensorIndex::from(&self.source, provided_indexes))
            }

            /**
             * Selects the provided dimension name and index pairs in this TensorView, returning a
             * TensorView which has fewer dimensions than this Tensor, with the removed dimensions
             * always indexed as the provided values. The TensorIndex mutably borrows this
             * Tensor, and can therefore mutate it
             *
             * See [select](TensorView::select)
             */
            #[track_caller]
            pub fn select_mut(
                &mut self,
                provided_indexes: [(Dimension, usize); 1],
            ) -> TensorView<T, TensorIndex<T, &mut S, $d, 1>, { $d - 1 }> {
                TensorView::from(TensorIndex::from(&mut self.source, provided_indexes))
            }

            /**
             * Selects the provided dimension name and index pairs in this TensorView, returning a
             * TensorView which has fewer dimensions than this Tensor, with the removed dimensions
             * always indexed as the provided values. The TensorIndex takes ownership of this
             * Tensor, and can therefore mutate it
             *
             * See [select](TensorView::select)
             */
            #[track_caller]
            pub fn select_owned(
                self,
                provided_indexes: [(Dimension, usize); 1],
            ) -> TensorView<T, TensorIndex<T, S, $d, 1>, { $d - 1 }> {
                TensorView::from(TensorIndex::from(self.source, provided_indexes))
            }
        }
    };
}

tensor_view_select_impl!(impl TensorView 6 1);
tensor_view_select_impl!(impl TensorView 5 1);
tensor_view_select_impl!(impl TensorView 4 1);
tensor_view_select_impl!(impl TensorView 3 1);
tensor_view_select_impl!(impl TensorView 2 1);
tensor_view_select_impl!(impl TensorView 1 1);

macro_rules! tensor_view_expand_impl {
    (impl Tensor $d:literal 1) => {
        impl<T, S> TensorView<T, S, $d>
        where
            S: TensorRef<T, $d>,
        {
            /**
             * Expands the dimensionality of this tensor by adding dimensions of length 1 at
             * a particular position within the shape, returning a TensorView which has more
             * dimensions than this TensorView.
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
            ) -> TensorView<T, TensorExpansion<T, &S, $d, 1>, { $d + 1 }> {
                TensorView::from(TensorExpansion::from(&self.source, extra_dimension_names))
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
            ) -> TensorView<T, TensorExpansion<T, &mut S, $d, 1>, { $d + 1 }> {
                TensorView::from(TensorExpansion::from(
                    &mut self.source,
                    extra_dimension_names,
                ))
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
            ) -> TensorView<T, TensorExpansion<T, S, $d, 1>, { $d + 1 }> {
                TensorView::from(TensorExpansion::from(self.source, extra_dimension_names))
            }
        }
    };
}

tensor_view_expand_impl!(impl Tensor 0 1);
tensor_view_expand_impl!(impl Tensor 1 1);
tensor_view_expand_impl!(impl Tensor 2 1);
tensor_view_expand_impl!(impl Tensor 3 1);
tensor_view_expand_impl!(impl Tensor 4 1);
tensor_view_expand_impl!(impl Tensor 5 1);

/**
 * Debug implementations for TensorView additionally show the visible data and visible dimensions
 * reported by the source as fields. This is in addition to recursive debug content of the actual
 * source.
 */
impl<T, S, const D: usize> std::fmt::Debug for TensorView<T, S, D>
where
    T: std::fmt::Debug,
    S: std::fmt::Debug + TensorRef<T, D>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("TensorView")
            .field("visible", &DebugSourceVisible::from(&self.source))
            .field("dimensions", &self.source.view_shape())
            .field("source", &self.source)
            .finish()
    }
}

struct DebugSourceVisible<T, S, const D: usize> {
    source: S,
    _type: PhantomData<T>,
}

impl<T, S, const D: usize> DebugSourceVisible<T, S, D>
where
    T: std::fmt::Debug,
    S: std::fmt::Debug + TensorRef<T, D>,
{
    fn from(source: S) -> DebugSourceVisible<T, S, D> {
        DebugSourceVisible {
            source,
            _type: PhantomData,
        }
    }
}

impl<T, S, const D: usize> std::fmt::Debug for DebugSourceVisible<T, S, D>
where
    T: std::fmt::Debug,
    S: std::fmt::Debug + TensorRef<T, D>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list().entries(IndexOrderReferenceIterator::from(&self.source)).finish()
    }
}

#[test]
fn test_debug() {
    let x = Tensor::from([("rows", 3), ("columns", 4)], (0..12).collect());
    let view = TensorView::from(&x);
    let debugged = format!("{:?}\n{:?}", x, view);
    assert_eq!(
        debugged,
        r#"Tensor { data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dimensions: [("rows", 3), ("columns", 4)], strides: [4, 1] }
TensorView { visible: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dimensions: [("rows", 3), ("columns", 4)], source: Tensor { data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dimensions: [("rows", 3), ("columns", 4)], strides: [4, 1] } }"#
    )
}

#[test]
fn test_debug_clipped() {
    let x = Tensor::from([("rows", 2), ("columns", 3)], (0..6).collect());
    let view = TensorView::from(&x).range_owned([("columns", IndexRange::new(1, 2))]).unwrap();
    let debugged = format!("{:#?}\n{:#?}", x, view);
    println!("{:#?}\n{:#?}", x, view);
    assert_eq!(
        debugged,
        r#"Tensor {
    data: [
        0,
        1,
        2,
        3,
        4,
        5,
    ],
    dimensions: [
        (
            "rows",
            2,
        ),
        (
            "columns",
            3,
        ),
    ],
    strides: [
        3,
        1,
    ],
}
TensorView {
    visible: [
        1,
        2,
        4,
        5,
    ],
    dimensions: [
        (
            "rows",
            2,
        ),
        (
            "columns",
            2,
        ),
    ],
    source: TensorRange {
        source: Tensor {
            data: [
                0,
                1,
                2,
                3,
                4,
                5,
            ],
            dimensions: [
                (
                    "rows",
                    2,
                ),
                (
                    "columns",
                    3,
                ),
            ],
            strides: [
                3,
                1,
            ],
        },
        range: [
            IndexRange {
                start: 0,
                length: 2,
            },
            IndexRange {
                start: 1,
                length: 2,
            },
        ],
        _type: PhantomData,
    },
}"#
    )
}

/**
 * Any tensor view of a Displayable type implements Display
 */
impl<T: std::fmt::Display, S, const D: usize> std::fmt::Display for TensorView<T, S, D>
where
    T: std::fmt::Display,
    S: TensorRef<T, D>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(&self.source, f)
    }
}

/**
 * A Tensor can be converted to a TensorView of that Tensor.
 */
impl<T, const D: usize> From<Tensor<T, D>> for TensorView<T, Tensor<T, D>, D> {
    fn from(tensor: Tensor<T, D>) -> TensorView<T, Tensor<T, D>, D> {
        TensorView::from(tensor)
    }
}

/**
 * A reference to a Tensor can be converted to a TensorView of that referenced Tensor.
 */
impl<'a, T, const D: usize> From<&'a Tensor<T, D>> for TensorView<T, &'a Tensor<T, D>, D> {
    fn from(tensor: &Tensor<T, D>) -> TensorView<T, &Tensor<T, D>, D> {
        TensorView::from(tensor)
    }
}

/**
 * A mutable reference to a Tensor can be converted to a TensorView of that mutably referenced
 * Tensor.
 */
impl<'a, T, const D: usize> From<&'a mut Tensor<T, D>> for TensorView<T, &'a mut Tensor<T, D>, D> {
    fn from(tensor: &mut Tensor<T, D>) -> TensorView<T, &mut Tensor<T, D>, D> {
        TensorView::from(tensor)
    }
}

/**
 * A reference to a TensorView can be converted to an owned TensorView with a reference to the
 * source type of that first TensorView.
 */
impl<'a, T, S, const D: usize> From<&'a TensorView<T, S, D>> for TensorView<T, &'a S, D>
where
    S: TensorRef<T, D>,
{
    fn from(tensor_view: &TensorView<T, S, D>) -> TensorView<T, &S, D> {
        TensorView::from(tensor_view.source_ref())
    }
}

/**
 * A mutable reference to a TensorView can be converted to an owned TensorView with a mutable
 * reference to the source type of that first TensorView.
 */
impl<'a, T, S, const D: usize> From<&'a mut TensorView<T, S, D>> for TensorView<T, &'a mut S, D>
where
    S: TensorRef<T, D>,
{
    fn from(tensor_view: &mut TensorView<T, S, D>) -> TensorView<T, &mut S, D> {
        TensorView::from(tensor_view.source_ref_mut())
    }
}
