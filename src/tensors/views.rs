use std::marker::PhantomData;

use crate::tensors::indexing::TensorAccess;
use crate::tensors::{Dimension, Tensor};

mod indexes;
pub mod traits;

pub use indexes::*;

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
* implementing type is required to uphold several invariants.
*
* 1 - Any valid index as described in Indexing will yield a safe reference when calling
* `get_reference_unchecked` and `get_reference_unchecked_mut`.
*
* 2 - The view shape that defines which indexes are valid may not be changed by a shared reference
* to the TensorRef implementation. ie, the tensor may not be resized while a mutable reference is
* held to it, except by that reference.
*
* Essentially, interior mutability causes problems, since code looping through the range of valid
* indexes in a TensorRef needs to be able to rely on that range of valid indexes not changing.
* This is trivially the case by default since a [Tensor](Tensor) does not have any form of
* interior mutability, and therefore an iterator holding a shared reference to a Tensor prevents
* that tensor being resized. However, a type *wrongly* implementing TensorRef could introduce
* interior mutability by putting the Tensor in an `Arc<Mutex<>>` which would allow another thread
* to resize a tensor while an iterator was looping through previously valid indexes on a different
* thread. This is the same contract as
* [`NoInteriorMutability`](crate::matrices::views::NoInteriorMutability) used in in
* the matrix APIs.
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

#[derive(Debug)]
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
    pub fn get(&self, dimensions: [Dimension; D]) -> TensorAccess<T, &S, D> {
        TensorAccess::from(&self.source, dimensions)
    }

    #[track_caller]
    pub fn get_mut(&mut self, dimensions: [Dimension; D]) -> TensorAccess<T, &mut S, D> {
        TensorAccess::from(&mut self.source, dimensions)
    }

    #[track_caller]
    pub fn get_owned(self, dimensions: [Dimension; D]) -> TensorAccess<T, S, D> {
        TensorAccess::from(self.source, dimensions)
    }

    pub fn shape(&self) -> [(Dimension, usize); D] {
        self.source.view_shape()
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
     * Creates a TensorAccess which will index into the dimensions this Tensor was
     * created with in the same order as they were provided. The TensorAccess mutably borrows
     * the Tensor, and can therefore mutate it. See [TensorAccess::from_source_order].
     */

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
     * of the Tensor, and can therefore mutate it. The TensorAccess mutably borrows
     * the source, and can therefore mutate it if it implements TensorMut.
     * See [TensorAccess::from_source_order].
     */
    pub fn source_order_owned(self) -> TensorAccess<T, S, D> {
        TensorAccess::from_source_order(self.source)
    }
}

impl<T, S, const D: usize> TensorView<T, S, D> where S: TensorMut<T, D> {}

impl<T, S, const D: usize> TensorView<T, S, D>
where
    T: Clone,
    S: TensorRef<T, D>,
{
    /**
     * Returns a new Tensor which has the same data as this tensor, but with the order of the
     * dimensions and corresponding order of data changed.
     *
     * For example, with a `[("row", x), ("column", y)]` tensor you could call
     * `transpose(["y", "x"])` which would return a new tensor where every (y,x) of its data
     * corresponds to (x,y) in the original.
     *
     * This method need not shift *all* the dimensions though, you could also swap the width
     * and height of images in a tensor with a shape of
     * `[("batch", b), ("h", h), ("w", w), ("c", c)]` via `transpose(["batch", "w", "h", "c"])`
     * which would return a new tensor where every (b,w,h,c) of its data corresponds to (b,h,w,c)
     * in the original.
     *
     * # Panics
     *
     * If the set of dimensions in the tensor does not match the set of dimensions provided. The
     * order need not match (and if the order does match, this function is just an expensive
     * clone).
     */
    #[track_caller]
    pub fn transpose(&self, dimensions: [Dimension; D]) -> Tensor<T, D> {
        // TODO: Handle error case, propagate as Dimension names to transpose to must be the same set of dimension names in the tensor
        let transposed_order = TensorAccess::from(&self.source, dimensions);
        let transposed_shape = transposed_order.shape();
        Tensor::from(
            transposed_shape,
            transposed_order.index_order_iter().collect(),
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
        self.source_order().map(mapping_function)
    }

    /**
     * Creates and returns a new tensor with all values from the original and
     * the index of each value mapped by a function.
     */
    pub fn map_with_index<U>(&self, mapping_function: impl Fn([usize; D], T) -> U) -> Tensor<U, D> {
        self.source_order().map_with_index(mapping_function)
    }
}

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
        self.source_order_mut().map_mut(mapping_function)
    }

    /**
     * Applies a function to all values and each value's index in the tensor view, modifying
     * the tensor view in place.
     */
    pub fn map_mut_with_index(&mut self, mapping_function: impl Fn([usize; D], T) -> T) {
        self.source_order_mut().map_mut_with_index(mapping_function);
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
             * This is a shorthand for manually constructing the TensorView and TensorIndex
             *
             * ```
             * use easy_ml::tensors::Tensor;
             * use easy_ml::tensors::views::{TensorView, TensorIndex};
             * let vector = TensorView::from(Tensor::from([("a", 2)], vec![ 16, 8 ]));
             * let scalar = vector.select([("a", 0)]);
             * let also_scalar = TensorView::from(TensorIndex::from(vector.source_ref(), [("a", 0)]));
             * assert_eq!(scalar.get([]).get([]), also_scalar.get([]).get([]));
             * assert_eq!(scalar.get([]).get([]), 16);
             * ```
             *
             * Note: due to limitations in Rust's const generics support, this method is only
             * implemented for `provided_indexes` of length 1 and `D` from 1 to 6. You can fall
             * back to manual construction to create `TensorIndex`es with multiple provided
             * indexes if you need to reduce dimensionality by more than 1 dimension at a time.
             */
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
            pub fn select_mut(
                &mut self,
                provided_indexes: [(Dimension, usize); 1],
            ) -> TensorView<T, TensorIndex<T, &mut S, $d, 1>, { $d - 1 }> {
                TensorView::from(TensorIndex::from(&mut self.source, provided_indexes))
            }

            /**
             * Selects the provided dimension name and index pairs in this TensorView, returning a
             * TensorView which has fewer dimensions than this Tensor, with the removed dimensions
             * always indexed as the provided values. The TensorIndex takes ownership ofthis
             * Tensor, and can therefore mutate it
             *
             * See [select](TensorView::select)
             */
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


/**
 * Any tensor view of a Displayable type implements Display
 */
impl<T: std::fmt::Display, S> std::fmt::Display for TensorView<T, S, 0>
where
    T: std::fmt::Display,
    S: TensorRef<T, 0>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(&self.source, f)
    }
}

/**
 * Any tensor view of a Displayable type implements Display
 */
impl<T: std::fmt::Display, S> std::fmt::Display for TensorView<T, S, 1>
where
    T: std::fmt::Display,
    S: TensorRef<T, 1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(&self.source, f)
    }
}

/**
 * Any tensor view of a Displayable type implements Display
 */
impl<T: std::fmt::Display, S> std::fmt::Display for TensorView<T, S, 2>
where
    T: std::fmt::Display,
    S: TensorRef<T, 2>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(&self.source, f)
    }
}

/**
 * Any tensor view of a Displayable type implements Display
 */
impl<T: std::fmt::Display, S> std::fmt::Display for TensorView<T, S, 3>
where
    T: std::fmt::Display,
    S: TensorRef<T, 3>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(&self.source, f)
    }
}
