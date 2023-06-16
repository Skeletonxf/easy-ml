use crate::tensors::Dimension;
use crate::tensors::views::DataLayout;

mod traits;

pub use traits::*;

/**
 * A shared/immutable tensorlike container (or a portion of it) of some type and number of
 * dimensions.
 *
 * This is a generalisation of [TensorRef](TensorRef) and all the same invariants must be upheld
 * by implementations.
 *
 * Unlike TensorRef, implementations may return non reference types, they are not limited to
 * returning references to data held inside them.
 *
 * # Safety
 *
 * See [TensorRef](TensorRef).
 */
pub unsafe trait TensorlikeRef<'a, T, const D: usize, Ref = &'a T> {
    /**
     * Gets a value at the index if the index is in range. Otherwise returns None.
     */
    fn get_value(&'a self, indexes: [usize; D]) -> Option<Ref>;

    /**
     * The shape this container has. See [dimensions](crate::tensors::dimensions) for an overview.
     * The product of the lengths in the pairs define how many elements are in the tensor
     * (or the portion of it that is visible).
     */
    fn view_shape(&self) -> [(Dimension, usize); D];

    /**
     * Gets a value at the index without doing any bounds checking. For a safe
     * alternative see [get_value](TensorlikeRef::get_value).
     *
     * # Safety
     *
     * Calling this method with an out-of-bounds index is *[undefined behavior]* even if the
     * resulting value is not used. Valid indexes are defined as in [TensorRef].
     *
     * [undefined behavior]: <https://doc.rust-lang.org/reference/behavior-considered-undefined.html>
     * [TensorRef]: TensorRef
     */
    unsafe fn get_value_unchecked(&'a self, indexes: [usize; D]) -> Ref;

    /**
     * The way the data in this container is laid out in memory. In particular,
     * [`Linear`](DataLayout) has several requirements on what is returned that must be upheld
     * by implementations of this trait, which are documented on [TensorRef](TensorRef).
     * Implementations that don't return references to values laid out contiguously in memory
     * must **never** return `DataLayout::Linear`.
     */
    fn data_layout(&self) -> DataLayout<D>;
}

/**
 * A unique/mutable tensorlike container (or a portion of it) of some type.
 *
 * Unlike TensorMut, implementations may return non reference types, they are not limited to
 * returning references to data held inside them. However, it is expected that implementations
 * return types which are actually mutable in some way.
 *
 * # Safety
 *
 * See [TensorValue](TensorValue).
 */
pub unsafe trait TensorlikeMut<'a, T, const D: usize, Ref = &'a T, Mut = &'a mut T>: TensorlikeRef<'a, T, D, Ref> {
    /**
     * Gets a mutable version of the value at the index, if the index is in range. Otherwise
     * returns None.
     */
    fn get_value_mut(&'a mut self, indexes: [usize; D]) -> Option<Mut>;

    /**
     * Gets a mutable version of the value at the index without doing any bounds checking.
     * For a safe alternative see [get_value_mut](TensorlikeRef::get_value_mut).
     *
     * # Safety
     *
     * Calling this method with an out-of-bounds index is *[undefined behavior]* even if the
     * resulting value is not used. Valid indexes are defined as in [TensorRef].
     *
     * [undefined behavior]: <https://doc.rust-lang.org/reference/behavior-considered-undefined.html>
     * [TensorRef]: TensorRef
     */
    unsafe fn get_value_unchecked_mut(&'a mut self, indexes: [usize; D]) -> Mut;
}
