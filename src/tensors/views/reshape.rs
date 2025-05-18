use crate::tensors;
use crate::tensors::dimensions;
use crate::tensors::indexing::ShapeIterator;
use crate::tensors::views::{DataLayout, TensorMut, TensorRef};
use crate::tensors::Dimension;
use std::marker::PhantomData;

/**
 * A new shape to override indexing an existing tensor. The dimensionality and individual
 * dimension lengths can be changed, but the total number of elements in the new shape must
 * match the existing tensor's shape. Elements are still iterated in the same order as the
 * source tensor.
 *
 * If you just need to rename dimensions without changing them, see
 * [TensorRename](tensors::views::TensorRename)
 */
#[derive(Clone, Debug)]
pub struct TensorReshape<T, S, const D: usize> {
    source: S,
    shape: [(Dimension, usize); D],
    strides: [usize; D],
    _type: PhantomData<T>,
}

impl<T, S, const D: usize> TensorReshape<T, S, D>
where
    S: TensorRef<T, D>,
{
    /**
     * Creates a TensorReshape from a source and a new shape to override the
     * view_shape with. The new shape must correspond to the same number of total
     * elements, but it need not match on dimensionality or individual dimension lengths.
     *
     * If you just need to rename dimensions without changing them, see
     * [TensorRename](tensors::views::TensorRename)
     *
     * # Panics
     *
     * - If the new shape has a different number of elements to the existing
     * shape in the source
     * - If the new shape has duplicate dimension names
     */
    #[track_caller]
    pub fn from(source: S, shape: [(Dimension, usize); D]) -> TensorReshape<T, S, D> {
        if dimensions::has_duplicates(&shape) {
            panic!("Dimension names must all be unique: {:?}", &shape);
        }
        let existing_one_dimensional_length = dimensions::elements(&source.view_shape());
        let given_one_dimensional_length = dimensions::elements(&shape);
        if given_one_dimensional_length != existing_one_dimensional_length {
            panic!(
                "Number of elements required by provided shape {:?} are {:?} but number of elements in source are: {:?} due to shape of {:?}",
                &shape,
                &given_one_dimensional_length,
                &existing_one_dimensional_length,
                &source.view_shape()
            );
        }
        TensorReshape {
            source,
            shape,
            strides: tensors::compute_strides(&shape),
            _type: PhantomData,
        }
    }

    // TODO: Consider helper function here for when caller only wants to change dimension
    // lengths and is fine to use existing dimensionality and dimension names, which
    // removes a lot of potential errors.

    /**
     * Consumes the TensorReshape, yielding the source it was created from.
     */
    #[allow(dead_code)]
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Gives a reference to the TensorReshape's source (in which the data has its original shape).
     */
    // # Safety
    //
    // Giving out a mutable reference to our source could allow it to be changed out from under us
    // and make the number of elements in our shape invalid. However, since the source implements
    // TensorRef interior mutability is not allowed, so we can give out shared references without
    // breaking our own integrity.
    #[allow(dead_code)]
    pub fn source_ref(&self) -> &S {
        &self.source
    }
}

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// by only flattening the caller's index into a one dimensional one, and then unflattening
// it back into the source dimensionality and not introducing interior mutability, we implement
// TensorRef correctly as well.
/**
 * A TensorReshape implements TensorRef, with the data iterated in the same order as the
 * original source.
 */
unsafe impl<T, S, const D: usize> TensorRef<T, D> for TensorReshape<T, S, D>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        let one_dimensional_index =
            tensors::get_index_direct(&indexes, &self.strides, &self.shape)?;
        // TODO: Is there a more efficient way to do this?
        self.source.get_reference(
            ShapeIterator::from(self.source.view_shape()).nth(one_dimensional_index)?,
        )
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.shape
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        // It is the caller's responsibility to always call with indexes in range,
        // therefore out of bounds lookups created by get_index_direct_unchecked should never
        // happen.
        let one_dimensional_index = tensors::get_index_direct_unchecked(&indexes, &self.strides);
        self.source.get_reference_unchecked(
            ShapeIterator::from(self.source.view_shape())
                .nth(one_dimensional_index)
                .unwrap(),
        )
    }

    fn data_layout(&self) -> DataLayout<D> {
        // There might be some cases where assigning a new shape maintains a linear order
        // but it seems like a lot of effort to maintain a correct mapping from the original
        // linear order to the new one, given we can change even dimensionality in this mapping.
        DataLayout::Other
    }
}

/**
 * A TensorReshape implements TensorMut, with the data iterated in the same order as the
 * original source.
 */
unsafe impl<T, S, const D: usize> TensorMut<T, D> for TensorReshape<T, S, D>
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        let one_dimensional_index =
            tensors::get_index_direct(&indexes, &self.strides, &self.shape)?;
        // TODO: Is there a more efficient way to do this?
        self.source.get_reference_mut(
            ShapeIterator::from(self.source.view_shape()).nth(one_dimensional_index)?,
        )
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        // It is the caller's responsibility to always call with indexes in range,
        // therefore out of bounds lookups created by get_index_direct_unchecked should never
        // happen.
        let one_dimensional_index = tensors::get_index_direct_unchecked(&indexes, &self.strides);
        self.source.get_reference_unchecked_mut(
            ShapeIterator::from(self.source.view_shape())
                .nth(one_dimensional_index)
                .unwrap(),
        )
    }
}
