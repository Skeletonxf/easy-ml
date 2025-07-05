use crate::tensors;
use crate::tensors::Dimension;
use crate::tensors::dimensions;
use crate::tensors::views::{DataLayout, TensorMut, TensorRef};
use std::marker::PhantomData;

/**
 * A new shape to override indexing an existing tensor. The dimensionality and individual
 * dimension lengths can be changed, but the total number of elements in the new shape must
 * match the existing tensor's shape. Elements are still in the same order (in memory) as
 * the source tensor, none of the data is moved around - though iteration might be in
 * a different order with the new shape.
 *
 * If you just need to rename dimensions without changing them, see
 * [TensorRename](tensors::views::TensorRename)
 *
 * This types' generics can be read as a TensorReshape is generic over some element of type T
 * from an existing source of type S of dimensionality D, and this tensor has dimensionality D2,
 * which might be different to D, but will have the same total number of elements.
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * use easy_ml::tensors::views::{TensorReshape, TensorView};
 * let tensor = Tensor::from([("a", 2), ("b", 2)], (0..4).collect());
 * let flat = TensorView::from(TensorReshape::from(tensor, [("i", 4)])); // or use tensor.reshape_view_owned
 * assert_eq!(flat, Tensor::from([("i", 4)], (0..4).collect()));
 * ```
 */
#[derive(Clone, Debug)]
pub struct TensorReshape<T, S, const D: usize, const D2: usize> {
    source: S,
    shape: [(Dimension, usize); D2],
    strides: [usize; D2],
    source_strides: [usize; D],
    _type: PhantomData<T>,
}

impl<T, S, const D: usize, const D2: usize> TensorReshape<T, S, D, D2>
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
     * If you don't need to change the dimensionality, see
     * [from](TensorReshape::from_existing_dimensions)
     *
     * # Panics
     *
     * - If the new shape has a different number of elements to the existing
     * shape in the source
     * - If the new shape has duplicate dimension names
     */
    #[track_caller]
    pub fn from(source: S, shape: [(Dimension, usize); D2]) -> TensorReshape<T, S, D, D2> {
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
        let source_strides = tensors::compute_strides(&source.view_shape());
        TensorReshape {
            source,
            shape,
            strides: tensors::compute_strides(&shape),
            source_strides,
            _type: PhantomData,
        }
    }

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

impl<T, S, const D: usize> TensorReshape<T, S, D, D>
where
    S: TensorRef<T, D>,
{
    /**
     * Creates a TensorReshape from a source and new dimension lengths with the same dimensionality
     * as the source to override the view_shape with. The new shape must correspond to the same
     * number of total elements, but it need not match on individual dimension lengths.
     *
     * If you just need to rename dimensions without changing them, see
     * [TensorRename](tensors::views::TensorRename)
     * If you need to change the dimensionality, see [from](TensorReshape::from)
     *
     * # Panics
     *
     * - If the new shape has a different number of elements to the existing
     * shape in the source
     */
    #[track_caller]
    pub fn from_existing_dimensions(source: S, lengths: [usize; D]) -> TensorReshape<T, S, D, D> {
        let previous_shape = source.view_shape();
        let shape = std::array::from_fn(|n| (previous_shape[n].0, lengths[0]));
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
        let source_strides = tensors::compute_strides(&source.view_shape());
        TensorReshape {
            source,
            shape,
            strides: tensors::compute_strides(&shape),
            source_strides,
            _type: PhantomData,
        }
    }
}

fn unflatten<const D: usize>(nth: usize, strides: &[usize; D]) -> [usize; D] {
    let mut steps_remaining = nth;
    let mut index = [0; D];
    for d in 0..D {
        let stride = strides[d];
        // If the stride was 20, then 0-19 for indexes would be 0, 20-39 would be 1
        // and so on
        index[d] = steps_remaining / stride;
        // Given such a stride of 20, we then need to look at what was rounded off
        // An index of 0 or 20 into such a stride would mean we're done, 1 or 21 would
        // mean we have 1 step left and so on
        steps_remaining = steps_remaining % stride;
    }
    index
}

#[test]
fn unflatten_produces_indices_in_n_dimensions() {
    let strides = tensors::compute_strides(&[("x", 2), ("y", 2)]);
    assert_eq!([0, 0], unflatten(0, &strides));
    assert_eq!([0, 1], unflatten(1, &strides));
    assert_eq!([1, 0], unflatten(2, &strides));
    assert_eq!([1, 1], unflatten(3, &strides));

    let strides = tensors::compute_strides(&[("x", 3), ("y", 2)]);
    assert_eq!([0, 0], unflatten(0, &strides));
    assert_eq!([0, 1], unflatten(1, &strides));
    assert_eq!([1, 0], unflatten(2, &strides));
    assert_eq!([1, 1], unflatten(3, &strides));
    assert_eq!([2, 0], unflatten(4, &strides));
    assert_eq!([2, 1], unflatten(5, &strides));

    let strides = tensors::compute_strides(&[("x", 2), ("y", 3)]);
    assert_eq!([0, 0], unflatten(0, &strides));
    assert_eq!([0, 1], unflatten(1, &strides));
    assert_eq!([0, 2], unflatten(2, &strides));
    assert_eq!([1, 0], unflatten(3, &strides));
    assert_eq!([1, 1], unflatten(4, &strides));
    assert_eq!([1, 2], unflatten(5, &strides));

    let strides = tensors::compute_strides(&[("x", 2), ("y", 3), ("z", 1)]);
    assert_eq!([0, 0, 0], unflatten(0, &strides));
    assert_eq!([0, 1, 0], unflatten(1, &strides));
    assert_eq!([0, 2, 0], unflatten(2, &strides));
    assert_eq!([1, 0, 0], unflatten(3, &strides));
    assert_eq!([1, 1, 0], unflatten(4, &strides));
    assert_eq!([1, 2, 0], unflatten(5, &strides));

    let strides = tensors::compute_strides(&[("batch", 1), ("x", 2), ("y", 3)]);
    assert_eq!([0, 0, 0], unflatten(0, &strides));
    assert_eq!([0, 0, 1], unflatten(1, &strides));
    assert_eq!([0, 0, 2], unflatten(2, &strides));
    assert_eq!([0, 1, 0], unflatten(3, &strides));
    assert_eq!([0, 1, 1], unflatten(4, &strides));
    assert_eq!([0, 1, 2], unflatten(5, &strides));

    let strides = tensors::compute_strides(&[("x", 2), ("y", 3), ("z", 2)]);
    assert_eq!([0, 0, 0], unflatten(0, &strides));
    assert_eq!([0, 0, 1], unflatten(1, &strides));
    assert_eq!([0, 1, 0], unflatten(2, &strides));
    assert_eq!([0, 1, 1], unflatten(3, &strides));
    assert_eq!([0, 2, 0], unflatten(4, &strides));
    assert_eq!([0, 2, 1], unflatten(5, &strides));
    assert_eq!([1, 0, 0], unflatten(6, &strides));
    assert_eq!([1, 0, 1], unflatten(7, &strides));
    assert_eq!([1, 1, 0], unflatten(8, &strides));
    assert_eq!([1, 1, 1], unflatten(9, &strides));
    assert_eq!([1, 2, 0], unflatten(10, &strides));
    assert_eq!([1, 2, 1], unflatten(11, &strides));
}

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// by only flattening the caller's index into a one dimensional one, and then unflattening
// it back into the source dimensionality and not introducing interior mutability, we implement
// TensorRef correctly as well.
/**
 * A TensorReshape implements TensorRef, with the data in the same order as the original source.
 */
unsafe impl<T, S, const D: usize, const D2: usize> TensorRef<T, D2> for TensorReshape<T, S, D, D2>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D2]) -> Option<&T> {
        let one_dimensional_index =
            tensors::get_index_direct(&indexes, &self.strides, &self.shape)?;
        self.source
            .get_reference(unflatten(one_dimensional_index, &self.source_strides))
    }

    fn view_shape(&self) -> [(Dimension, usize); D2] {
        self.shape
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D2]) -> &T {
        unsafe {
            // It is the caller's responsibility to always call with indexes in range,
            // therefore out of bounds lookups created by get_index_direct_unchecked should never
            // happen.
            let one_dimensional_index =
                tensors::get_index_direct_unchecked(&indexes, &self.strides);
            self.source
                .get_reference_unchecked(unflatten(one_dimensional_index, &self.source_strides))
        }
    }

    fn data_layout(&self) -> DataLayout<D2> {
        // There might be some cases where assigning a new shape maintains a linear order
        // but it seems like a lot of effort to maintain a correct mapping from the original
        // linear order to the new one, given we can change even dimensionality in this mapping.
        DataLayout::Other
    }
}

/**
 * A TensorReshape implements TensorMut, with the data in the same order as the original source.
 */
unsafe impl<T, S, const D: usize, const D2: usize> TensorMut<T, D2> for TensorReshape<T, S, D, D2>
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D2]) -> Option<&mut T> {
        let one_dimensional_index =
            tensors::get_index_direct(&indexes, &self.strides, &self.shape)?;
        self.source
            .get_reference_mut(unflatten(one_dimensional_index, &self.source_strides))
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D2]) -> &mut T {
        unsafe {
            // It is the caller's responsibility to always call with indexes in range,
            // therefore out of bounds lookups created by get_index_direct_unchecked should never
            // happen.
            let one_dimensional_index =
                tensors::get_index_direct_unchecked(&indexes, &self.strides);
            self.source
                .get_reference_unchecked_mut(unflatten(one_dimensional_index, &self.source_strides))
        }
    }
}
