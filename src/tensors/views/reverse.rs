use crate::tensors::dimensions;
use crate::tensors::views::{DataLayout, TensorMut, TensorRef};
use crate::tensors::Dimension;
use std::marker::PhantomData;

/**
 * A view over a tensor where some or all of the dimensions are iterated in reverse order.
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * use easy_ml::tensors::views::{TensorView, TensorReverse};
 * let ab = Tensor::from([("a", 2), ("b", 3)], (0..6).collect());
 * let reversed = ab.reverse(&["a"]);
 * let also_reversed = TensorView::from(TensorReverse::from(&ab, &["a"]));
 * assert_eq!(reversed, also_reversed);
 * assert_eq!(
 *     reversed,
 *     Tensor::from(
 *         [("a", 2), ("b", 3)],
 *         vec![
 *             3, 4, 5,
 *             0, 1, 2,
 *         ]
 *     )
 * );
 * ```
 */
#[derive(Clone, Debug)]
pub struct TensorReverse<T, S, const D: usize> {
    source: S,
    reversed: [bool; D],
    _type: PhantomData<T>,
}

impl<T, S, const D: usize> TensorReverse<T, S, D>
where
    S: TensorRef<T, D>,
{
    /**
     * Creates a TensorReverse from a source and a list of dimension names to reverse the
     * order of iteration for. The list cannot be more than the number of dimensions in the source
     * but it does not need to contain every dimension in the source. Any dimensions in the source
     * but not in the list of dimension names to reverse will continue to iterate in their normal
     * order.
     *
     * # Panics
     *
     * - If a dimension name is not in the source's shape or is repeated.
     */
    #[track_caller]
    pub fn from(source: S, dimensions: &[Dimension]) -> TensorReverse<T, S, D> {
        if crate::tensors::dimensions::has_duplicates_names(&dimensions) {
            panic!("Dimension names must all be unique: {:?}", &dimensions);
        }
        let shape = source.view_shape();
        if let Some(dimension) = dimensions.iter().find(|d| !dimensions::contains(&shape, d)) {
            panic!(
                "Dimension names to reverse must be in the source: {:?} is not in {:?}",
                dimension,
                shape
            );
        }
        let reversed = std::array::from_fn(|i| dimensions.contains(&shape[i].0));
        TensorReverse {
            source,
            reversed,
            _type: PhantomData,
        }
    }

    /**
     * Consumes the TensorReverse, yielding the source it was created from.
     */
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Gives a reference to the TensorReverse's source (in which the iteration order may be
     * different).
     */
    pub fn source_ref(&self) -> &S {
        &self.source
    }

    /**
     * Gives a mutable reference to the TensorReverse's source (in which the iteration order may
     * be different).
     */
    pub fn source_ref_mut(&mut self) -> &mut S {
        &mut self.source
    }
}

fn reverse_indexes<const D: usize>(
    indexes: &[usize; D],
    shape: &[(Dimension, usize); D],
    reversed: &[bool; D]
) -> [usize; D] {
    std::array::from_fn(|d| {
        if reversed[d] {
            let length = shape[d].1;
            // TensorRef requires dimensions are not of 0 length, so this never underflows
            let last_index = length - 1;
            let index = indexes[d];
            // swap dimension indexing, so 0 becomes length-1, and length-1 becomes 0
            last_index - index
        } else {
            indexes[d]
        }
    })
}

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// by only reversing some indexes and not introducing interior mutability, we implement
// TensorRef correctly as well.
/**
 * A TensorReverse implements TensorRef, with the dimension names the TensorReverse was created
 * with iterating in reverse order compared to the dimension names in the original source.
 */
unsafe impl<T, S, const D: usize> TensorRef<T, D> for TensorReverse<T, S, D>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.source.get_reference(
            reverse_indexes(&indexes, &self.view_shape(), &self.reversed)
        )
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.source.view_shape()
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        self.source.get_reference_unchecked(
            reverse_indexes(&indexes, &self.view_shape(), &self.reversed)
        )
    }

    fn data_layout(&self) -> DataLayout<D> {
        // There might be some specific cases where reversing maintains a linear order but
        // in general I think reversing only some indexes is going to mean any attempt at being
        // able to take a slice that matches up with our view_shape is gone.
        DataLayout::Other
    }
}

// # Safety
//
// The type implementing TensorMut must implement it correctly, so by delegating to it
// by only reversing some indexes and not introducing interior mutability, we implement
// TensorMut correctly as well.
/**
 * A TensorReverse implements TensorMut, with the dimension names the TensorReverse was created
 * with iterating in reverse order compared to the dimension names in the original source.
 */
unsafe impl<T, S, const D: usize> TensorMut<T, D> for TensorReverse<T, S, D>
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.source.get_reference_mut(
            reverse_indexes(&indexes, &self.view_shape(), &self.reversed)
        )
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        self.source.get_reference_unchecked_mut(
            reverse_indexes(&indexes, &self.view_shape(), &self.reversed)
        )
    }
}

#[test]
fn test_reversed_tensors() {
    use crate::tensors::Tensor;
    let tensor = Tensor::from([("a", 2), ("b", 3), ("c", 2)], (0..12).collect());
    assert_eq!(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], tensor.iter().collect::<Vec<_>>());
    let reversed = tensor.reverse_owned(&["a", "c"]);
    assert_eq!(vec![7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4], reversed.iter().collect::<Vec<_>>());
}
