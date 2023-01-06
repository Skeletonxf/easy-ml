use crate::tensors::dimensions;
use crate::tensors::views::{DataLayout, TensorMut, TensorRef};
use crate::tensors::Dimension;
use std::marker::PhantomData;

/**
 * A combination of new dimension names and a tensor.
 *
 * The provided dimension names override the dimension names in the
 * [`view_shape`](TensorRef::view_shape) of the TensorRef exposed.
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * use easy_ml::tensors::views::{TensorRename, TensorView};
 * let a_b = Tensor::from([("a", 2), ("b", 2)], (0..4).collect());
 * let b_c = TensorView::from(TensorRename::from(&a_b, ["b", "c"]));
 * let also_b_c = a_b.rename_view(["b", "c"]);
 * assert_ne!(a_b, b_c);
 * assert_eq!(b_c, also_b_c);
 * ```
 */
#[derive(Clone, Debug)]
pub struct TensorRename<T, S, const D: usize> {
    source: S,
    dimensions: [Dimension; D],
    _type: PhantomData<T>,
}

impl<T, S, const D: usize> TensorRename<T, S, D>
where
    S: TensorRef<T, D>,
{
    /**
     * Creates a TensorRename from a source and a list of dimension names to override the
     * view_shape with.
     *
     * # Panics
     *
     * - If a dimension name is not unique
     */
    #[track_caller]
    pub fn from(source: S, dimensions: [Dimension; D]) -> TensorRename<T, S, D> {
        if crate::tensors::dimensions::has_duplicates_names(&dimensions) {
            panic!("Dimension names must all be unique: {:?}", &dimensions);
        }
        TensorRename {
            source,
            dimensions,
            _type: PhantomData,
        }
    }

    /**
     * Consumes the TensorRename, yielding the source it was created from.
     */
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Gives a reference to the TensorRename's source (in which the dimension names may be
     * different).
     */
    pub fn source_ref(&self) -> &S {
        &self.source
    }

    /**
     * Gives a mutable reference to the TensorRename's source (in which the dimension names may be
     * different).
     */
    pub fn source_ref_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /**
     * Gets the dimension names this TensorRename is overriding the source it
     * was created from with.
     */
    pub fn get_names(&self) -> &[Dimension; D] {
        &self.dimensions
    }

    // # Safety
    //
    // Giving out a mutable reference to our dimension names could allow a caller to make
    // them non unique which would invalidate our TensorRef implementation. However, a setter
    // method is fine because we can ensure this invariant is not broken.
    /**
     * Sets the dimension names this TensorRename is overriding the source it
     * was created from with.
     *
     * # Panics
     *
     * - If a dimension name is not unique
     */
    #[track_caller]
    pub fn set_names(&mut self, dimensions: [Dimension; D]) {
        if crate::tensors::dimensions::has_duplicates_names(&dimensions) {
            panic!("Dimension names must all be unique: {:?}", &dimensions);
        }
        self.dimensions = dimensions;
    }
}

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, and ensuring we do
// not introduce non unique dimension names, we implement TensorRef correctly as well.
/**
 * A TensorRename implements TensorRef, with the dimension names the TensorRename was created
 * with overriding the dimension names in the original source.
 */
unsafe impl<T, S, const D: usize> TensorRef<T, D> for TensorRename<T, S, D>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.source.get_reference(indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        let mut shape = self.source.view_shape();
        for (i, element) in shape.iter_mut().enumerate() {
            *element = (self.dimensions[i], element.1);
        }
        shape
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        self.source.get_reference_unchecked(indexes)
    }

    fn data_layout(&self) -> DataLayout<D> {
        let data_layout = self.source.data_layout();
        match data_layout {
            DataLayout::Linear(order) => {
                let shape = self.source.view_shape();
                // Map the dimension name order to position in the view shape instead of name
                let order_d: [usize; D] = std::array::from_fn(|i| {
                    let name = order[i];
                    dimensions::position_of(&shape, name)
                        .unwrap_or_else(|| panic!(
                            "Source implementation contained dimension {} in data_layout that was not in the view_shape {:?} which breaks the contract of TensorRef",
                            name, &shape
                        ))
                });
                // TensorRename doesn't move dimensions around, so now we can map from position
                // order to our new dimension names.
                DataLayout::Linear(std::array::from_fn(|i| self.dimensions[order_d[i]]))
            }
            _ => data_layout,
        }
    }
}

// # Safety
//
// The type implementing TensorMut must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, and ensuring we do
// not introduce non unique dimension names, we implement TensorMut correctly as well.
/**
 * A TensorRename implements TensorMut, with the dimension names the TensorRename was created
 * with overriding the dimension names in the original source.
 */
unsafe impl<T, S, const D: usize> TensorMut<T, D> for TensorRename<T, S, D>
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.source.get_reference_mut(indexes)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        self.source.get_reference_unchecked_mut(indexes)
    }
}

#[test]
fn test_renamed_view_shape() {
    use crate::tensors::Tensor;
    let tensor = Tensor::from([("a", 2), ("b", 2)], (0..4).collect());
    let b_c = tensor.rename_view(["b", "c"]);
    assert_eq!([("b", 2), ("c", 2)], b_c.shape());
}
