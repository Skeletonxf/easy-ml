/**
 * Generic n dimensional [named tensors](http://nlp.seas.harvard.edu/NamedTensor).
 *
 * Tensors are generic over some type `T` and some usize `D`. If `T` is [Numeric](super::numeric)
 * then the tensor can be used in a mathematical way. `D` is the number of dimensions in the tensor
 * and a compile time constant.
 */

use crate::tensors::indexing::TensorAccess;
use crate::tensors::views::{TensorMut, TensorRef, TensorView};

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
type Dimension = &'static str;

/**
 * A [named tensor](http://nlp.seas.harvard.edu/NamedTensor).
 *
 * TODO: Summary
 *
 * See also:
 * - [indexing](indexing)
 */
pub struct Tensor<T, const D: usize> {
    data: Vec<T>,
    dimensions: [(Dimension, usize); D],
    strides: [usize; D],
}

/**
 * Returns the product of the provided dimension lengths
 *
 * This is equal to the number of elements that will be stored for these dimensions.
 * A 0 dimensional tensor stores exactly 1 element, a 1 dimensional tensor stores N elements,
 * a 2 dimensional tensor stores NxM elements and so on.
 */
pub fn elements<const D: usize>(dimensions: &[(Dimension, usize); D]) -> usize {
    dimensions.iter().map(|d| d.1).product()
}

impl<T, const D: usize> Tensor<T, D> {
    #[track_caller]
    pub fn new(data: Vec<T>, dimensions: [(Dimension, usize); D]) -> Self {
        assert_eq!(
            data.len(),
            elements(&dimensions),
            "Length of dimensions must match size of data"
        );
        assert!(
            !has_duplicates(&dimensions),
            "Dimension names must all be unique"
        );

        let strides = compute_strides(&dimensions);
        Tensor {
            data,
            dimensions,
            strides,
        }
    }

    pub fn shape(&self) -> [(Dimension, usize); D] {
        self.dimensions
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
}

unsafe impl<'source, T, const D: usize> TensorRef<T, D> for &'source Tensor<T, D> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions)?;
        self.data.get(i)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        Tensor::shape(self)
    }
}

unsafe impl<'source, T, const D: usize> TensorRef<T, D> for &'source mut Tensor<T, D> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions)?;
        self.data.get(i)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        Tensor::shape(self)
    }
}

unsafe impl<'source, T, const D: usize> TensorMut<T, D> for &'source mut Tensor<T, D> {
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions)?;
        self.data.get_mut(i)
    }
}

unsafe impl<T, const D: usize> TensorRef<T, D> for Tensor<T, D> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions)?;
        self.data.get(i)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        Tensor::shape(self)
    }
}

unsafe impl<T, const D: usize> TensorMut<T, D> for Tensor<T, D> {
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions)?;
        self.data.get_mut(i)
    }
}

fn compute_strides<const D: usize>(dimensions: &[(Dimension, usize); D]) -> [usize; D] {
    let mut strides = [0; D];
    for d in 0..D {
        strides[d] = dimensions
            .iter()
            .skip(d + 1)
            .map(|d| d.1)
            .product();
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

fn has_duplicates(dimensions: &[(Dimension, usize)]) -> bool {
    for i in 1..dimensions.len() {
        let name = dimensions[i - 1].0;
        if dimensions[i..].iter().any(|d| d.0 == name) {
            return true;
        }
    }
    false
}

fn has(dimensions: &[(Dimension, usize)], name: Dimension) -> bool {
    dimensions.iter().any(|d| d.0 == name)
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

    #[track_caller]
    pub fn get(&self, dimensions: [Dimension; D]) -> TensorAccess<T, &Tensor<T, D>, D> {
        TensorAccess::from(self, dimensions)
    }

    #[track_caller]
    pub fn get_mut(&mut self, dimensions: [Dimension; D]) -> TensorAccess<T, &mut Tensor<T, D>, D> {
        TensorAccess::from(self, dimensions)
    }

    #[track_caller]
    pub fn get_owned(self, dimensions: [Dimension; D]) -> TensorAccess<T, Tensor<T, D>, D> {
        TensorAccess::from(self, dimensions)
    }

    /**
     * Renames the dimension names of the tensor without changing the lengths of the dimensions
     * in the tensor.
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * let mut tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], [("x", 2), ("y", 3)]);
     * tensor.rename(["y", "z"]);
     * assert_eq!([("y", 2), ("z", 3)], tensor.shape());
     * ```
     */
    // TODO: View version
    pub fn rename(&mut self, dimensions: [Dimension; D]) {
        for d in 0..D {
            self.dimensions[d].0 = dimensions[d];
        }
    }
}

impl<T, const D: usize> Tensor<T, D>
where
    T: Clone,
{
    // TODO: Establish a consistent naming scheme for [(Dimension, usize)], [usize] and [Dimension]
    // TODO: Actually test this
    // TODO: Mut version
    // TODO: View version
    #[track_caller]
    pub fn transpose(&self, dimensions: [Dimension; D]) -> Tensor<T, D> {
        let transposed_order = TensorAccess::from(self, dimensions); // TODO: Handle error case, propagate as Dimension names to transpose to must be the same set of dimension names in the tensor
        let transposed_shape = transposed_order.shape();
        let dummy = transposed_order.get_reference([0; D]).clone();

        let mut transposed = Tensor::new(vec![dummy; elements(&self.dimensions)], transposed_shape);

        let mut transposed_elements = transposed_order.index_reference_iter();
        for elem in transposed.data.iter_mut() {
            *elem = transposed_elements.next().unwrap().clone();
        }

        transposed
    }
}

#[test]
fn indexing_test() {
    let tensor = Tensor::new(vec![1, 2, 3, 4], [("x", 2), ("y", 2)]);
    let xy = tensor.get(["x", "y"]);
    let yx = tensor.get(["y", "x"]);
    assert_eq!(xy.get([0, 0]), 1);
    assert_eq!(xy.get([0, 1]), 2);
    assert_eq!(xy.get([1, 0]), 3);
    assert_eq!(xy.get([1, 1]), 4);
    assert_eq!(yx.get([0, 0]), 1);
    assert_eq!(yx.get([0, 1]), 3);
    assert_eq!(yx.get([1, 0]), 2);
    assert_eq!(yx.get([1, 1]), 4);
}

#[test]
#[should_panic]
fn repeated_name() {
    Tensor::new(vec![1, 2, 3, 4], [("x", 2), ("x", 2)]);
}

#[test]
#[should_panic]
fn wrong_size() {
    Tensor::new(vec![1, 2, 3, 4], [("x", 2), ("y", 3)]);
}

#[test]
#[should_panic]
fn bad_indexing() {
    let tensor = Tensor::new(vec![1, 2, 3, 4], [("x", 2), ("y", 2)]);
    tensor.get(["x", "x"]);
}
