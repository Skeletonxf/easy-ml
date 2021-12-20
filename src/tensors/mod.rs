use crate::tensors::indexing::TensorAccess;
use crate::tensors::views::{TensorMut, TensorRef, TensorView};

mod dimensions;
pub mod indexing;
pub mod views;

pub use dimensions::*;

// A named tensor http://nlp.seas.harvard.edu/NamedTensor
pub struct Tensor<T, const D: usize> {
    data: Vec<T>,
    dimensions: [(Dimension, usize); D],
    strides: [usize; D],
}

impl<T, const D: usize> Tensor<T, D> {
    #[track_caller]
    pub fn new(data: Vec<T>, dimensions: [(Dimension, usize); D]) -> Self {
        assert_eq!(
            data.len(),
            dimensions.iter().map(|d| d.1).product(),
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
}

#[test]
fn indexing_test() {
    let tensor = Tensor::new(vec![1, 2, 3, 4], [of("x", 2), of("y", 2)]);
    let xy = tensor.get([dimension("x"), dimension("y")]);
    let yx = tensor.get([dimension("y"), dimension("x")]);
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
    Tensor::new(vec![1, 2, 3, 4], [of("x", 2), of("x", 2)]);
}

#[test]
#[should_panic]
fn wrong_size() {
    Tensor::new(vec![1, 2, 3, 4], [of("x", 2), of("y", 3)]);
}

#[test]
#[should_panic]
fn bad_indexing() {
    let tensor = Tensor::new(vec![1, 2, 3, 4], [of("x", 2), of("y", 2)]);
    tensor.get([dimension("x"), dimension("x")]);
}
