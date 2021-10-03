use crate::tensors::views::{TensorAccess, TensorView, TensorRef, TensorMut, TensorRefAccess, TensorMutAccess};

pub mod views;
mod dimensions;

pub use dimensions::*;

// A named tensor http://nlp.seas.harvard.edu/NamedTensor
pub struct Tensor<T, const D: usize> {
    data: Vec<T>,
    dimensions: [(Dimension, usize); D],
}

impl<T, const D: usize> Tensor<T, D> {
    #[track_caller]
    pub fn new(data: Vec<T>, dimensions: [(Dimension, usize); D]) -> Self {
        assert_eq!(
            data.len(),
            dimensions.iter().map(|d| d.1).fold(1, |d1, d2| d1 * d2),
            "Length of dimensions must match size of data"
        );
        assert!(
            !has_duplicates(&dimensions),
            "Dimension names must all be unique"
        );

        Tensor { data, dimensions }
    }

    pub fn shape(&self) -> [(Dimension, usize); D] {
        self.dimensions.clone()
    }
}

pub struct TensorRefAccessor<'source, T, const D: usize> {
    tensor: &'source Tensor<T, D>,
    // mapping from the memory order of the tensor to the dimension order this accessor
    // was created from
    dimension_mapping: [usize; D],
    // strides for each dimension in the memory order of the tensor
    strides: [usize; D],
}

pub struct TensorMutAccessor<'source, T, const D: usize> {
    tensor: &'source mut Tensor<T, D>,
    // mapping from the memory order of the tensor to the dimension order this accessor
    // was created from
    dimension_mapping: [usize; D],
    // strides for each dimension in the memory order of the tensor
    strides: [usize; D],
}

pub struct TensorOwnedAccessor<T, const D: usize> {
    tensor: Tensor<T, D>,
    // mapping from the memory order of the tensor to the dimension order this accessor
    // was created from
    dimension_mapping: [usize; D],
    // strides for each dimension in the memory order of the tensor
    strides: [usize; D],
}

unsafe impl <'source, T, const D: usize> TensorRefAccess<T, D> for TensorRefAccessor<'source, T, D> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        let i = get_index(
            &indexes,
            &self.dimension_mapping,
            &self.strides,
            &self.tensor.dimensions
        )?;
        self.tensor.data.get(i)
    }
}

unsafe impl <'source, T, const D: usize> TensorRefAccess<T, D> for TensorMutAccessor<'source, T, D> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        let i = get_index(
            &indexes,
            &self.dimension_mapping,
            &self.strides,
            &self.tensor.dimensions
        )?;
        self.tensor.data.get(i)
    }
}

unsafe impl <'source, T, const D: usize> TensorMutAccess<T, D> for TensorMutAccessor<'source, T, D> {
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        let i = get_index(
            &indexes,
            &self.dimension_mapping,
            &self.strides,
            &self.tensor.dimensions
        )?;
        self.tensor.data.get_mut(i)
    }
}

unsafe impl <T, const D: usize> TensorRefAccess<T, D> for TensorOwnedAccessor<T, D> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        let i = get_index(
            &indexes,
            &self.dimension_mapping,
            &self.strides,
            &self.tensor.dimensions
        )?;
        self.tensor.data.get(i)
    }
}

unsafe impl <T, const D: usize> TensorMutAccess<T, D> for TensorOwnedAccessor<T, D> {
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        let i = get_index(
            &indexes,
            &self.dimension_mapping,
            &self.strides,
            &self.tensor.dimensions
        )?;
        self.tensor.data.get_mut(i)
    }
}

unsafe impl <'source, T, const D: usize> TensorRef<T, D> for &'source Tensor<T, D> {
    type Accessor = TensorRefAccessor<'source, T, D>;

    fn get_references(self, dimensions: [Dimension; D]) -> Option<Self::Accessor> {
        let dimension_mapping = dimension_mapping(&self.dimensions, &dimensions)?;
        let strides = compute_strides(&self.dimensions);
        Some(TensorRefAccessor {
            dimension_mapping,
            strides,
            tensor: self,
        })
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        Tensor::shape(self)
    }
}

unsafe impl <'source, T, const D: usize> TensorRef<T, D> for &'source mut Tensor<T, D> {
    type Accessor = TensorMutAccessor<'source, T, D>;

    fn get_references(self, dimensions: [Dimension; D]) -> Option<Self::Accessor> {
        let dimension_mapping = dimension_mapping(&self.dimensions, &dimensions)?;
        let strides = compute_strides(&self.dimensions);
        Some(TensorMutAccessor {
            dimension_mapping,
            strides,
            tensor: self,
        })
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        Tensor::shape(self)
    }
}

unsafe impl <'source, T, const D: usize> TensorMut<T, D> for &'source mut Tensor<T, D> {
    type AccessorMut = TensorMutAccessor<'source, T, D>;

    fn get_references_mut(self, dimensions: [Dimension; D]) -> Option<Self::AccessorMut> {
        self.get_references(dimensions)
    }
}

unsafe impl <T, const D: usize> TensorRef<T, D> for Tensor<T, D> {
    type Accessor = TensorOwnedAccessor<T, D>;

    fn get_references(self, dimensions: [Dimension; D]) -> Option<Self::Accessor> {
        let dimension_mapping = dimension_mapping(&self.dimensions, &dimensions)?;
        let strides = compute_strides(&self.dimensions);
        Some(TensorOwnedAccessor {
            dimension_mapping,
            strides,
            tensor: self,
        })
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        Tensor::shape(self)
    }
}

unsafe impl <T, const D: usize> TensorMut<T, D> for Tensor<T, D> {
    type AccessorMut = TensorOwnedAccessor<T, D>;

    fn get_references_mut(self, dimensions: [Dimension; D]) -> Option<Self::AccessorMut> {
        self.get_references(dimensions)
    }
}

// Computes a mapping from a set of dimensions in memory order to a matching set of
// dimensions in an arbitary order.
// Returns a list where each dimension in the memory order is mapped to the requested order,
// such that if the memory order is x,y,z but the requested order is z,y,x then the mapping
// is [2,1,0] as this maps the first dimension x to the third dimension x, the second dimension y
// to the second dimension y, and the third dimension z to to the first dimension z.
fn dimension_mapping<const D: usize>(
    memory: &[(Dimension, usize); D],
    requested: &[Dimension; D]
) -> Option<[usize; D]> {
    let mut mapping = [ 0; D ];
    for d in 0..D {
        let dimension = memory[d].0;
        // happy path, requested dimension is in the same order as in memory
        let order = if requested[d] == dimension {
            d
        } else {
            // If dimensions are in a different order, find the requested dimension with the
            // matching dimension name.
            // Since both lists are the same length and we know our memory order won't contain
            // duplicates this also ensures the two lists have exactly the same set of names
            // as otherwise one of these `find`s will fail.
            let (n, _) = requested.iter().enumerate().find(|(_, d)| **d == dimension)?;
            n
        };
        mapping[d] = order;
    }
    Some(mapping)
}

fn compute_strides<const D: usize>(dimensions: &[(Dimension, usize); D]) -> [usize; D] {
    let mut strides = [0; D];
    for d in 0..D {
        strides[d] = dimensions
          .iter()
          .skip(d + 1)
          .map(|d| d.1)
          .fold(1, |d1, d2| d1 * d2);
    }
    strides
}

// returns the 1 dimensional index to use to get the requested index into some tensor
#[inline]
fn get_index<const D: usize>(
    // indexes to use
    indexes: &[usize; D],
    // mapping from indexes to memory order of the tensor
    dimension_mapping: &[usize; D],
    // strides for indexing into the tensor
    strides: &[usize; D],
    // dimensions of the tensor to index into
    dimensions: &[(Dimension, usize); D],
) -> Option<usize> {
    let mut index = 0;
    for d in 0..D {
        let n = indexes[dimension_mapping[d]];
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
    pub fn get(&self, dimensions: [Dimension; D]) -> TensorAccess<T, TensorRefAccessor<T, D>, D> {
        match self.get_references(dimensions) {
            Some(access) => TensorAccess::from(access),
            None => panic!(
                "Unable to index {:?}, Tensor dimensions are {:?}.",
                dimensions, self.shape()
            ),
        }
    }

    #[track_caller]
    pub fn get_mut(&mut self, dimensions: [Dimension; D]) -> TensorAccess<T, TensorMutAccessor<T, D>, D> {
        let shape = self.shape();
        match self.get_references(dimensions) {
            Some(access) => TensorAccess::from(access),
            None => panic!(
                "Unable to index {:?}, Tensor dimensions are {:?}.",
                dimensions, shape
            ),
        }
    }

    #[track_caller]
    pub fn get_owned(self, dimensions: [Dimension; D]) -> TensorAccess<T, TensorOwnedAccessor<T, D>, D> {
        let shape = self.shape();
        match self.get_references(dimensions) {
            Some(access) => TensorAccess::from(access),
            None => panic!(
                "Unable to index {:?}, Tensor dimensions are {:?}.",
                dimensions, shape
            ),
        }
    }

    //
    // #[track_caller]
    // pub fn get_reference(&self, dimensions: [(Dimension, usize); D]) -> &T {
    //     match self.try_get_reference(dimensions) {
    //         Some(reference) => reference,
    //         None => panic!(
    //             "Unable to index with {:?}, Tensor dimensions are {:?}.",
    //             dimensions, self.shape()
    //         )
    //     }
    // }
}

// impl<T, const D: usize> Tensor<T, D>
// where
//     T: Clone,
// {
//     #[track_caller]
//     pub fn get(&self, dimensions: [(Dimension, usize); D]) -> T {
//         self.get_reference(dimensions).clone()
//     }
// }

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
fn bad_indexing_test() {
    let tensor = Tensor::new(vec![1, 2, 3, 4], [of("x", 2), of("y", 2)]);
    let xx = (&tensor).get_references([dimension("x"), dimension("x")]);
    assert!(xx.is_none());
}

#[test]
fn test_dimension_mapping() {
    let mapping = dimension_mapping(&[of("x", 0), of("y", 0), of("z", 0)], &[dimension("x"), dimension("y"), dimension("z")]);
    assert_eq!([0, 1, 2], mapping.unwrap());
    let mapping = dimension_mapping(&[of("x", 0), of("y", 0), of("z", 0)], &[dimension("z"), dimension("y"), dimension("x")]);
    assert_eq!([2, 1, 0], mapping.unwrap());
}
