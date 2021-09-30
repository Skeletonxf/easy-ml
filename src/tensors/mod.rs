use crate::tensors::views::{TensorView, TensorRef};

pub mod views;

// A named tensor http://nlp.seas.harvard.edu/NamedTensor
pub struct Tensor<T, const D: usize> {
    data: Vec<T>,
    dimensions: [(Dimension, usize); D],
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Dimension {
    name: &'static str,
}

impl Dimension {
    pub fn new(name: &'static str) -> Self {
        Dimension { name }
    }
}

pub fn dimension(name: &'static str) -> Dimension {
    Dimension::new(name)
}

pub fn of(name: &'static str, length: usize) -> (Dimension, usize) {
    (dimension(name), length)
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

impl<T, const D: usize> Tensor<T, D> {
    pub(crate) fn _try_get_reference(&self, mut dimensions: [(Dimension, usize); D]) -> Option<&T> {
        // go through our dimensions in memory order
        for (i, (dimension, _)) in self.dimensions.iter().enumerate() {
            // check if the dimensions given are in the same order
            if dimensions[i].0 != *dimension {
                // swap input dimensions to put them in the same order as our memory order,
                // returning None if we don't have a match
                let (j, _) = dimensions
                    .iter()
                    .enumerate()
                    .find(|(_, (d, _))| *d == *dimension)?;
                if j < i {
                    return None;
                }
                // put this dimension into the same order as our memory order
                dimensions.swap(i, j);
            }
        }

        let mut index = 0;
        for (d, (_, n)) in dimensions.iter().enumerate() {
            let stride = self
                .dimensions
                .iter()
                .skip(d + 1)
                .map(|d| d.1)
                .fold(1, |d1, d2| d1 * d2);
            index += n * stride;
        }
        self.data.get(index)
    }

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
    pub fn get_reference(&self, dimensions: [(Dimension, usize); D]) -> &T {
        match self.try_get_reference(dimensions) {
            Some(reference) => reference,
            None => panic!(
                "Unable to index with {:?}, Tensor dimensions are {:?}.",
                dimensions, self.shape()
            )
        }
    }
}

impl<T, const D: usize> Tensor<T, D>
where
    T: Clone,
{
    #[track_caller]
    pub fn get(&self, dimensions: [(Dimension, usize); D]) -> T {
        self.get_reference(dimensions).clone()
    }
}

#[test]
fn indexing_test() {
    let tensor = Tensor::new(vec![1, 2, 3, 4], [of("x", 2), of("y", 2)]);
    fn get_xy(tensor: &Tensor<u32, 2>, x: usize, y: usize) -> u32 {
        tensor.get([of("x", x), of("y", y)])
    }
    assert_eq!(get_xy(&tensor, 0, 0), 1);
    assert_eq!(get_xy(&tensor, 0, 1), 2);
    assert_eq!(get_xy(&tensor, 1, 0), 3);
    assert_eq!(get_xy(&tensor, 1, 1), 4);
    fn get_yx(tensor: &Tensor<u32, 2>, x: usize, y: usize) -> u32 {
        tensor.get([of("y", y), of("x", x)])
    }
    assert_eq!(get_yx(&tensor, 0, 0), 1);
    assert_eq!(get_yx(&tensor, 0, 1), 2);
    assert_eq!(get_yx(&tensor, 1, 0), 3);
    assert_eq!(get_yx(&tensor, 1, 1), 4);
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
