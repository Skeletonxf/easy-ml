use std::marker::PhantomData;
use crate::tensors::Dimension;
use crate::tensors::views::TensorRef;

#[derive(Clone, Debug)]
pub struct TensorIndex<T, S, const D: usize, const I: usize> {
    source: S,
    index: [(Dimension, usize); I],
    _type: PhantomData<T>,
}

impl <T, S, const D: usize, const I: usize> TensorIndex<T, S, D, I>
where
    S: TensorRef<T, D>
{
    pub fn from(source: S, index: [(Dimension, usize); I]) -> TensorIndex<T, S, D, I> {
        assert!(index.iter().all(|d| source.view_shape().contains(d)));
        TensorIndex {
            source,
            index,
            _type: PhantomData,
        }
    }
}

fn calculate_view_shape_1(source: &[(Dimension, usize)], index: &[(Dimension, usize)]) -> [(Dimension, usize); 1] {
    let mut iter = source.iter().filter(|d| !index.contains(d));
    [ iter.next().unwrap().clone() ]
}

fn calculate_view_shape_2(source: &[(Dimension, usize)], index: &[(Dimension, usize)]) -> [(Dimension, usize); 2] {
    let mut iter = source.iter().filter(|d| !index.contains(d));
    [ iter.next().unwrap().clone(), iter.next().unwrap().clone() ]
}

fn calculate_view_shape_3(source: &[(Dimension, usize)], index: &[(Dimension, usize)]) -> [(Dimension, usize); 3] {
    let mut iter = source.iter().filter(|d| !index.contains(d));
    [ iter.next().unwrap().clone(), iter.next().unwrap().clone(), iter.next().unwrap().clone() ]
}

unsafe impl <T, S> TensorRef<T, 2> for TensorIndex<T, S, 3, 1>
where
    S: TensorRef<T, 3>
 {
    fn try_get_reference(&self, dimensions: [(Dimension, usize); 2]) -> Option<&T> {
        let combined_dimensions = [self.index[0], dimensions[0], dimensions[1]];
        self.source.try_get_reference(combined_dimensions)
    }

    fn view_shape(&self) -> [(Dimension, usize); 2] {
        calculate_view_shape_2(&self.source.view_shape(), &self.index)
    }
}

unsafe impl <T, S> TensorRef<T, 1> for TensorIndex<T, S, 2, 1>
where
    S: TensorRef<T, 2>
 {
    fn try_get_reference(&self, dimensions: [(Dimension, usize); 1]) -> Option<&T> {
        let combined_dimensions = [self.index[0], dimensions[0]];
        self.source.try_get_reference(combined_dimensions)
    }

    fn view_shape(&self) -> [(Dimension, usize); 1] {
        calculate_view_shape_1(&self.source.view_shape(), &self.index)
    }
}
