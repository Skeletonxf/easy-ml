use crate::tensors::views::TensorRef;
use crate::tensors::Dimension;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct TensorIndex<T, S, const D: usize, const I: usize> {
    source: S,
    provided: [Option<usize>; D],
    _type: PhantomData<T>,
}

impl<T, S, const D: usize, const I: usize> TensorIndex<T, S, D, I>
where
    S: TensorRef<T, D>,
{
    /**
     * Creates a TensorIndex from a source and a list of provided dimension name/index pairs.
     *
     * The corresponding dimensions in the source will be masked to always return the provided
     * index. Henece, a matrix can be viewed as a vector if you provide one of the row/column
     * index to use. More generally, the tensor the TensorIndex exposes will have a dimensionality
     * of D - I, where D is the dimensionality of the source, and I is the dimensionality of the
     * provided indexes.
     */
    pub fn from(source: S, provided_index: [(Dimension, usize); I]) -> TensorIndex<T, S, D, I> {
        let shape = source.view_shape();
        if I > D {
            panic!("D - I must be >= 0, D: {:?}, I: {:?}", D, I);
        }
        let mut provided = [ None; D ];
        for (name, index) in &provided_index {
            // Every provided index must match a dimension name in the source and be a valid
            // index within the length
            match shape.iter().enumerate().find(|(_i, (n, length))| n == name && index < length) {
                None => panic!(
                    "Provided indexes must all correspond to valid indexes into the source shape, source shape: {:?}, provided: {:?}",
                    shape,
                    provided_index,
                ),
                // Assign the provided index to the matching position of the source
                Some((i, (_n, _length))) => provided[i] = Some(*index),
            }
        }
        TensorIndex {
            source,
            provided,
            _type: PhantomData,
        }
    }
}

macro_rules! tensor_ref_impl {
    (unsafe impl TensorRef for TensorIndex $d:literal $i:literal) => {
        // # Safety
        // The source we index into implements TensorRef, and we do not give out any mutable
        // references to it. Since it may not implement interior mutability due to implementing
        // TensorRef, we know that it won't change under us. Since we know it won't change under
        // us, we can rely on the invariants when we created the provided array. The provided
        // array therefore will be in the same order as the source's view_shape. Hence we can
        // index correctly by filling in the None slots of provided with the supplied indexes,
        // which also have to be in order.
        unsafe impl <T, S> TensorRef<T, {$d - $i}> for TensorIndex<T, S, $d, $i>
        where
            S: TensorRef<T, $d>
         {
            fn get_reference(&self, indexes: [usize; $d - $i]) -> Option<&T> {
                let mut supplied = indexes.iter();
                // Indexes have to be in the order of our shape, so they must fill in the None
                // slots of our provided array since we created that in the same order as our
                // view_shape
                let mut combined = self.provided.iter().map(|provided| match provided {
                    None => *supplied.next().unwrap(),
                    Some(i) => *i,
                });
                let index = [ 0; $d ].map(|_| combined.next().unwrap());
                self.source.get_reference(index)
            }

            fn view_shape(&self) -> [(Dimension, usize); $d - $i] {
                let shape = self.source.view_shape();
                let mut unprovided = shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| self.provided[*i].is_none())
                    .map(|(_, (name, length))| (*name, *length));
                [ ("", 0); $d - $i ].map(|_| unprovided.next().unwrap())
            }
        }
    };
}

tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 6 1);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 6 2);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 6 3);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 6 4);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 6 5);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 6 6);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 5 1);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 5 2);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 5 3);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 5 4);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 5 5);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 4 1);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 4 2);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 4 3);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 4 4);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 3 1);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 3 2);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 3 3);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 2 1);
tensor_ref_impl!(unsafe impl TensorRef for TensorIndex 2 2);

#[test]
fn dimensionality_reduction() {
    use crate::tensors::Tensor;
    use crate::tensors::views::TensorView;
    let tensor = Tensor::from([("batch", 2), ("row", 2), ("column", 2)], vec![
        0, 1,
        2, 3,

        4, 5,
        6, 7
    ]);
    // selects second 2x2
    let matrix = TensorView::from(TensorIndex::from(&tensor, [("batch", 1)]));
    assert_eq!(matrix.shape(), [("row", 2), ("column", 2)]);
    assert_eq!(matrix, Tensor::from([("row", 2), ("column", 2)], vec![
        4, 5,
        6, 7
    ]));
    // selects first column
    let vector = TensorView::from(TensorIndex::from(matrix.source(), [("column", 0)]));
    assert_eq!(vector.shape(), [("row", 2)]);
    assert_eq!(vector, Tensor::from([("row", 2)], vec![
        4, 6
    ]));

    // selects second row of data
    let matrix = TensorView::from(TensorIndex::from(&tensor, [("row", 1)]));
    assert_eq!(matrix.shape(), [("batch", 2), ("column", 2)]);
    assert_eq!(matrix, Tensor::from([("batch", 2), ("column", 2)], vec![
        2, 3,
        6, 7
    ]));

    // selects second column of data
    let matrix = TensorView::from(TensorIndex::from(&tensor, [("column", 1)]));
    assert_eq!(matrix.shape(), [("batch", 2), ("row", 2)]);
    assert_eq!(matrix, Tensor::from([("batch", 2), ("row", 2)], vec![
        1, 3,
        5, 7
    ]));
    // selects first batch
    let vector = TensorView::from(TensorIndex::from(matrix.source(), [("batch", 0)]));
    assert_eq!(vector.shape(), [("row", 2)]);
    assert_eq!(vector, Tensor::from([("row", 2)], vec![
        1, 3,
    ]));
}
