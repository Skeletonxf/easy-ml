/*!
 * Tensor operations
 */

 use crate::numeric::{Numeric, NumericRef};
 use crate::tensors::Tensor;
 use crate::tensors::indexing::TensorAccess;
 use crate::tensors::views::TensorRef;
 use crate::tensors::Dimension;

/**
 * Helper struct of two equally sized tensor accesses.
 */
struct TensorAccessElementwise<T, S1, S2, const D: usize> {
    shape: [(Dimension, usize); D],
    left: TensorAccess<T, S1, D>,
    right: TensorAccess<T, S2, D>,
}

fn tensor_access_elementwise<T, S1, S2, const D: usize>(left: S1, right: S2) -> TensorAccessElementwise<T, S1, S2, D>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    let left_shape = left.view_shape();
    let left_dimensions = left_shape.map(|(dimension, _length)| dimension);
    let left_access = TensorAccess::try_from(left, left_dimensions).unwrap(); // FIXME TODO: Should have a constructor which just uses the memory order and removes need to handle failure
    let right_access = TensorAccess::try_from(right, left_dimensions).unwrap(); // FIXME
    // If right_access can be created from the dimension order of the left tensor, we can
    // iterate elementwise through both using identical indexes on our TensorAccess structs.
    TensorAccessElementwise {
        shape: left_shape,
        left: left_access,
        right: right_access,
    }
}

fn tensor_access_addition_0<T, S1, S2>(tensors: TensorAccessElementwise<T, S1, S2, 0>) -> Tensor<T, 0>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S1: TensorRef<T, 0>,
    S2: TensorRef<T, 0>,
{
    Tensor::from_scalar(
        tensors.left.get_reference([]) + tensors.right.get_reference([])
    )
}

fn tensor_access_addition_1<T, S1, S2>(tensors: TensorAccessElementwise<T, S1, S2, 1>) -> Tensor<T, 1>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S1: TensorRef<T, 1>,
    S2: TensorRef<T, 1>,
{
    let length = tensors.shape[0].1;
    let mut data = vec![T::zero(); length];
    for (i, elem) in data.iter_mut().enumerate() {
        *elem = tensors.left.get_reference([i]) + tensors.right.get_reference([i]);
    }
    Tensor::new(data, tensors.shape)
}

fn tensor_access_addition_2<T, S1, S2>(tensors: TensorAccessElementwise<T, S1, S2, 2>) -> Tensor<T, 2>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S1: TensorRef<T, 2>,
    S2: TensorRef<T, 2>,
{
    let length = tensors.shape.iter().map(|(_dimension, length)| length).product();
    let mut data = vec![T::zero(); length];
    let mut index = [0; 2];
    for elem in data.iter_mut() {
        *elem = tensors.left.get_reference(index) + tensors.right.get_reference(index);
        index[1] += 1;
        for d in (1..2).rev() {
            if index[d] == tensors.shape[d].1 {
                // ran to end of this dimension with our index
                // In the 2D case, we finished indexing through every column in the row,
                // and it's now time to move onto the next row.
                index[d] = 0;
                index[d - 1] += 1;
            }
        }
    }
    Tensor::new(data, tensors.shape)
}

fn tensor_access_addition<T, S1, S2, const D: usize>(tensors: TensorAccessElementwise<T, S1, S2, D>) -> Tensor<T, D>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    let length = tensors.shape.iter().map(|(_dimension, length)| length).product();
    let mut data = vec![T::zero(); length];
    let mut index = [0; D];
    for elem in data.iter_mut() {
        *elem = tensors.left.get_reference(index) + tensors.right.get_reference(index);
        if D > 0 {
            // Increment index of final dimension. In the 2D case, we iterate through a row by
            // incrementing through every column index.
            index[D - 1] += 1;
            for d in (1..D).rev() {
                if index[d] == tensors.shape[d].1 {
                    // ran to end of this dimension with our index
                    // In the 2D case, we finished indexing through every column in the row,
                    // and it's now time to move onto the next row.
                    index[d] = 0;
                    index[d - 1] += 1;
                }
            }
            // Since we calculated the length from the shape, we will never go past the final
            // index of the shape and therefore don't need to handle it
            // TODO: This should be extracted into a TensorAccess iterator implementation
        }
    }
    Tensor::new(data, tensors.shape)
}
