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

fn tensor_addition<T, S1, S2, const D: usize>(left: S1, right: S2) -> Tensor<T, D>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    use crate::tensors::dimensions::names_of;
    let left_shape = left.view_shape();
    let access_order = names_of(&left_shape);
    let left_access = TensorAccess::from_source_order(left);
    let right_access = match TensorAccess::try_from(right, access_order) {
        Ok(right_access) => right_access,
        Err(error) => panic!(
            "Dimensions of left and right tensors do not have the same set of names (left: {:?}, right: {:?})",
            left_shape,
            error.actual
        ),
    };
    // If right_access can be created from the dimension order of the left tensor, we can
    // iterate elementwise through both using identical indexes on our TensorAccess structs
    // if each dimension length is the same.
    assert_eq!(
        left_access.shape(),
        right_access.shape(),
        "Dimensions of left and right do not have the same lengths along each dimension (left: {:?}, right: {:?})",
        left_access.shape(),
        right_access.shape(),
    );
    Tensor::from(
        left_shape,
        left_access
        .index_reference_iter()
        .zip(right_access.index_reference_iter())
        .map(|(x, y)| x + y)
        .collect()
    )
}

#[test]
fn tmp_addition_test() {
    let tensor_1: Tensor<i32, 2> = Tensor::from([("r", 2), ("c", 2)], vec![ 1, 2, 3, 4 ]);
    let tensor_2: Tensor<i32, 2> = Tensor::from([("c", 2), ("r", 2)], vec![ 3, 8, 2, 1 ]);
    let added: Tensor<i32, 2> = tensor_addition::<i32, _, _, 2 >(tensor_1, tensor_2);
    // TODO: Implement Eq
    assert_eq!(
        added
            .get(["r", "c"])
            .index_reference_iter()
            .cloned()
            .collect::<Vec<i32>>(),
        Tensor::from([("r", 2), ("c", 2)], vec![ 4, 4, 11, 5 ])
            .get(["r", "c"])
            .index_reference_iter()
            .cloned()
            .collect::<Vec<i32>>()
    );
}
