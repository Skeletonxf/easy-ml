/*!
 * Tensor operations
 */

use crate::numeric::{Numeric, NumericRef};
use crate::tensors::Tensor;
use crate::tensors::indexing::TensorAccess;
use crate::tensors::views::{TensorRef, TensorView};

// Common tensor equality definition (list of dimension names must match, and elements must match)
#[inline]
pub(crate) fn tensor_equality<T, S1, S2, const D: usize>(left: &S1, right: &S2) -> bool
where
    T: PartialEq,
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    left.view_shape() == right.view_shape()
        && TensorAccess::from_source_order(left).index_reference_iter()
            .zip(TensorAccess::from_source_order(right).index_reference_iter())
            .all(|(x, y)| x == y)
}

// Common tensor similarity definition (sets of dimensions must match, and elements using a
// common dimension order must match - one tensor could be transposed to make them equal if
// they are similar but not equal).
#[inline]
pub(crate) fn tensor_similarity<T, S1, S2, const D: usize>(left: &S1, right: &S2) -> bool
where
    T: PartialEq,
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    use crate::tensors::dimensions::names_of;
    let left_shape = left.view_shape();
    let access_order = names_of(&left_shape);
    let left_access = TensorAccess::from_source_order(left);
    let right_access = match TensorAccess::try_from(right, access_order) {
        Ok(right_access) => right_access,
        Err(_) => return false,
    };
    // left_access.shape() is equal to left_shape so no need to compute it
    if left_shape != right_access.shape() {
        return false;
    }
    left_access.index_reference_iter()
        .zip(right_access.index_reference_iter())
        .all(|(x, y)| x == y)
}

/**
 * Two Tensors are equal if they have an equal shape and all their elements are equal
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * let one = Tensor::from([("a", 2), ("b", 3)], vec![
 *     1, 2, 3,
 *     4, 5, 6
 * ]);
 * let two = Tensor::from([("b", 3), ("a", 2)], vec![
 *     1, 4,
 *     2, 5,
 *     3, 6
 * ]);
 * let three = Tensor::from([("b", 2), ("a", 3)], vec![
 *     1, 2, 3,
 *     4, 5, 6
 * ]);
 * assert_eq!(one, one);
 * assert_eq!(two, two);
 * assert_eq!(three, three);
 * assert_ne!(one, two); // similar, but dimension order is not the same
 * assert_eq!(one, two.transpose(["a", "b"])); // transposing b to the order of a makes it equal
 * assert_ne!(one, three); // elementwise data is same, but dimensions are not equal
 * assert_ne!(two, three); // dimensions are not equal, and elementwise data is not the same
 * ```
 */
impl<T: PartialEq, const D: usize> PartialEq for Tensor<T, D> {
    fn eq(&self, other: &Self) -> bool {
        tensor_equality(self, other)
    }
}

// /**
//  * Two TensorViews are equal if they have a similar shape and all their elements are equal
//  * when comparing both tensors via the same dimension order. Differences in their
//  * source types are ignored.
//  *
//  * A similar shape means both tensors have the same set of dimension name and lengths
//  * but the order of these pairs may be different.
//  *
//  * Elements are compared by iterating through the right most index of the left tensor
//  * and the corresponding index in the right tensor when [accessed](TensorAccess) using
//  * the dimension order of the left. When both tensors have their dimensions in the same
//  * order, this corresponds to the right most index of the right tensor as well, and will
//  * be an elementwise comparison.
//  *
//  * ```
//  * use easy_ml::tensors::Tensor;
//  * use easy_ml::tensors::views::TensorView;
//  * let one = Tensor::from([("a", 2), ("b", 3)], vec![
//  *    1, 2, 3,
//  *    4, 5, 6
//  * ]);
//  * let two = Tensor::from([("b", 3), ("a", 2)], vec![
//  *    1, 4,
//  *    2, 5,
//  *    3, 6
//  * ]);
//  * let three = Tensor::from([("b", 2), ("a", 3)], vec![
//  *     1, 2, 3,
//  *     4, 5, 6
//  * ]);
//  * let one = TensorView::from(&one);
//  * let two = TensorView::from(&two);
//  * let three = TensorView::from(&three);
//  * assert_eq!(one, two);
//  * assert_ne!(one, three);
//  * assert_ne!(two, three);
//  * assert_eq!(one, one);
//  * assert_eq!(two, two);
//  * assert_eq!(three, three);
//  * ```
//  */
impl<T, S1, S2, const D: usize> PartialEq<TensorView<T, S2, D>> for TensorView<T, S1, D>
where
    T: PartialEq,
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    fn eq(&self, other: &TensorView<T, S2, D>) -> bool {
        tensor_equality(self.source_ref(), other.source_ref())
    }
}

/*
 * Two Tensors are similar if they have the same set of dimension names and lengths even
 * if two shapes are not in the same order, and all their elements are equal
 * when comparing both tensors via the same dimension ordering.
 *
 * Elements are compared by iterating through the right most index of the left tensor
 * and the corresponding index in the right tensor when [accessed](TensorAccess) using
 * the dimension order of the left. When both tensors have their dimensions in the same
 * order, this corresponds to the right most index of the right tensor as well, and will
 * be an elementwise comparison.
 *
 * If two Tensors are similar, you can transpose one of them to the dimension order of the
 * other and they will be equal.
 */

// /**
//  * A Tensor and a TensorView are equal if they have a similar shape and all their
//  * elements are equal when comparing both tensors via the same dimension order.
//  *
//  * A similar shape means both tensors have the same set of dimension name and lengths
//  * but the order of these pairs may be different.
//  *
//  * Elements are compared by iterating through the right most index of the left tensor
//  * and the corresponding index in the right tensor when [accessed](TensorAccess) using
//  * the dimension order of the left. When both tensors have their dimensions in the same
//  * order, this corresponds to the right most index of the right tensor as well, and will
//  * be an elementwise comparison.
//  *
//  * ```
//  * use easy_ml::tensors::Tensor;
//  * use easy_ml::tensors::views::TensorView;
//  * let one = Tensor::from([("a", 2), ("b", 3)], vec![
//  *    1, 2, 3,
//  *    4, 5, 6
//  * ]);
//  * let two = Tensor::from([("b", 3), ("a", 2)], vec![
//  *    1, 4,
//  *    2, 5,
//  *    3, 6
//  * ]);
//  * let three = Tensor::from([("b", 2), ("a", 3)], vec![
//  *     1, 2, 3,
//  *     4, 5, 6
//  * ]);
//  * assert_eq!(one, TensorView::from(&two));
//  * assert_ne!(one, TensorView::from(&three));
//  * assert_ne!(two, TensorView::from(&three));
//  * assert_eq!(one, TensorView::from(&one));
//  * assert_eq!(two, TensorView::from(&two));
//  * assert_eq!(three, TensorView::from(&three));
//  */
impl<T, S, const D: usize> PartialEq<TensorView<T, S, D>> for Tensor<T, D>
where
    T: PartialEq,
    S: TensorRef<T, D>,
{
    fn eq(&self, other: &TensorView<T, S, D>) -> bool {
        tensor_equality(self, other.source_ref())
    }
}

// /**
//  * A TensorView and a Tensor are equal if they have a similar shape and all their
//  * elements are equal when comparing both tensors via the same dimension order.
//  *
//  * A similar shape means both tensors have the same set of dimension name and lengths
//  * but the order of these pairs may be different.
//  *
//  * Elements are compared by iterating through the right most index of the left tensor
//  * and the corresponding index in the right tensor when [accessed](TensorAccess) using
//  * the dimension order of the left. When both tensors have their dimensions in the same
//  * order, this corresponds to the right most index of the right tensor as well, and will
//  * be an elementwise comparison.
//  *
//  * ```
//  * use easy_ml::tensors::Tensor;
//  * use easy_ml::tensors::views::TensorView;
//  * let one = Tensor::from([("a", 2), ("b", 3)], vec![
//  *    1, 2, 3,
//  *    4, 5, 6
//  * ]);
//  * let two = Tensor::from([("b", 3), ("a", 2)], vec![
//  *    1, 4,
//  *    2, 5,
//  *    3, 6
//  * ]);
//  * let three = Tensor::from([("b", 2), ("a", 3)], vec![
//  *     1, 2, 3,
//  *     4, 5, 6
//  * ]);
//  * assert_eq!(TensorView::from(&one), two);
//  * assert_ne!(TensorView::from(&one), three);
//  * assert_ne!(TensorView::from(&two), three);
//  * assert_eq!(TensorView::from(&one), one);
//  * assert_eq!(TensorView::from(&two), two);
//  * assert_eq!(TensorView::from(&three), three);
//  * ```
//  */
impl<T, S, const D: usize> PartialEq<Tensor<T, D>> for TensorView<T, S, D>
where
    T: PartialEq,
    S: TensorRef<T, D>,
{
    fn eq(&self, other: &Tensor<T, D>) -> bool {
        tensor_equality(self.source_ref(), other)
    }
}

fn tensor_addition<T, S1, S2, const D: usize>(left: S1, right: S2) -> Tensor<T, D>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    if left.view_shape() != right.view_shape() {
        panic!(
            "Dimensions of left and right tensors are not the same: (left: {:?}, right: {:?})",
            left.view_shape(),
            right.view_shape()
        );
    }
    Tensor::from(
        left.view_shape(),
        TensorAccess::from_source_order(left).index_reference_iter()
            .zip(TensorAccess::from_source_order(right).index_reference_iter())
            .map(|(x, y)| x + y)
            .collect()
    )
}

#[test]
fn tmp_addition_test() {
    let tensor_1: Tensor<i32, 2> = Tensor::from([("r", 2), ("c", 2)], vec![ 1, 2, 3, 4 ]);
    let tensor_2: Tensor<i32, 2> = Tensor::from([("r", 2), ("c", 2)], vec![ 3, 2, 8, 1 ]);
    let added: Tensor<i32, 2> = tensor_addition::<i32, _, _, 2 >(tensor_1, tensor_2);
    assert_eq!(
        added,
        Tensor::from([("r", 2), ("c", 2)], vec![ 4, 4, 11, 5 ])
    );
}

#[should_panic]
#[test]
fn tmp_addition_test_similar_not_matching() {
    let tensor_1: Tensor<i32, 2> = Tensor::from([("r", 2), ("c", 2)], vec![ 1, 2, 3, 4 ]);
    let tensor_2: Tensor<i32, 2> = Tensor::from([("c", 2), ("r", 2)], vec![ 3, 8, 2, 1 ]);
    let _: Tensor<i32, 2> = tensor_addition::<i32, _, _, 2 >(tensor_1, tensor_2);
}
