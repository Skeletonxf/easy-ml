/*!
 * Tensor operations
 */

use crate::numeric::{Numeric, NumericRef};
use crate::tensors::indexing::{IndexOrderReferenceIterator, TensorAccess};
use crate::tensors::views::{TensorIndex, TensorRef, TensorView};
use crate::tensors::{Dimension, Tensor};

use std::ops::{Add, Mul, Sub};

// Common tensor equality definition (list of dimension names must match, and elements must match)
#[inline]
pub(crate) fn tensor_equality<T, S1, S2, const D: usize>(left: &S1, right: &S2) -> bool
where
    T: PartialEq,
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    left.view_shape() == right.view_shape()
        && IndexOrderReferenceIterator::from(left)
            .zip(IndexOrderReferenceIterator::from(right))
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
    left_access
        .index_order_reference_iter()
        .zip(right_access.index_order_reference_iter())
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

/**
 * Two TensorViews are equal if they have an equal shape and all their elements are equal.
 * Differences in their source types are ignored.
 */
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

/**
 * A Tensor and a TensorView can be compared for equality. The tensors are equal if they
 * have an equal shape and all their elements are equal.
 */
impl<T, S, const D: usize> PartialEq<TensorView<T, S, D>> for Tensor<T, D>
where
    T: PartialEq,
    S: TensorRef<T, D>,
{
    fn eq(&self, other: &TensorView<T, S, D>) -> bool {
        tensor_equality(self, other.source_ref())
    }
}

/**
 * A TensorView and a Tensor can be compared for equality. The tensors are equal if they
 * have an equal shape and all their elements are equal.
 */
impl<T, S, const D: usize> PartialEq<Tensor<T, D>> for TensorView<T, S, D>
where
    T: PartialEq,
    S: TensorRef<T, D>,
{
    fn eq(&self, other: &Tensor<T, D>) -> bool {
        tensor_equality(self.source_ref(), other)
    }
}

/**
 * Similarity comparisons. This is a looser comparison than [PartialEq](PartialEq),
 * but anything which returns true for [PartialEq::eq](PartialEq::eq) will also return true
 * for [Similar::similar].
 *
 * This trait is sealed and cannot be implemented for types outside this crate.
 */
pub trait Similar<Rhs: ?Sized = Self>: private::Sealed {
    /**
     * Tests if two values are similar. This is a looser comparison than [PartialEq](PartialEq),
     * but anything which is PartialEq is also similar.
     */
    #[must_use]
    fn similar(&self, other: &Rhs) -> bool;
}

mod private {
    use crate::tensors::views::{TensorRef, TensorView};
    use crate::tensors::Tensor;

    pub trait Sealed<Rhs: ?Sized = Self> {}

    impl<T: PartialEq, const D: usize> Sealed for Tensor<T, D> {}
    impl<T, S1, S2, const D: usize> Sealed<TensorView<T, S2, D>> for TensorView<T, S1, D>
    where
        T: PartialEq,
        S1: TensorRef<T, D>,
        S2: TensorRef<T, D>,
    {
    }
    impl<T, S, const D: usize> Sealed<TensorView<T, S, D>> for Tensor<T, D>
    where
        T: PartialEq,
        S: TensorRef<T, D>,
    {
    }
    impl<T, S, const D: usize> Sealed<Tensor<T, D>> for TensorView<T, S, D>
    where
        T: PartialEq,
        S: TensorRef<T, D>,
    {
    }
}

/**
 * Two Tensors are similar if they have the same **set** of dimension names and lengths even
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
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * use easy_ml::tensors::operations::Similar;
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
 * assert!(one.similar(&one));
 * assert!(two.similar(&two));
 * assert!(three.similar(&three));
 * assert!(one.similar(&two)); // similar, dimension order is not the same
 * assert!(!one.similar(&three)); // elementwise data is same, but dimensions are not equal
 * assert!(!two.similar(&three)); // dimension lengths are not the same, and elementwise data is not the same
 * ```
 */
impl<T: PartialEq, const D: usize> Similar for Tensor<T, D> {
    fn similar(&self, other: &Tensor<T, D>) -> bool {
        tensor_similarity(self, other)
    }
}

/**
 * Two TensorViews are similar if they have the same **set** of dimension names and
 * lengths even if two shapes are not in the same order, and all their elements are equal
 * when comparing both tensors via the same dimension ordering.
 * Differences in their source types are ignored.
 */
impl<T, S1, S2, const D: usize> Similar<TensorView<T, S2, D>> for TensorView<T, S1, D>
where
    T: PartialEq,
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    fn similar(&self, other: &TensorView<T, S2, D>) -> bool {
        tensor_similarity(self.source_ref(), other.source_ref())
    }
}

/**
 * A Tensor and a TensorView can be compared for similarity. The tensors are similar if they
 * have the same **set** of dimension names and lengths even if two shapes are not in the same
 * order, and all their elements are equal when comparing both tensors via the same dimension
 * ordering.
 */
impl<T, S, const D: usize> Similar<TensorView<T, S, D>> for Tensor<T, D>
where
    T: PartialEq,
    S: TensorRef<T, D>,
{
    fn similar(&self, other: &TensorView<T, S, D>) -> bool {
        tensor_similarity(self, other.source_ref())
    }
}

/**
 * A TensorView and a Tensor can be compared for similarity. The tensors are similar if they
 * have the same **set** of dimension names and lengths even if two shapes are not in the same
 * order, and all their elements are equal when comparing both tensors via the same dimension
 * ordering.
 */
impl<T, S, const D: usize> Similar<Tensor<T, D>> for TensorView<T, S, D>
where
    T: PartialEq,
    S: TensorRef<T, D>,
{
    fn similar(&self, other: &Tensor<T, D>) -> bool {
        tensor_similarity(self.source_ref(), other)
    }
}

#[track_caller]
#[inline]
fn assert_same_dimensions<const D: usize>(
    left_shape: [(Dimension, usize); D],
    right_shape: [(Dimension, usize); D],
) {
    if left_shape != right_shape {
        panic!(
            "Dimensions of left and right tensors are not the same: (left: {:?}, right: {:?})",
            left_shape, right_shape
        );
    }
}

#[track_caller]
#[inline]
fn tensor_view_addition_iter<'l, 'r, T, S1, S2, const D: usize>(
    left_iter: S1,
    left_shape: [(Dimension, usize); D],
    right_iter: S2,
    right_shape: [(Dimension, usize); D],
) -> Tensor<T, D>
where
    T: Numeric,
    T: 'l,
    T: 'r,
    for<'a> &'a T: NumericRef<T>,
    S1: Iterator<Item = &'l T>,
    S2: Iterator<Item = &'r T>,
{
    assert_same_dimensions(left_shape, right_shape);
    // LxM + LxM -> LxM
    Tensor::from(
        left_shape,
        left_iter.zip(right_iter).map(|(x, y)| x + y).collect(),
    )
}

#[track_caller]
#[inline]
fn tensor_view_subtraction_iter<'l, 'r, T, S1, S2, const D: usize>(
    left_iter: S1,
    left_shape: [(Dimension, usize); D],
    right_iter: S2,
    right_shape: [(Dimension, usize); D],
) -> Tensor<T, D>
where
    T: Numeric,
    T: 'l,
    T: 'r,
    for<'a> &'a T: NumericRef<T>,
    S1: Iterator<Item = &'l T>,
    S2: Iterator<Item = &'r T>,
{
    assert_same_dimensions(left_shape, right_shape);
    // LxM - LxM -> LxM
    Tensor::from(
        left_shape,
        left_iter.zip(right_iter).map(|(x, y)| x - y).collect(),
    )
}

/**
 * Computes the dot product (also known as scalar product) on two equal length iterators,
 * yielding a scalar which is the sum of the products of each pair in the iterators.
 *
 * https://en.wikipedia.org/wiki/Dot_product
 */
#[inline]
fn scalar_product<'l, 'r, T, S1, S2>(left_iter: S1, right_iter: S2) -> T
where
    T: Numeric,
    T: 'l,
    T: 'r,
    for<'a> &'a T: NumericRef<T>,
    S1: Iterator<Item = &'l T>,
    S2: Iterator<Item = &'r T>,
{
    left_iter.zip(right_iter).map(|(x, y)| x * y).sum()
}

#[track_caller]
#[inline]
fn tensor_view_vector_product_iter<'l, 'r, T, S1, S2>(
    left_iter: S1,
    left_shape: [(Dimension, usize); 1],
    right_iter: S2,
    right_shape: [(Dimension, usize); 1],
) -> T
where
    T: Numeric,
    T: 'l,
    T: 'r,
    for<'a> &'a T: NumericRef<T>,
    S1: Iterator<Item = &'l T>,
    S2: Iterator<Item = &'r T>,
{
    if left_shape[0].1 != right_shape[0].1 {
        panic!(
            "Dimension lengths of left and right tensors are not the same: (left: {:?}, right: {:?})",
            left_shape, right_shape
        );
    }
    // [a,b,c] . [d,e,f] -> a*d + b*e + c*f
    scalar_product::<T, S1, S2>(left_iter, right_iter)
}

#[track_caller]
#[inline]
fn tensor_view_matrix_product<T, S1, S2>(left: S1, right: S2) -> Tensor<T, 2>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S1: TensorRef<T, 2>,
    S2: TensorRef<T, 2>,
{
    if left.view_shape()[1].1 != right.view_shape()[0].1 {
        panic!(
            "Mismatched tensors, left is {:?}, right is {:?}, * is only defined for MxN * NxL dimension lengths",
            left.view_shape(), right.view_shape()
        );
    }
    // LxM * MxN -> LxN
    // [a,b,c; d,e,f] * [g,h; i,j; k,l] -> [a*g+b*i+c*k, a*h+b*j+c*l; d*g+e*i+f*k, d*h+e*j+f*l]
    // Matrix multiplication gives us another Matrix where each element [i,j] is the dot product
    // of the i'th row in the left matrix and the j'th column in the right matrix.
    let mut tensor = Tensor::empty([left.view_shape()[0], right.view_shape()[1]], T::zero());
    for ([i, j], x) in tensor
        .source_order_mut()
        .index_order_reference_mut_iter()
        .with_index()
    {
        // Select the i'th row in the left tensor to give us a vector
        let left = TensorAccess::from_source_order(TensorIndex::from(
            &left,
            [(left.view_shape()[0].0, i)],
        ));
        // Select the j'th column in the right tensor to give us a vector
        let right = TensorAccess::from_source_order(TensorIndex::from(
            &right,
            [(right.view_shape()[1].0, j)],
        ));
        // Since we checked earlier that we have MxN * NxL these two vectors have the same length.
        *x = scalar_product::<T, _, _>(
            left.index_order_reference_iter(),
            right.index_order_reference_iter(),
        )
    }
    tensor
}

#[test]
fn test_matrix_product() {
    #[rustfmt::skip]
    let left = Tensor::from([("r", 2), ("c", 3)], vec![
        1, 2, 3,
        4, 5, 6
    ]);
    #[rustfmt::skip]
    let right = Tensor::from([("r", 3), ("c", 2)], vec![
        10, 11,
        12, 13,
        14, 15
    ]);
    let result = tensor_view_matrix_product::<i32, _, _>(left, right);
    #[rustfmt::skip]
    assert_eq!(
        result,
        Tensor::from(
            [("r", 2), ("c", 2)],
            vec![
                1 * 10 + 2 * 12 + 3 * 14, 1 * 11 + 2 * 13 + 3 * 15,
                4 * 10 + 5 * 12 + 6 * 14, 4 * 11 + 5 * 13 + 6 * 15
            ]
        )
    );
}

// Tensor multiplication (⊗) gives another Tensor where each element [i,j,k] is the dot product of
// the [i,j,*] vector in the left tensor and the [*,j,k] vector in the right tensor???

macro_rules! tensor_view_reference_tensor_view_reference_operation_iter {
    (impl $op:tt for TensorView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2, const D: usize> $op<&TensorView<T, S2, D>> for &TensorView<T, S1, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: TensorRef<T, D>,
            S2: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &TensorView<T, S2, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.index_order_reference_iter(),
                    self.shape(),
                    rhs.index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_view_reference_tensor_view_reference_operation {
    (impl $op:tt for TensorView $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<&TensorView<T, S2, $d>> for &TensorView<T, S1, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: TensorRef<T, $d>,
            S2: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &TensorView<T, S2, $d>) -> Self::Output {
                $implementation::<T, _, _>(self.source_ref(), rhs.source_ref())
            }
        }
    };
}

tensor_view_reference_tensor_view_reference_operation_iter!(impl Add for TensorView { fn add } tensor_view_addition_iter "Elementwise addition for two referenced tensor views");
tensor_view_reference_tensor_view_reference_operation_iter!(impl Sub for TensorView { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for two referenced tensor views");
tensor_view_reference_tensor_view_reference_operation!(impl Mul for TensorView 2 { fn mul } tensor_view_matrix_product "Matrix multiplication of two referenced 2-dimensional tensors");

macro_rules! tensor_view_reference_tensor_view_value_operation_iter {
    (impl $op:tt for TensorView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2, const D: usize> $op<TensorView<T, S2, D>> for &TensorView<T, S1, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: TensorRef<T, D>,
            S2: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: TensorView<T, S2, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.index_order_reference_iter(),
                    self.shape(),
                    rhs.index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_view_reference_tensor_view_value_operation {
    (impl $op:tt for TensorView $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<TensorView<T, S2, $d>> for &TensorView<T, S1, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: TensorRef<T, $d>,
            S2: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: TensorView<T, S2, $d>) -> Self::Output {
                $implementation::<T, _, _>(self.source_ref(), rhs.source_ref())
            }
        }
    };
}

tensor_view_reference_tensor_view_value_operation_iter!(impl Add for TensorView { fn add } tensor_view_addition_iter "Elementwise addition for two tensor views with one referenced");
tensor_view_reference_tensor_view_value_operation_iter!(impl Sub for TensorView { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for two tensor views with one referenced");
tensor_view_reference_tensor_view_value_operation!(impl Mul for TensorView 2 { fn mul } tensor_view_matrix_product "Matrix multiplication of two 2-dimensional tensors with one referenced");

macro_rules! tensor_view_value_tensor_view_reference_operation_iter {
    (impl $op:tt for TensorView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2, const D: usize> $op<&TensorView<T, S2, D>> for TensorView<T, S1, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: TensorRef<T, D>,
            S2: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &TensorView<T, S2, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.index_order_reference_iter(),
                    self.shape(),
                    rhs.index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_view_value_tensor_view_reference_operation {
    (impl $op:tt for TensorView $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<&TensorView<T, S2, $d>> for TensorView<T, S1, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: TensorRef<T, $d>,
            S2: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &TensorView<T, S2, $d>) -> Self::Output {
                $implementation::<T, _, _>(self.source_ref(), rhs.source_ref())
            }
        }
    };
}

tensor_view_value_tensor_view_reference_operation_iter!(impl Add for TensorView { fn add } tensor_view_addition_iter "Elementwise addition for two tensor views with one referenced");
tensor_view_value_tensor_view_reference_operation_iter!(impl Sub for TensorView { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for two tensor views with one referenced");
tensor_view_value_tensor_view_reference_operation!(impl Mul for TensorView 2 { fn mul } tensor_view_matrix_product "Matrix multiplication of two 2-dimensional tensors with one referenced");

macro_rules! tensor_view_value_tensor_view_value_operation_iter {
    (impl $op:tt for TensorView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2, const D: usize> $op<TensorView<T, S2, D>> for TensorView<T, S1, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: TensorRef<T, D>,
            S2: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: TensorView<T, S2, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.index_order_reference_iter(),
                    self.shape(),
                    rhs.index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_view_value_tensor_view_value_operation {
    (impl $op:tt for TensorView $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<TensorView<T, S2, $d>> for TensorView<T, S1, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: TensorRef<T, $d>,
            S2: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: TensorView<T, S2, $d>) -> Self::Output {
                $implementation::<T, _, _>(self.source_ref(), rhs.source_ref())
            }
        }
    };
}

tensor_view_value_tensor_view_value_operation_iter!(impl Add for TensorView { fn add } tensor_view_addition_iter "Elementwise addition for two tensor views");
tensor_view_value_tensor_view_value_operation_iter!(impl Sub for TensorView { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for two tensor views");
tensor_view_value_tensor_view_value_operation!(impl Mul for TensorView 2 { fn mul } tensor_view_matrix_product "Matrix multiplication of two 2-dimensional tensors");

macro_rules! tensor_view_reference_tensor_reference_operation_iter {
    (impl $op:tt for TensorView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S, const D: usize> $op<&Tensor<T, D>> for &TensorView<T, S, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Tensor<T, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.index_order_reference_iter(),
                    self.shape(),
                    rhs.direct_index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_view_reference_tensor_reference_operation {
    (impl $op:tt for TensorView $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&Tensor<T, $d>> for &TensorView<T, S, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Tensor<T, $d>) -> Self::Output {
                $implementation::<T, _, _>(self.source_ref(), rhs)
            }
        }
    };
}

tensor_view_reference_tensor_reference_operation_iter!(impl Add for TensorView { fn add } tensor_view_addition_iter "Elementwise addition for a referenced tensor view and a referenced tensor");
tensor_view_reference_tensor_reference_operation_iter!(impl Sub for TensorView { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for a referenced tensor view and a referenced tensor");
tensor_view_reference_tensor_reference_operation!(impl Mul for TensorView 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for a 2-dimensional referenced tensor view and a referenced tensor");

macro_rules! tensor_view_reference_tensor_value_operation_iter {
    (impl $op:tt for TensorView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S, const D: usize> $op<Tensor<T, D>> for &TensorView<T, S, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Tensor<T, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.index_order_reference_iter(),
                    self.shape(),
                    rhs.direct_index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_view_reference_tensor_value_operation {
    (impl $op:tt for TensorView $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<Tensor<T, $d>> for &TensorView<T, S, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Tensor<T, $d>) -> Self::Output {
                $implementation::<T, _, _>(self.source_ref(), rhs)
            }
        }
    };
}

tensor_view_reference_tensor_value_operation_iter!(impl Add for TensorView { fn add } tensor_view_addition_iter "Elementwise addition for a referenced tensor view and a tensor");
tensor_view_reference_tensor_value_operation_iter!(impl Sub for TensorView { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for a referenced tensor view and a tensor");
tensor_view_reference_tensor_value_operation!(impl Mul for TensorView 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for a 2-dimensional referenced tensor view and a tensor");

macro_rules! tensor_view_value_tensor_reference_operation_iter {
    (impl $op:tt for TensorView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S, const D: usize> $op<&Tensor<T, D>> for TensorView<T, S, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Tensor<T, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.index_order_reference_iter(),
                    self.shape(),
                    rhs.direct_index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_view_value_tensor_reference_operation {
    (impl $op:tt for TensorView $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&Tensor<T, $d>> for TensorView<T, S, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Tensor<T, $d>) -> Self::Output {
                $implementation::<T, _, _>(self.source_ref(), rhs)
            }
        }
    };
}

tensor_view_value_tensor_reference_operation_iter!(impl Add for TensorView { fn add } tensor_view_addition_iter "Elementwise addition for a tensor view and a referenced tensor");
tensor_view_value_tensor_reference_operation_iter!(impl Sub for TensorView { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for a tensor view and a referenced tensor");
tensor_view_value_tensor_reference_operation!(impl Mul for TensorView 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for a 2-dimensional tensor view and a referenced tensor");

macro_rules! tensor_view_value_tensor_value_operation_iter {
    (impl $op:tt for TensorView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S, const D: usize> $op<Tensor<T, D>> for TensorView<T, S, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Tensor<T, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.index_order_reference_iter(),
                    self.shape(),
                    rhs.direct_index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_view_value_tensor_value_operation {
    (impl $op:tt for TensorView $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<Tensor<T, $d>> for TensorView<T, S, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Tensor<T, $d>) -> Self::Output {
                $implementation::<T, _, _>(self.source_ref(), rhs)
            }
        }
    };
}

tensor_view_value_tensor_value_operation_iter!(impl Add for TensorView { fn add } tensor_view_addition_iter "Elementwise addition for a tensor view and a tensor");
tensor_view_value_tensor_value_operation_iter!(impl Sub for TensorView { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for a tensor view and a tensor");
tensor_view_value_tensor_value_operation!(impl Mul for TensorView 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for a 2-dimensional tensor view and a tensor");

macro_rules! tensor_reference_tensor_view_reference_operation_iter {
    (impl $op:tt for Tensor { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S, const D: usize> $op<&TensorView<T, S, D>> for &Tensor<T, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &TensorView<T, S, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.direct_index_order_reference_iter(),
                    self.shape(),
                    rhs.index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_reference_tensor_view_reference_operation {
    (impl $op:tt for Tensor $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&TensorView<T, S, $d>> for &Tensor<T, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &TensorView<T, S, $d>) -> Self::Output {
                $implementation::<T, _, _>(self, rhs.source_ref())
            }
        }
    };
}

tensor_reference_tensor_view_reference_operation_iter!(impl Add for Tensor { fn add } tensor_view_addition_iter "Elementwise addition for a referenced tensor and a referenced tensor view");
tensor_reference_tensor_view_reference_operation_iter!(impl Sub for Tensor { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for a referenced tensor and a referenced tensor view");
tensor_reference_tensor_view_reference_operation!(impl Mul for Tensor 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for a 2-dimensional referenced tensor and a referenced tensor view");

macro_rules! tensor_reference_tensor_view_value_operation_iter {
    (impl $op:tt for Tensor { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S, const D: usize> $op<TensorView<T, S, D>> for &Tensor<T, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: TensorView<T, S, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.direct_index_order_reference_iter(),
                    self.shape(),
                    rhs.index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_reference_tensor_view_value_operation {
    (impl $op:tt for Tensor $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<TensorView<T, S, $d>> for &Tensor<T, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: TensorView<T, S, $d>) -> Self::Output {
                $implementation::<T, _, _>(self, rhs.source_ref())
            }
        }
    };
}

tensor_reference_tensor_view_value_operation_iter!(impl Add for Tensor { fn add } tensor_view_addition_iter "Elementwise addition for a referenced tensor and a tensor view");
tensor_reference_tensor_view_value_operation_iter!(impl Sub for Tensor { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for a referenced tensor and a tensor view");
tensor_reference_tensor_view_value_operation!(impl Mul for Tensor 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for a 2-dimensional referenced tensor and a tensor view");

macro_rules! tensor_value_tensor_view_reference_operation_iter {
    (impl $op:tt for Tensor { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S, const D: usize> $op<&TensorView<T, S, D>> for Tensor<T, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &TensorView<T, S, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.direct_index_order_reference_iter(),
                    self.shape(),
                    rhs.index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_value_tensor_view_reference_operation {
    (impl $op:tt for Tensor $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&TensorView<T, S, $d>> for Tensor<T, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &TensorView<T, S, $d>) -> Self::Output {
                $implementation::<T, _, _>(self, rhs.source_ref())
            }
        }
    };
}

tensor_value_tensor_view_reference_operation_iter!(impl Add for Tensor { fn add } tensor_view_addition_iter "Elementwise addition for a tensor and a referenced tensor view");
tensor_value_tensor_view_reference_operation_iter!(impl Sub for Tensor { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for a tensor and a referenced tensor view");
tensor_value_tensor_view_reference_operation!(impl Mul for Tensor 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for a 2-dimensional tensor and a referenced tensor view");

macro_rules! tensor_value_tensor_view_value_operation_iter {
    (impl $op:tt for Tensor { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S, const D: usize> $op<TensorView<T, S, D>> for Tensor<T, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, D>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: TensorView<T, S, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.direct_index_order_reference_iter(),
                    self.shape(),
                    rhs.index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_value_tensor_view_value_operation {
    (impl $op:tt for Tensor $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<TensorView<T, S, $d>> for Tensor<T, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: TensorRef<T, $d>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: TensorView<T, S, $d>) -> Self::Output {
                $implementation::<T, _, _>(self, rhs.source_ref())
            }
        }
    };
}

tensor_value_tensor_view_value_operation_iter!(impl Add for Tensor { fn add } tensor_view_addition_iter "Elementwise addition for a tensor and a tensor view");
tensor_value_tensor_view_value_operation_iter!(impl Sub for Tensor { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for a tensor and a tensor view");
tensor_value_tensor_view_value_operation!(impl Mul for Tensor 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for a 2-dimensional tensor and a tensor view");

macro_rules! tensor_reference_tensor_reference_operation_iter {
    (impl $op:tt for Tensor { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, const D: usize> $op<&Tensor<T, D>> for &Tensor<T, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Tensor<T, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.direct_index_order_reference_iter(),
                    self.shape(),
                    rhs.direct_index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_reference_tensor_reference_operation {
    (impl $op:tt for Tensor $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<&Tensor<T, $d>> for &Tensor<T, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Tensor<T, $d>) -> Self::Output {
                $implementation::<T, _, _>(self, rhs)
            }
        }
    };
}

tensor_reference_tensor_reference_operation_iter!(impl Add for Tensor { fn add } tensor_view_addition_iter "Elementwise addition for two referenced tensors");
tensor_reference_tensor_reference_operation_iter!(impl Sub for Tensor { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for two referenced tensors");
tensor_reference_tensor_reference_operation!(impl Mul for Tensor 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for two 2-dimensional referenced tensors");

macro_rules! tensor_reference_tensor_value_operation_iter {
    (impl $op:tt for Tensor { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, const D: usize> $op<Tensor<T, D>> for &Tensor<T, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Tensor<T, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.direct_index_order_reference_iter(),
                    self.shape(),
                    rhs.direct_index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_reference_tensor_value_operation {
    (impl $op:tt for Tensor $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<Tensor<T, $d>> for &Tensor<T, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Tensor<T, $d>) -> Self::Output {
                $implementation::<T, _, _>(self, rhs)
            }
        }
    };
}

tensor_reference_tensor_value_operation_iter!(impl Add for Tensor { fn add } tensor_view_addition_iter "Elementwise addition for two tensors with one referenced");
tensor_reference_tensor_value_operation_iter!(impl Sub for Tensor { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for two tensors with one referenced");
tensor_reference_tensor_value_operation!(impl Mul for Tensor 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for two 2-dimensional tensors with one referenced");

macro_rules! tensor_value_tensor_reference_operation_iter {
    (impl $op:tt for Tensor { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, const D: usize> $op<&Tensor<T, D>> for Tensor<T, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Tensor<T, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.direct_index_order_reference_iter(),
                    self.shape(),
                    rhs.direct_index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_value_tensor_reference_operation {
    (impl $op:tt for Tensor $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<&Tensor<T, $d>> for Tensor<T, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Tensor<T, $d>) -> Self::Output {
                $implementation::<T, _, _>(self, rhs)
            }
        }
    };
}

tensor_value_tensor_reference_operation_iter!(impl Add for Tensor { fn add } tensor_view_addition_iter "Elementwise addition for two tensors with one referenced");
tensor_value_tensor_reference_operation_iter!(impl Sub for Tensor { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for two tensors with one referenced");
tensor_value_tensor_reference_operation!(impl Mul for Tensor 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for two 2-dimensional tensors with one referenced");

macro_rules! tensor_value_tensor_value_operation_iter {
    (impl $op:tt for Tensor { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, const D: usize> $op<Tensor<T, D>> for Tensor<T, D>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Tensor<T, D>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Tensor<T, D>) -> Self::Output {
                $implementation::<T, _, _, D>(
                    self.direct_index_order_reference_iter(),
                    self.shape(),
                    rhs.direct_index_order_reference_iter(),
                    rhs.shape(),
                )
            }
        }
    };
}

macro_rules! tensor_value_tensor_value_operation {
    (impl $op:tt for Tensor $d:literal { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<Tensor<T, $d>> for Tensor<T, $d>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Tensor<T, $d>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Tensor<T, $d>) -> Self::Output {
                $implementation::<T, _, _>(self, rhs)
            }
        }
    };
}

tensor_value_tensor_value_operation_iter!(impl Add for Tensor { fn add } tensor_view_addition_iter "Elementwise addition for two tensors");
tensor_value_tensor_value_operation_iter!(impl Sub for Tensor { fn sub } tensor_view_subtraction_iter "Elementwise subtraction for two tensors");
tensor_value_tensor_value_operation!(impl Mul for Tensor 2 { fn mul } tensor_view_matrix_product "Matrix multiplication for two 2-dimensional tensors");

#[test]
fn elementwise_addition_test_all_16_combinations() {
    fn tensor() -> Tensor<i8, 1> {
        Tensor::from([("a", 1)], vec![1])
    }
    fn tensor_view() -> TensorView<i8, Tensor<i8, 1>, 1> {
        TensorView::from(tensor())
    }
    let mut results = Vec::with_capacity(16);
    results.push(tensor() + tensor());
    results.push(tensor() + &tensor());
    results.push(&tensor() + tensor());
    results.push(&tensor() + &tensor());
    results.push(tensor_view() + tensor());
    results.push(tensor_view() + &tensor());
    results.push(&tensor_view() + tensor());
    results.push(&tensor_view() + &tensor());
    results.push(tensor() + tensor_view());
    results.push(tensor() + &tensor_view());
    results.push(&tensor() + tensor_view());
    results.push(&tensor() + &tensor_view());
    results.push(tensor_view() + tensor_view());
    results.push(tensor_view() + &tensor_view());
    results.push(&tensor_view() + tensor_view());
    results.push(&tensor_view() + &tensor_view());
    for total in results {
        assert_eq!(total.get(["a"]).get([0]), 2);
    }
}

#[test]
fn elementwise_addition_test() {
    let tensor_1: Tensor<i32, 2> = Tensor::from([("r", 2), ("c", 2)], vec![1, 2, 3, 4]);
    let tensor_2: Tensor<i32, 2> = Tensor::from([("r", 2), ("c", 2)], vec![3, 2, 8, 1]);
    let added: Tensor<i32, 2> = tensor_1 + tensor_2;
    assert_eq!(added, Tensor::from([("r", 2), ("c", 2)], vec![4, 4, 11, 5]));
}

#[should_panic]
#[test]
fn elementwise_addition_test_similar_not_matching() {
    let tensor_1: Tensor<i32, 2> = Tensor::from([("r", 2), ("c", 2)], vec![1, 2, 3, 4]);
    let tensor_2: Tensor<i32, 2> = Tensor::from([("c", 2), ("r", 2)], vec![3, 8, 2, 1]);
    let _: Tensor<i32, 2> = tensor_1 + tensor_2;
}

#[test]
fn matrix_multiplication_test_all_16_combinations() {
    #[rustfmt::skip]
    fn tensor_1() -> Tensor<i8, 2> {
        Tensor::from([("r", 2), ("c", 3)], vec![
            1, 2, 3,
            4, 5, 6
        ])
    }
    fn tensor_1_view() -> TensorView<i8, Tensor<i8, 2>, 2> {
        TensorView::from(tensor_1())
    }
    #[rustfmt::skip]
    fn tensor_2() -> Tensor<i8, 2> {
        Tensor::from([("a", 3), ("b", 2)], vec![
            1, 2,
            3, 4,
            5, 6
        ])
    }
    fn tensor_2_view() -> TensorView<i8, Tensor<i8, 2>, 2> {
        TensorView::from(tensor_2())
    }
    let mut results = Vec::with_capacity(16);
    results.push(tensor_1() * tensor_2());
    results.push(tensor_1() * &tensor_2());
    results.push(&tensor_1() * tensor_2());
    results.push(&tensor_1() * &tensor_2());
    results.push(tensor_1_view() * tensor_2());
    results.push(tensor_1_view() * &tensor_2());
    results.push(&tensor_1_view() * tensor_2());
    results.push(&tensor_1_view() * &tensor_2());
    results.push(tensor_1() * tensor_2_view());
    results.push(tensor_1() * &tensor_2_view());
    results.push(&tensor_1() * tensor_2_view());
    results.push(&tensor_1() * &tensor_2_view());
    results.push(tensor_1_view() * tensor_2_view());
    results.push(tensor_1_view() * &tensor_2_view());
    results.push(&tensor_1_view() * tensor_2_view());
    results.push(&tensor_1_view() * &tensor_2_view());
    for total in results {
        #[rustfmt::skip]
        assert_eq!(
            total,
            Tensor::from(
                [("r", 2), ("b", 2)],
                vec![
                    1 * 1 + 2 * 3 + 3 * 5, 1 * 2 + 2 * 4 + 3 * 6,
                    4 * 1 + 5 * 3 + 6 * 5, 4 * 2 + 5 * 4 + 6 * 6,
                ]
            )
        );
    }
}

impl<T> Tensor<T, 1>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
{
    pub(crate) fn scalar_product_less_generic<S>(&self, rhs: TensorView<T, S, 1>) -> T
    where
        S: TensorRef<T, 1>,
    {
        let left_shape = self.shape();
        let right_shape = rhs.shape();
        assert_same_dimensions(left_shape, right_shape);
        tensor_view_vector_product_iter::<T, _, _>(
            self.direct_index_order_reference_iter(),
            left_shape,
            rhs.index_order_reference_iter(),
            right_shape,
        )
    }
}

impl<T, S> TensorView<T, S, 1>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S: TensorRef<T, 1>,
{
    pub(crate) fn scalar_product_less_generic<S2>(&self, rhs: TensorView<T, S2, 1>) -> T
    where
        S2: TensorRef<T, 1>,
    {
        let left_shape = self.shape();
        let right_shape = rhs.shape();
        assert_same_dimensions(left_shape, right_shape);
        tensor_view_vector_product_iter::<T, _, _>(
            self.index_order_reference_iter(),
            left_shape,
            rhs.index_order_reference_iter(),
            right_shape,
        )
    }
}
