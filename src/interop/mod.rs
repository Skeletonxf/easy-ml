/*!
 * Interopability APIs between
 * [Matrix](crate::matrices::Matrix)/[MatrixView](crate::matrices::views::MatrixView) and
 * [Tensor](crate::tensors::Tensor)/[TensorView](crate::tensors::views::TensorView).
 */

use crate::matrices::views::{MatrixRef, NoInteriorMutability};
use crate::tensors::views::TensorRef;
use crate::tensors::Dimension;

use std::fmt;
use std::marker::PhantomData;

// TODO: MatrixRefTensor, instead of augmenting it with dimension names, we strip them.

/**
 * A wrapper around a Matrix type that implements TensorRef and can thus be used in a
 * [TensorView](crate::tensors::views::TensorView)
 *
 * ```
 * use easy_ml::matrices::Matrix;
 * use easy_ml::tensors::views::TensorView;
 * use easy_ml::interop::TensorRefMatrix;
 * let matrix = Matrix::from(vec![
 *     vec![ 1, 3, 5, 7 ],
 *     vec![ 2, 4, 6, 8 ]
 * ]);
 * // We can always unwrap here because we know a 2x4 matrix is a valid input
 * let tensor_view = TensorView::from(TensorRefMatrix::from(&matrix).unwrap());
 * assert_eq!(
 *     matrix.row_iter(1).eq(tensor_view.select([("row", 1)]).index_order_iter()),
 *     true
 * );
 * ```
 */
#[derive(Clone, Debug)]
pub struct TensorRefMatrix<T, S, N> {
    source: S,
    names: N,
    _type: PhantomData<T>,
}

/**
 * The first and second dimension name a Matrix type wrapped in a
 * [TensorRefMatrix](TensorRefMatrix) will report on its view shape. If you don't care what the
 * dimension names are, [RowAndColumn](RowAndColumn) can be used which will hardcode the
 * dimension names to "row" and "column" respectively.
 */
pub trait DimensionNames {
    fn names(&self) -> [Dimension; 2];
}

/**
 * A zero size DimensionNames type that always returns `["row", "column"]`.
 */
#[derive(Clone, Debug)]
pub struct RowAndColumn;

impl DimensionNames for RowAndColumn {
    fn names(&self) -> [Dimension; 2] {
        ["row", "column"]
    }
}

/**
 * Any array of two dimension names will implement DimensionNames returning those names in the
 * same order.
 */
impl DimensionNames for [Dimension; 2] {
    fn names(&self) -> [Dimension; 2] {
        self.clone()
    }
}

// TODO: Generalise this to be usable for all the tensor APIs even if we opt to prefer panicking
// for the cases where only programmer error would reach the error path.
/**
 * An error indicating failure to do something with a Tensor because the requested shape
 * is not valid.
 */
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidShapeError {
    shape: [(Dimension, usize); 2],
}

impl InvalidShapeError {
    fn is_valid(&self) -> bool {
        !crate::tensors::dimensions::has_duplicates(&self.shape)
            && self.shape[0].1 > 0
            && self.shape[1].1 > 0
    }
}

impl fmt::Display for InvalidShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dimensions must be at least 1x1 with unique names: {:?}",
            self.shape
        )
    }
}

#[test]
fn test_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<InvalidShapeError>();
}

#[test]
fn test_send() {
    fn assert_send<T: Send>() {}
    assert_send::<InvalidShapeError>();
}

impl<T, S> TensorRefMatrix<T, S, RowAndColumn>
where
    S: MatrixRef<T> + NoInteriorMutability,
{
    /**
     * Creates a TensorRefMatrix wrapping a MatrixRef type and defaulting the dimension names
     * to "row" and "column" respectively.
     *
     * Result::Err is returned if the matrix dimension lengths are not at least 1x1.
     */
    pub fn from(source: S) -> Result<TensorRefMatrix<T, S, RowAndColumn>, InvalidShapeError> {
        TensorRefMatrix::with_names(source, RowAndColumn)
    }
}

impl<T, S, N> TensorRefMatrix<T, S, N>
where
    S: MatrixRef<T> + NoInteriorMutability,
    N: DimensionNames,
{
    /**
     * Creates a TensorRefMatrix wrapping a MatrixRef type and provided dimension names.
     *
     * Result::Err is returned if the provided dimension names are not unique, or the matrix
     * dimension lengths are not at least 1x1.
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * use easy_ml::tensors::views::TensorRef;
     * use easy_ml::interop::TensorRefMatrix;
     * assert_eq!(
     *     // We can always unwrap here because we know the input is 1x1 and "x" and "y" are unique
     *     // dimension names
     *     TensorRefMatrix::with_names(Matrix::from_scalar(1.0), ["x", "y"]).unwrap().view_shape(),
     *     [("x", 1), ("y", 1)]
     * );
     * ```
     */
    pub fn with_names(source: S, names: N) -> Result<TensorRefMatrix<T, S, N>, InvalidShapeError> {
        let dimensions = names.names();
        let shape = InvalidShapeError {
            shape: [
                (dimensions[0], source.view_rows()),
                (dimensions[1], source.view_columns()),
            ],
        };
        if shape.is_valid() {
            Ok(TensorRefMatrix {
                source,
                names,
                _type: PhantomData,
            })
        } else {
            Err(shape)
        }
    }
}

// # Safety
// The contract of MatrixRef<T> + NoInteriorMutability is essentially the ungeneralised version of
// TensorRef, so we're good on no interior mutability and valid indexing behaviour. The
// TensorRef only requirements are that "all dimension names in the view_shape must be unique"
// and "all dimension lengths in the view_shape must be non zero". We enforce both of these during
// construction, and the NoInteriorMutability bounds ensures these invariants remain valid.
unsafe impl<T, S, N> TensorRef<T, 2> for TensorRefMatrix<T, S, N>
where
    S: MatrixRef<T> + NoInteriorMutability,
    N: DimensionNames,
{
    fn get_reference(&self, indexes: [usize; 2]) -> Option<&T> {
        self.source.try_get_reference(indexes[0], indexes[1])
    }

    fn view_shape(&self) -> [(Dimension, usize); 2] {
        let (rows, columns) = (self.source.view_rows(), self.source.view_columns());
        let [row, column] = self.names.names();
        [(row, rows), (column, columns)]
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; 2]) -> &T {
        self.source.get_reference_unchecked(indexes[0], indexes[1])
    }
}
