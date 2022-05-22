/*!
 * Interopability APIs between
 * [Matrix](crate::matrices::Matrix)/[MatrixView](crate::matrices::views::MatrixView) and
 * [Tensor](crate::tensors::Tensor)/[TensorView](crate::tensors::views::TensorView).
 */

use crate::matrices::views::{MatrixRef, NoInteriorMutability};
use crate::tensors::Dimension;
use crate::tensors::views::TensorRef;

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
 * let tensor_view = TensorView::from(TensorRefMatrix::from(&matrix));
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

impl<T, S> TensorRefMatrix<T, S, RowAndColumn>
where
    S: MatrixRef<T> + NoInteriorMutability,
{
    /**
     * Creates a TensorRefMatrix wrapping a MatrixRef type and defaulting the dimension names
     * to "row" and "column" respectively.
     */
    pub fn from(source: S) -> TensorRefMatrix<T, S, RowAndColumn> {
        TensorRefMatrix::with_names(source, RowAndColumn)
    }
}

impl<T, S, N> TensorRefMatrix<T, S, N>
where
    S: MatrixRef<T> + NoInteriorMutability,
    N: DimensionNames,
{
    /**
     * Creates a TensorRefMatrix wrapping a MatrixRef type and provided dimension names
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * use easy_ml::tensors::views::TensorRef;
     * use easy_ml::interop::TensorRefMatrix;
     * assert_eq!(
     *     TensorRefMatrix::with_names(Matrix::from_scalar(1.0), ["x", "y"]).view_shape(),
     *     [("x", 1), ("y", 1)]
     * );
     * ```
     */
    pub fn with_names(source: S, names: N) -> TensorRefMatrix<T, S, N> {
        TensorRefMatrix {
            source,
            names,
            _type: PhantomData,
        }
    }
}

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
