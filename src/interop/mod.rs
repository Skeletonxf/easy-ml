/*!
 * Interopability APIs between
 * [Matrix](crate::matrices::Matrix)/[MatrixView](crate::matrices::views::MatrixView) and
 * [Tensor](crate::tensors::Tensor)/[TensorView](crate::tensors::views::TensorView).
 */

use crate::matrices::views::DataLayout as MDataLayout;
use crate::matrices::views::{MatrixMut, MatrixRef, NoInteriorMutability};
use crate::matrices::{Column, Row};
use crate::tensors::views::DataLayout as TDataLayout;
use crate::tensors::views::{TensorMut, TensorRef};
use crate::tensors::{Dimension, InvalidShapeError};

use std::marker::PhantomData;

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
 *     matrix.row_iter(1).eq(tensor_view.select([("row", 1)]).iter()),
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
        *self
    }
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
    pub fn from(source: S) -> Result<TensorRefMatrix<T, S, RowAndColumn>, InvalidShapeError<2>> {
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
    pub fn with_names(
        source: S,
        names: N,
    ) -> Result<TensorRefMatrix<T, S, N>, InvalidShapeError<2>> {
        let dimensions = names.names();
        let shape = InvalidShapeError::new([
            (dimensions[0], source.view_rows()),
            (dimensions[1], source.view_columns()),
        ]);
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

    fn data_layout(&self) -> TDataLayout<2> {
        let [rows_dimension, columns_dimension] = self.names.names();
        // Row major and column major are the less generalised versions of
        // a linear data layout. Since our view shape is hardcoded here to rows
        // then columns, a row major matrix means the most significant dimension
        // is the first, and the least significant dimension is the second. Similarly
        // a column major matrix means the opposite.
        match self.source.data_layout() {
            MDataLayout::RowMajor => TDataLayout::Linear([rows_dimension, columns_dimension]),
            MDataLayout::ColumnMajor => TDataLayout::Linear([columns_dimension, rows_dimension]),
            MDataLayout::Other => TDataLayout::Other,
        }
    }
}

// # Safety
// The contract of MatrixMut<T> + NoInteriorMutability is essentially the ungeneralised version of
// TensorMut, so we're good on no interior mutability and valid indexing behaviour. The
// TensorMut only requirements are that "all dimension names in the view_shape must be unique"
// and "all dimension lengths in the view_shape must be non zero". We enforce both of these during
// construction, and the NoInteriorMutability bounds ensures these invariants remain valid.
unsafe impl<T, S, N> TensorMut<T, 2> for TensorRefMatrix<T, S, N>
where
    S: MatrixMut<T> + NoInteriorMutability,
    N: DimensionNames,
{
    fn get_reference_mut(&mut self, indexes: [usize; 2]) -> Option<&mut T> {
        self.source.try_get_reference_mut(indexes[0], indexes[1])
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; 2]) -> &mut T {
        self.source
            .get_reference_unchecked_mut(indexes[0], indexes[1])
    }
}

/**
 * A wrapper around a Tensor<_, 2> type that implements MatrixRef and can thus be used in a
 * [MatrixView](crate::matrices::views::MatrixView)
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * use easy_ml::matrices::views::MatrixView;
 * use easy_ml::interop::MatrixRefTensor;
 * let tensor = Tensor::from([("row", 2), ("column", 4)], vec![
 *     1, 3, 5, 7,
 *     2, 4, 6, 8
 * ]);
 * let matrix_view = MatrixView::from(MatrixRefTensor::from(&tensor));
 * assert_eq!(
 *     matrix_view.row_iter(1).eq(tensor.select([("row", 1)]).iter()),
 *     true
 * );
 * ```
 */
pub struct MatrixRefTensor<T, S> {
    source: S,
    _type: PhantomData<T>,
}

impl<T, S> MatrixRefTensor<T, S>
where
    S: TensorRef<T, 2>,
{
    /**
     * Creates a MatrixRefTensor wrapping a TensorRef type and stripping the dimension names.
     *
     * The first dimension in the TensorRef type becomes the rows, and the second dimension the
     * columns. If your tensor is the other way around,
     * [reorder it first](crate::tensors::indexing::TensorAccess).
     */
    pub fn from(source: S) -> MatrixRefTensor<T, S> {
        MatrixRefTensor {
            source,
            _type: PhantomData,
        }
    }
}

// # Safety
// The contract of TensorRef<T, 2> covers everything the compiler can't check for MatrixRef<T>
// so if we just delegate to the tensor source and hide the dimension names, the index based API
// meets every requirement by default.
unsafe impl<T, S> MatrixRef<T> for MatrixRefTensor<T, S>
where
    S: TensorRef<T, 2>,
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        self.source.get_reference([row, column])
    }

    fn view_rows(&self) -> Row {
        self.source.view_shape()[0].1
    }

    fn view_columns(&self) -> Column {
        self.source.view_shape()[1].1
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        self.source.get_reference_unchecked([row, column])
    }

    fn data_layout(&self) -> MDataLayout {
        let rows_dimension = self.source.view_shape()[0].0;
        let columns_dimension = self.source.view_shape()[1].0;
        // Row major and column major are the less generalised versions of
        // a linear data layout. Since our view shape is always interpreted here as rows
        // then columns, a row major matrix means the most significant dimension
        // is the first, and the least significant dimension is the second. Similarly
        // a column major matrix means the opposite.
        let data_layout = self.source.data_layout();
        if data_layout == TDataLayout::Linear([rows_dimension, columns_dimension]) {
            MDataLayout::RowMajor
        } else if data_layout == TDataLayout::Linear([columns_dimension, rows_dimension]) {
            MDataLayout::ColumnMajor
        } else {
            match self.source.data_layout() {
                TDataLayout::NonLinear => MDataLayout::Other,
                TDataLayout::Other => MDataLayout::Other,
                // This branch should never happen as no other Linear layouts are valid according
                // to the docs the source implementation must follow but we need to keep the Rust
                // compiler happy
                TDataLayout::Linear([_, _]) => MDataLayout::Other,
            }
        }
    }
}

// # Safety
// The contract of TensorMut<T, 2> covers everything the compiler can't check for MatrixMut<T>
// so if we just delegate to the tensor source and hide the dimension names, the index based API
// meets every requirement by default.
unsafe impl<T, S> MatrixMut<T> for MatrixRefTensor<T, S>
where
    S: TensorMut<T, 2>,
{
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        self.source.get_reference_mut([row, column])
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        self.source.get_reference_unchecked_mut([row, column])
    }
}

// # Safety
// No interior mutability is implied by TensorRef
unsafe impl<T, S> NoInteriorMutability for MatrixRefTensor<T, S> where S: TensorRef<T, 2> {}
