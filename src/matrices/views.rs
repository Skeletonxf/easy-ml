/*!
 * Generic views into a matrix.
 *
 * The concept of a view into a matrix is built from the low level [MatrixRef](MatrixRef) and
 * [MatrixMut](MatrixMut) traits which define having read and read/write access to Matrix data
 * respectively, and the high level API implemented on the [MatrixView](MatrixView) struct.
 *
 * Since a Matrix is itself a MatrixRef, the APIs for the traits are purposefully verbose to
 * avoid name clashes with methods defined on the Matrix and MatrixView types. You should
 * typically use MatrixRef and MatrixMut implementations via the MatrixView struct.
 */

use std::marker::PhantomData;

use crate::matrices::{Column, Row};
use crate::matrices::iterators::{
    ColumnIterator, ColumnMajorIterator, ColumnMajorReferenceIterator, ColumnReferenceIterator,
    DiagonalIterator, DiagonalReferenceIterator, RowIterator, RowMajorIterator,
    RowMajorReferenceIterator, RowReferenceIterator,
};

pub mod traits;

/**
* A shared/immutable reference to a matrix (or a portion of it) of some type.
*
* # Indexing
*
* Valid indexes into a MatrixRef range from 0 inclusive to `view_rows` exclusive for rows and
* from 0 inclusive to `view_columns` exclusive for columns. Even if a 4x4 matrix creates some
* 2x2 MatrixRef that can view only its center, the indexes used on the MatrixRef would be
* 0,0 to 1,1, not 1,1 to 2,2 as corresponding on the matrix.
*
* # Safety
*
* In order to support returning references without bounds checking in a useful way, the
* implementing type is required to uphold several invariants.
*
* 1 - Any valid index as described in Indexing will yield a safe reference when calling
* `get_reference_unchecked` and `get_reference_unchecked_mut` - It is the caller's responsbility
* to check `view_rows`/`view_columns` and request only indexes in range.
*
* 2 - Either the `view_rows`/`view_columns` that define which indexes are valid may not
* be changed by a shared reference to the MatrixRef, or `get_reference_unchecked` and
* `get_reference_unchecked_mut` must panic if the index is invalid.
*
* Essentially, interior mutability causes problems, since code looping through the range of valid
* indexes in a MatrixRef needs to be able to rely on that range of valid indexes not changing.
* This is trivially the case by default since a [Matrix](super::Matrix) does not have any form of
* interior mutability, and therefore an iterator holding a shared reference to a Matrix prevents
* that matrix being resized. However, a type implementing MatrixRef could introduce interior
* mutability by putting the Matrix in a `Arc<Mutex<>>` which would allow another thread to
* resize a matrix while an iterator was looping through previously valid indexes on a different
* thread. For such cases, the MatrixRef implementation must ensure that invalid indexes panic
* for `get_reference_unchecked` and `get_reference_unchecked_mut` to prevent undefined behavior.
* Note that it is okay to be able to resize a Matrix backing a MatrixRef if that always requires
* an exclusive reference to the MatrixRef/Matrix, since the exclusivity prevents the above
* scenario.
*/
pub unsafe trait MatrixRef<T> {
    /**
     * Gets a reference to the value at the index if the index is in range. Otherwise returns None.
     */
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T>;

    /**
     * The number of rows that this reference can view. This may be less than the actual number of
     * rows stored in the matrix.
     */
    fn view_rows(&self) -> Row;

    /**
     * The number of columns that this reference can view. This may be less than the actual number
     * of columns stored in the matrix.
     */
    fn view_columns(&self) -> Column;

    /**
     * Gets a reference to the value at the index without doing any bounds checking. For a safe
     * alternative see [try_get_reference](MatrixRef::try_get_reference).
     *
     * # Safety
     *
     * Calling this method with an out-of-bounds index is *[undefined behavior]* even if the
     * resulting reference is not used. Valid indexes are defined as in [MatrixRef].
     *
     * [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
     * [MatrixRef]: MatrixRef
     */
    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T;
}

/**
 * A unique/mutable reference to a matrix (or a portion of it) of some type.
 *
 * # Safety
 *
 * See [MatrixRef](MatrixRef).
 */
pub unsafe trait MatrixMut<T>: MatrixRef<T> {
    /**
     * Gets a mutable reference to the value at the index, if the index is in range. Otherwise
     * returns None.
     */
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T>;

    /**
     * Gets a mutable reference to the value at the index without doing any bounds checking.
     * For a safe alternative see [try_get_reference_mut](MatrixMut::try_get_reference_mut).
     *
     * # Safety
     *
     * Calling this method with an out-of-bounds index is *[undefined behavior]* even if the
     * resulting reference is not used. Valid indexes are defined as in [MatrixRef].
     *
     * [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
     * [MatrixRef]: MatrixRef
     */
    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T;
}

/**
 * A view into some or all of a matrix.
 *
 * A MatrixView has a similar relationship to a [`Matrix`](crate::matrices::Matrix) as a
 * `&str` has to a `String`, or an array slice to an array. A MatrixView cannot resize
 * its source, and may span only a portion of the source Matrix in each dimension.
 *
 * However a MatrixView is generic not only over the type of the data in the Matrix,
 * but also over the way the Matrix is 'sliced' and the two are orthogonal to each other.
 *
 * MatrixView closely mirrors the API of Matrix, minus resizing methods which are not available.
 */
 #[derive(Clone, Debug)]
pub struct MatrixView<T, S> {
    source: S,
    _type: PhantomData<T>,
    // TODO: Transposition?
}

// TODO mapping functions, trait implementations for MatrixView,
// linear_algebra numeric functions, numeric operators

/**
 * MatrixView methods which require only read access via a [MatrixRef](MatrixRef) source.
 */
impl <T, S> MatrixView<T, S>
where
    S: MatrixRef<T>
{
    /**
     * Creates a MatrixView from a source of some type.
     *
     * The lifetime of the source determines the lifetime of the MatrixView created. If the
     * MatrixView is created from a reference to a Matrix, then the MatrixView cannot live
     * longer than the Matrix referenced.
     */
    pub fn from(source: S) -> MatrixView<T, S> {
        MatrixView {
            source,
            _type: PhantomData,
        }
    }

    /**
     * Consumes the matrix view, yielding the source it was created from.
     */
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Returns the dimensionality of this matrix view in Row, Column format
     */
    pub fn size(&self) -> (Row, Column) {
        (self.rows(), self.columns())
    }

    /**
     * Gets the number of rows visible to this matrix view.
     */
    pub fn rows(&self) -> Row {
        self.source.view_rows()
    }

    /**
     * Gets the number of columns visible to this matrix view.
     */
    pub fn columns(&self) -> Column {
        self.source.view_columns()
    }

    /**
     * Gets a reference to the value at this row and column. Rows and Columns are 0 indexed.
     *
     * # Panics
     *
     * Panics if the index is out of range.
     */
    #[track_caller]
    pub fn get_reference(&self, row: Row, column: Column) -> &T {
        match self.source.try_get_reference(row, column) {
            Some(reference) => reference,
            None => panic!(
                "Index ({},{}) not in range, MatrixView range is (0,0) to ({},{}).",
                row, column, self.rows(), self.columns()
            )
        }
    }

    /**
     * Gets a reference to the value at the row and column if the index is in range.
     * Otherwise returns None.
     */
    pub fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        self.source.try_get_reference(row, column)
    }

    /**
     * Returns an iterator over references to a column vector in this matrix view.
     * Columns are 0 indexed.
     *
     * # Panics
     *
     * Panics if the column is not visible to this view.
     */
    #[track_caller]
    pub fn column_reference_iter(&self, column: Column) -> ColumnReferenceIterator<T, S> {
        ColumnReferenceIterator::from(&self.source, column)
    }

    /**
     * Returns an iterator over references to a row vector in this matrix view.
     * Rows are 0 indexed.
     *
     * # Panics
     *
     * Panics if the row is not visible to this view.
     */
    #[track_caller]
    pub fn row_reference_iter(&self, row: Row) -> RowReferenceIterator<T, S> {
        RowReferenceIterator::from(&self.source, row)
    }

    /**
     * Returns a column major iterator over references to all values in this matrix view,
     * proceeding through each column in order.
     */
    pub fn column_major_reference_iter(&self) -> ColumnMajorReferenceIterator<T, S> {
        ColumnMajorReferenceIterator::from(&self.source)
    }

    /**
     * Returns a row major iterator over references to all values in this matrix view,
     * proceeding through each row in order.
     */
    pub fn row_major_reference_iter(&self) -> RowMajorReferenceIterator<T, S> {
        RowMajorReferenceIterator::from(&self.source)
    }

    /**
     * Returns an iterator over references to the main diagonal in this matrix view.
     */
    pub fn diagonal_reference_iter(&self) -> DiagonalReferenceIterator<T, S> {
        DiagonalReferenceIterator::from(&self.source)
    }
}

/**
 * MatrixView methods which require only read access via a [MatrixRef](MatrixRef) source
 * and a clonable type.
 */
impl <T, S> MatrixView<T, S>
where
    S: MatrixRef<T>,
    T: Clone,
{
    /**
     * Gets a copy of the value at this row and column. Rows and Columns are 0 indexed.
     *
     * # Panics
     *
     * Panics if the index is out of range.
     */
    #[track_caller]
    pub fn get(&self, row: Row, column: Column) -> T {
        match self.source.try_get_reference(row, column) {
            Some(reference) => reference.clone(),
            None => panic!(
                "Index ({},{}) not in range, MatrixView range is (0,0) to ({},{}).",
                row, column, self.rows(), self.columns()
            )
        }
    }

    /**
     * Returns an iterator over a column vector in this matrix view. Columns are 0 indexed.
     *
     * If you have a matrix such as:
     * ```ignore
     * [
     *    1, 2, 3
     *    4, 5, 6
     *    7, 8, 9
     * ]
     * ```
     * then a column of 0, 1, and 2 will yield [1, 4, 7], [2, 5, 8] and [3, 6, 9]
     * respectively. If you do not need to copy the elements use
     * [`column_reference_iter`](MatrixView::column_reference_iter) instead.
     *
     * # Panics
     *
     * Panics if the column does not exist in this matrix.
     */
    #[track_caller]
    pub fn column_iter(&self, column: Column) -> ColumnIterator<T, S> {
        ColumnIterator::from(&self.source, column)
    }

    /**
     * Returns an iterator over a row vector in this matrix view. Rows are 0 indexed.
     *
     * If you have a matrix such as:
     * ```ignore
     * [
     *    1, 2, 3
     *    4, 5, 6
     *    7, 8, 9
     * ]
     * ```
     * then a row of 0, 1, and 2 will yield [1, 2, 3], [4, 5, 6] and [7, 8, 9]
     * respectively. If you do not need to copy the elements use
     * [`row_reference_iter`](MatrixView::row_reference_iter) instead.
     *
     * # Panics
     *
     * Panics if the row does not exist in this matrix.
     */
    #[track_caller]
    pub fn row_iter(&self, row: Row) -> RowIterator<T, S> {
        RowIterator::from(&self.source, row)
    }

    /**
     * Returns a column major iterator over all values in this matrix view, proceeding through each
     * column in order.
     *
     * If you have a matrix such as:
     * ```ignore
     * [
     *    1, 2
     *    3, 4
     * ]
     * ```
     * then the iterator will yield [1, 3, 2, 4]. If you do not need to copy the
     * elements use [`column_major_reference_iter`](MatrixView::column_major_reference_iter)
     * instead.
     */
    pub fn column_major_iter(&self) -> ColumnMajorIterator<T, S> {
        ColumnMajorIterator::from(&self.source)
    }

    /**
     * Returns a row major iterator over all values in this matrix view, proceeding through each
     * row in order.
     *
     * If you have a matrix such as:
     * ```ignore
     * [
     *    1, 2
     *    3, 4
     * ]
     * ```
     * then the iterator will yield [1, 2, 3, 4]. If you do not need to copy the
     * elements use [`row_major_reference_iter`](MatrixView::row_major_reference_iter) instead.
     */
    pub fn row_major_iter(&self) -> RowMajorIterator<T, S> {
        RowMajorIterator::from(&self.source)
    }

    /**
     * Returns a iterator over the main diagonal of this matrix view.
     *
     * If you have a matrix such as:
     * ```ignore
     * [
     *    1, 2
     *    3, 4
     * ]
     * ```
     * then the iterator will yield [1, 4]. If you do not need to copy the
     * elements use [`diagonal_reference_iter`](MatrixView::diagonal_reference_iter) instead.
     *
     * # Examples
     *
     * Computing a [trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra))
     * ```
     * use easy_ml::matrices::Matrix;
     * use easy_ml::matrices::views::MatrixView;
     * let view = MatrixView::from(Matrix::from(vec![
     *     vec![ 1, 2, 3 ],
     *     vec![ 4, 5, 6 ],
     *     vec![ 7, 8, 9 ],
     * ]));
     * let trace: i32 = view.diagonal_iter().sum();
     * assert_eq!(trace, 1 + 5 + 9);
     * ```
     */
    pub fn diagonal_iter(&self) -> DiagonalIterator<T, S> {
        DiagonalIterator::from(&self.source)
    }
}

/**
 * MatrixView methods which require mutable access via a [MatrixMut](MatrixMut) source.
 */
impl <T, S> MatrixView<T, S>
where
    S: MatrixMut<T> {
    /**
     * Sets a new value to this row and column. Rows and Columns are 0 indexed.
     *
     * # Panics
     *
     * Panics if the index is out of range.
     */
    #[track_caller]
    pub fn set(&mut self, row: Row, column: Column, value: T) {
        match self.source.try_get_reference_mut(row, column) {
            Some(reference) => *reference = value,
            None => panic!(
                "Index ({},{}) not in range, MatrixView range is (0,0) to ({},{}).",
                row, column, self.rows(), self.columns()
            )
        }
    }

    /**
     * Gets a mutable reference to the value at the row and column if the index is in range.
     * Otherwise returns None.
     */
    pub fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        self.source.try_get_reference_mut(row, column)
    }
}

// Common formatting logic used for Matrix and MatrixView Display implementations
pub(crate) fn format_view<T, S>(view: &S, f: &mut std::fmt::Formatter) -> std::fmt::Result
where
    T: std::fmt::Display,
    S: MatrixRef<T>
{
    let rows = view.view_rows();
    let columns = view.view_columns();
    write!(f, "[ ")?;
    for row in 0..rows {
        if row > 0 {
            write!(f, "  ")?;
        }
        for column in 0..columns {
            let value = match view.try_get_reference(row, column) {
                Some(x) => x,
                None => panic!(
                    "Expected ({},{}) to be in range of (0,0) to ({},{})",
                    row, column,
                    rows, columns
                ),
            };
            // default to 3 decimals but allow the caller to override
            // TODO: ideally want to set significant figures instead of decimals
            write!(f, "{:.*}", f.precision().unwrap_or(3), value)?;
            if column < columns - 1 {
                write!(f, ", ")?;
            }
        }
        if row < rows - 1 {
            writeln!(f)?;
        }
    }
    write!(f, " ]")
}

/**
 * Any matrix view of a Displayable type implements Display
 */
impl <T, S> std::fmt::Display for MatrixView<T, S>
where
    T: std::fmt::Display,
    S: MatrixRef<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        format_view(&self.source, f)
    }
}

#[test]
fn printing_matrices() {
    use crate::matrices::Matrix;
    let view = MatrixView::from(Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]));
    let formatted = view.to_string();
    assert_eq!("[ 1.000, 2.000\n  3.000, 4.000 ]", formatted);
}

/**
 * PartialEq is implemented as two matrix views are equal if and only if all their elements
 * are equal and they have the same size.
 */
impl <T, S> PartialEq for MatrixView<T, S>
where
    T: PartialEq,
    S: MatrixRef<T>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.rows() != other.rows() {
            return false;
        }
        if self.columns() != other.columns() {
            return false;
        }
        // perform elementwise check, return true only if every element in
        // each matrix is the same
        self.row_major_reference_iter()
            .zip(other.row_major_reference_iter())
            .all(|(x, y)| x == y)
    }
}

#[test]
fn creating_matrix_views_erased() {
    use crate::matrices::Matrix;
    let matrix = Matrix::from(vec![vec![1.0]]);
    let boxed: Box<dyn MatrixMut<f32>> = Box::new(matrix);
    let mut view = MatrixView::from(boxed);
    view.set(0, 0, view.get(0, 0) + 1.0);
    assert_eq!(2.0, view.get(0, 0));
}
