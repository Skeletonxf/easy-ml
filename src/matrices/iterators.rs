/*!
 * Iterators over parts of a Matrix
 *
 * - Over a row: Row[(Reference)](RowReferenceIterator)[(Mut)](RowReferenceMutIterator)[Iterator](RowIterator)
 * - Over a column: Column[(Reference)](ColumnReferenceIterator)[(Mut)](ColumnReferenceMutIterator)[Iterator](ColumnIterator)
 * - Over all data in row major order: RowMajor[(Reference)](RowMajorReferenceIterator)[(Mut)](RowMajorReferenceMutIterator)[Iterator](RowMajorIterator)
 * - Over all data in column major order: ColumnMajor[(Reference)](ColumnMajorReferenceIterator)[(Mut)](ColumnMajorReferenceMutIterator)[Iterator](ColumnMajorIterator)
 * - Over the main diagonal: Diagonal[(Reference)](DiagonalReferenceIterator)[(Mut)](DiagonalReferenceMutIterator)[Iterator](DiagonalIterator)
 *
 * # Examples
 *
 * Extending a matrix with new columns
 * ```
 * use easy_ml::matrices::Matrix;
 *
 * // we start with some matrix where the first and second columns correspond
 * // to x and y points
 * let mut matrix = Matrix::from(vec![
 *     vec![ 3.0, 4.0 ],
 *     vec![ 8.0, 1.0 ],
 *     vec![ 2.0, 9.0 ]]);
 * // insert a third column based on the formula x * y
 * matrix.insert_column_with(2, matrix.column_iter(0)
 *     // join together the x and y columns
 *     .zip(matrix.column_iter(1))
 *     // compute the values for the new column
 *     .map(|(x, y)| x * y)
 *     // Collect into a vector so we stop immutably borrowing from `matrix`.
 *     // This is only neccessary when we use the data from a Matrix to modify itself,
 *     // because the rust compiler enforces that we do not mutably and immutably borrow
 *     // something at the same time. If we used data from a different Matrix to update
 *     // `matrix` then we could stop at map and pass the iterator directly.
 *     .collect::<Vec<f64>>()
 *     // now that the Vec created owns the data for the new column and we have stopped
 *     // borrowing immutably from `matrix` we turn the vec back into an iterator and
 *     // mutably borrow `matrix` to add the new column
 *     .drain(..));
 * assert_eq!(matrix.get(0, 2), 3.0 * 4.0);
 * assert_eq!(matrix.get(1, 2), 8.0 * 1.0);
 * assert_eq!(matrix.get(2, 2), 2.0 * 9.0);
 * ```
 *
 * # Matrix layout and iterator performance
 *
 * Internally the Matrix type uses a flattened array with [row major storage](https://en.wikipedia.org/wiki/Row-_and_column-major_order)
 * of the data. Due to [CPU cache lines](https://stackoverflow.com/questions/3928995/how-do-cache-lines-work)
 * this means row major access of elements is likely to be faster than column major indexing,
 * once a matrix is large enough, so you should favor iterating through each row.
 *
 * ```
 * use easy_ml::matrices::Matrix;
 * let matrix = Matrix::from(vec![
 *    vec![ 1, 2 ],
 *    vec![ 3, 4 ]]);
 * // storage of elements is [1, 2, 3, 4]
 *
 * // row major access
 * for row in 0..2 {
 *     for column in 0..2 {
 *         println!("{}", matrix.get(row, column));
 *     }
 * } // -> 1 2 3 4
 * matrix.row_major_iter().for_each(|e| println!("{}", e)); // -> 1 2 3 4
 * matrix.row_major_iter().with_index().for_each(|e| println!("{:?}", e)); // -> ((0, 0), 1) ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)
 *
 * // column major access
 * for column in 0..2 {
 *     for row in 0..2 {
 *         println!("{}", matrix.get(row, column));
 *     }
 * } // -> 1 3 2 4
 * matrix.column_major_iter().for_each(|e| println!("{}", e)); // -> 1 3 2 4
 * matrix.column_major_iter().with_index().for_each(|e| println!("{:?}", e)); // // -> ((0, 0), 1), ((1, 0), 3), ((0, 1), 2), ((1, 1), 4)
 * ```
 *
 * Iterators are also able to elide array indexing bounds checks which may improve performance over
 * explicit calls to get or get_reference in a loop, however you can use unsafe getters to elide
 * bounds checks in loops as well so you are not forced to use iterators even if the checks are
 * a performance concern.
 */

use std::iter::{ExactSizeIterator, FusedIterator};
use std::marker::PhantomData;
use std::ops::Range;

use crate::matrices::views::{MatrixMut, MatrixRef, NoInteriorMutability};
use crate::matrices::{Column, Matrix, Row};

trait MatrixRefExtension<T>: MatrixRef<T> {
    /**
     * Helper for asserting starting indexes are in range.
     */
    fn index_is_valid(&self, row: Row, column: Column) -> bool {
        row < self.view_rows() && column < self.view_columns()
    }
}

impl<R, T> MatrixRefExtension<T> for R where R: MatrixRef<T> {}

/**
 * A wrapper around another iterator that iterates through each element in the iterator and
 * includes the index used to access the element.
 *
 * This is like a 2D version of [`enumerate`](std::iter::Iterator::enumerate).
 */
#[derive(Debug)]
pub struct WithIndex<I> {
    iterator: I,
}

impl<I> WithIndex<I> {
    /**
     * Consumes the WithIndex, yielding the iterator it was created from.
     */
    pub fn source(self) -> I {
        self.iterator
    }
}

/**
 * An iterator over a column in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * Depending on the column iterator you want to obtain,
 * can either iterate through 1, 3 or 2, 4.
 */
#[derive(Debug)]
pub struct ColumnIterator<'a, T: Clone, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column: Column,
    range: Range<Row>,
    _type: PhantomData<&'a T>,
}

impl<'a, T: Clone> ColumnIterator<'a, T> {
    /**
     * Constructs a column iterator over this matrix.
     *
     * # Panics
     *
     * Panics if the column does not exist in this matrix.
     */
    #[track_caller]
    pub fn new(matrix: &Matrix<T>, column: Column) -> ColumnIterator<T> {
        ColumnIterator::from(matrix, column)
    }
}

impl<'a, T: Clone, S: MatrixRef<T>> ColumnIterator<'a, T, S> {
    /**
     * Constructs a column iterator over this source.
     *
     * # Panics
     *
     * Panics if the column does not exist in this source.
     */
    #[track_caller]
    pub fn from(source: &S, column: Column) -> ColumnIterator<T, S> {
        assert!(
            source.index_is_valid(0, column),
            "Expected ({},{}) to be in range",
            0,
            column
        );
        ColumnIterator {
            matrix: source,
            column,
            range: 0..source.view_rows(),
            _type: PhantomData,
        }
    }
}

impl<'a, T: Clone, S: MatrixRef<T>> Iterator for ColumnIterator<'a, T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(row) => unsafe {
                // Safety: We initialised the range to 0..matrix.view_rows(), and
                // checked we can read from this column at creation, hence if
                // the view_rows have not changed this read is in bounds and if
                // they are able to be changed, then the MatrixRef implementation is
                // required to bounds check for us.
                Some(
                    self.matrix
                        .get_reference_unchecked(row, self.column)
                        .clone(),
                )
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}
impl<'a, T: Clone, S: MatrixRef<T>> FusedIterator for ColumnIterator<'a, T, S> {}
impl<'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for ColumnIterator<'a, T, S> {}

/**
 * An iterator over a row in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * Depending on the row iterator you want to obtain,
 * can either iterate through 1, 2 or 3, 4.
 */
#[derive(Debug)]
pub struct RowIterator<'a, T: Clone, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    row: Row,
    range: Range<Column>,
    _type: PhantomData<&'a T>,
}

impl<'a, T: Clone> RowIterator<'a, T> {
    /**
     * Constructs a row iterator over this matrix.
     *
     * # Panics
     *
     * Panics if the row does not exist in this matrix.
     */
    #[track_caller]
    pub fn new(matrix: &Matrix<T>, row: Row) -> RowIterator<T> {
        RowIterator::from(matrix, row)
    }
}

impl<'a, T: Clone, S: MatrixRef<T>> RowIterator<'a, T, S> {
    /**
     * Constructs a row iterator over this source.
     *
     * # Panics
     *
     * Panics if the row does not exist in this source.
     */
    #[track_caller]
    pub fn from(source: &S, row: Row) -> RowIterator<T, S> {
        assert!(
            source.index_is_valid(row, 0),
            "Expected ({},{}) to be in range",
            row,
            0
        );
        RowIterator {
            matrix: source,
            row,
            range: 0..source.view_columns(),
            _type: PhantomData,
        }
    }
}

impl<'a, T: Clone, S: MatrixRef<T>> Iterator for RowIterator<'a, T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(column) => unsafe {
                // Safety: We initialised the range to 0..matrix.view_columns(), and
                // checked we can read from this row at creation, hence if
                // the view_columns have not changed this read is in bounds and if
                // they are able to be changed, then the MatrixRef implementation is
                // required to bounds check for us.
                Some(
                    self.matrix
                        .get_reference_unchecked(self.row, column)
                        .clone(),
                )
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}
impl<'a, T: Clone, S: MatrixRef<T>> FusedIterator for RowIterator<'a, T, S> {}
impl<'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for RowIterator<'a, T, S> {}

// Common column major iterator logic
fn column_major_iter(
    finished: &mut bool,
    rows: Row,
    columns: Column,
    row_counter: &mut Row,
    column_counter: &mut Column,
) -> Option<(Row, Column)> {
    if *finished {
        return None;
    }

    let value = Some((*row_counter, *column_counter));

    if *row_counter == rows - 1 && *column_counter == columns - 1 {
        // reached end of matrix for next iteration
        *finished = true;
    }

    if *row_counter == rows - 1 {
        // reached end of a column, need to reset to first element in next column
        *row_counter = 0;
        *column_counter += 1;
    } else {
        // keep incrementing through this column
        *row_counter += 1;
    }

    value
}

// Common column major iterator size hint logic
fn column_major_size_hint(
    rows: Row,
    columns: Column,
    row_counter: Row,
    column_counter: Column,
) -> (usize, Option<usize>) {
    let remaining_columns = columns - column_counter;
    match remaining_columns {
        0 => (0, Some(0)),
        1 => {
            // we're on the last column, so return how many items are left for us to
            // go through with the row counter
            let remaining_rows = rows - row_counter;
            (remaining_rows, Some(remaining_rows))
        }
        x => {
            // we still have at least one full column left in addition to what's left
            // for this column's row counter
            let remaining_rows = rows - row_counter;
            // each full column takes as many iterations as the matrix has rows
            let remaining_full_columns = (x - 1) * rows;
            let remaining = remaining_rows + remaining_full_columns;
            (remaining, Some(remaining))
        }
    }
}

/**
 * A column major iterator over all values in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as 1, 3, 2, 4
 */
#[derive(Debug)]
pub struct ColumnMajorIterator<'a, T: Clone, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column_counter: Column,
    columns: Column,
    row_counter: Row,
    rows: Row,
    finished: bool,
    _type: PhantomData<&'a T>,
}

impl<'a, T: Clone> ColumnMajorIterator<'a, T> {
    /**
     * Constructs a column major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> ColumnMajorIterator<T> {
        ColumnMajorIterator::from(matrix)
    }
}

impl<'a, T: Clone, S: MatrixRef<T>> ColumnMajorIterator<'a, T, S> {
    /**
     * Constructs a column major iterator over this source.
     */
    pub fn from(source: &S) -> ColumnMajorIterator<T, S> {
        ColumnMajorIterator {
            matrix: source,
            column_counter: 0,
            columns: source.view_columns(),
            row_counter: 0,
            rows: source.view_rows(),
            finished: !source.index_is_valid(0, 0),
            _type: PhantomData,
        }
    }

    /**
     * Constructors an iterator which also yields the row and column index of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T: Clone, S: MatrixRef<T>> Iterator for ColumnMajorIterator<'a, T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        column_major_iter(
            &mut self.finished,
            self.rows,
            self.columns,
            &mut self.row_counter,
            &mut self.column_counter,
        )
        .map(|(row, column)| unsafe {
            // Safety: We checked on creation that 0,0 is in range, and after getting
            // our next value we check if we hit the end of the matrix and will avoid
            // calling this on our next loop if we finished. Hence if the view size has
            // not changed this read is in bounds and if they are able to be changed,
            // then the MatrixRef implementation is required to bounds check for us.
            self.matrix.get_reference_unchecked(row, column).clone()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        column_major_size_hint(
            self.rows,
            self.columns,
            self.row_counter,
            self.column_counter,
        )
    }
}
impl<'a, T: Clone, S: MatrixRef<T>> FusedIterator for ColumnMajorIterator<'a, T, S> {}
impl<'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for ColumnMajorIterator<'a, T, S> {}

impl<'a, T: Clone, S: MatrixRef<T>> Iterator for WithIndex<ColumnMajorIterator<'a, T, S>> {
    type Item = ((Row, Column), T);

    fn next(&mut self) -> Option<Self::Item> {
        let (row, column) = (self.iterator.row_counter, self.iterator.column_counter);
        self.iterator.next().map(|x| ((row, column), x))
    }
}
impl<'a, T: Clone, S: MatrixRef<T>> FusedIterator for WithIndex<ColumnMajorIterator<'a, T, S>> {}
impl<'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for WithIndex<ColumnMajorIterator<'a, T, S>> {}

// Common row major iterator logic
fn row_major_iter(
    finished: &mut bool,
    rows: Row,
    columns: Column,
    row_counter: &mut Row,
    column_counter: &mut Column,
) -> Option<(Row, Column)> {
    if *finished {
        return None;
    }

    let value = Some((*row_counter, *column_counter));

    if *column_counter == columns - 1 && *row_counter == rows - 1 {
        // reached end of matrix for next iteration
        *finished = true;
    }

    if *column_counter == columns - 1 {
        // reached end of a row, need to reset to first element in next row
        *column_counter = 0;
        *row_counter += 1;
    } else {
        // keep incrementing through this row
        *column_counter += 1;
    }

    value
}

// Common row major iterator size hint logic
fn row_major_size_hint(
    rows: Row,
    columns: Column,
    row_counter: Row,
    column_counter: Column,
) -> (usize, Option<usize>) {
    let remaining_rows = rows - row_counter;
    match remaining_rows {
        0 => (0, Some(0)),
        1 => {
            // we're on the last row, so return how many items are left for us to
            // go through with the column counter
            let remaining_columns = columns - column_counter;
            (remaining_columns, Some(remaining_columns))
        }
        x => {
            // we still have at least one full row left in addition to what's left
            // for this row's column counter
            let remaining_columns = columns - column_counter;
            // each full row takes as many iterations as the matrix has columns
            let remaining_full_rows = (x - 1) * columns;
            let remaining = remaining_columns + remaining_full_rows;
            (remaining, Some(remaining))
        }
    }
}

/**
 * A row major iterator over all values in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as 1, 2, 3, 4
 */
#[derive(Debug)]
pub struct RowMajorIterator<'a, T: Clone, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column_counter: Column,
    columns: Column,
    row_counter: Row,
    rows: Row,
    finished: bool,
    _type: PhantomData<&'a T>,
}

impl<'a, T: Clone> RowMajorIterator<'a, T> {
    /**
     * Constructs a row major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> RowMajorIterator<T> {
        RowMajorIterator::from(matrix)
    }
}

impl<'a, T: Clone, S: MatrixRef<T>> RowMajorIterator<'a, T, S> {
    /**
     * Constructs a row major iterator over this source.
     */
    pub fn from(source: &S) -> RowMajorIterator<T, S> {
        RowMajorIterator {
            matrix: source,
            column_counter: 0,
            columns: source.view_columns(),
            row_counter: 0,
            rows: source.view_rows(),
            finished: !source.index_is_valid(0, 0),
            _type: PhantomData,
        }
    }

    /**
     * Constructors an iterator which also yields the row and column index of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T: Clone, S: MatrixRef<T>> Iterator for RowMajorIterator<'a, T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        row_major_iter(
            &mut self.finished,
            self.rows,
            self.columns,
            &mut self.row_counter,
            &mut self.column_counter,
        )
        .map(|(row, column)| unsafe {
            // Safety: We checked on creation that 0,0 is in range, and after getting
            // our next value we check if we hit the end of the matrix and will avoid
            // calling this on our next loop if we finished. Hence if the view size has
            // not changed this read is in bounds and if they are able to be changed,
            // then the MatrixRef implementation is required to bounds check for us.
            self.matrix.get_reference_unchecked(row, column).clone()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        row_major_size_hint(
            self.rows,
            self.columns,
            self.row_counter,
            self.column_counter,
        )
    }
}
impl<'a, T: Clone, S: MatrixRef<T>> FusedIterator for RowMajorIterator<'a, T, S> {}
impl<'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for RowMajorIterator<'a, T, S> {}

impl<'a, T: Clone, S: MatrixRef<T>> Iterator for WithIndex<RowMajorIterator<'a, T, S>> {
    type Item = ((Row, Column), T);

    fn next(&mut self) -> Option<Self::Item> {
        let (row, column) = (self.iterator.row_counter, self.iterator.column_counter);
        self.iterator.next().map(|x| ((row, column), x))
    }
}
impl<'a, T: Clone, S: MatrixRef<T>> FusedIterator for WithIndex<RowMajorIterator<'a, T, S>> {}
impl<'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for WithIndex<RowMajorIterator<'a, T, S>> {}

/**
 * An iterator over references to a column in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * Depending on the row iterator you want to obtain,
 * can either iterate through &1, &3 or &2, &4.
 */
#[derive(Debug)]
pub struct ColumnReferenceIterator<'a, T, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column: Column,
    range: Range<Row>,
    _type: PhantomData<&'a T>,
}

impl<'a, T> ColumnReferenceIterator<'a, T> {
    /**
     * Constructs a column iterator over this matrix.
     *
     * # Panics
     *
     * Panics if the column does not exist in this matrix.
     */
    #[track_caller]
    pub fn new(matrix: &Matrix<T>, column: Column) -> ColumnReferenceIterator<T> {
        ColumnReferenceIterator::from(matrix, column)
    }
}

impl<'a, T, S: MatrixRef<T>> ColumnReferenceIterator<'a, T, S> {
    /**
     * Constructs a column iterator over this source.
     *
     * # Panics
     *
     * Panics if the column does not exist in this source.
     */
    #[track_caller]
    pub fn from(source: &S, column: Column) -> ColumnReferenceIterator<T, S> {
        assert!(
            source.index_is_valid(0, column),
            "Expected ({},{}) to be in range",
            0,
            column
        );
        ColumnReferenceIterator {
            matrix: source,
            column,
            range: 0..source.view_rows(),
            _type: PhantomData,
        }
    }
}

impl<'a, T, S: MatrixRef<T>> Iterator for ColumnReferenceIterator<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(row) => unsafe {
                // Safety: We initialised the range to 0..matrix.view_rows(), and
                // checked we can read from this column at creation, hence if
                // the view_rows have not changed this read is in bounds and if
                // they are able to be changed, then the MatrixRef implementation is
                // required to bounds check for us.
                Some(self.matrix.get_reference_unchecked(row, self.column))
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}
impl<'a, T, S: MatrixRef<T>> FusedIterator for ColumnReferenceIterator<'a, T, S> {}
impl<'a, T, S: MatrixRef<T>> ExactSizeIterator for ColumnReferenceIterator<'a, T, S> {}

/**
 * An iterator over references to a row in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * Depending on the row iterator you want to obtain,
 * can either iterate through &1, &2 or &3, &4.
 */
#[derive(Debug)]
pub struct RowReferenceIterator<'a, T, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    row: Row,
    range: Range<Column>,
    _type: PhantomData<&'a T>,
}

impl<'a, T> RowReferenceIterator<'a, T> {
    /**
     * Constructs a row iterator over this matrix.
     *
     * # Panics
     *
     * Panics if the row does not exist in this matrix.
     */
    #[track_caller]
    pub fn new(matrix: &Matrix<T>, row: Row) -> RowReferenceIterator<T> {
        RowReferenceIterator::from(matrix, row)
    }
}

impl<'a, T, S: MatrixRef<T>> RowReferenceIterator<'a, T, S> {
    /**
     * Constructs a row iterator over this source.
     *
     * # Panics
     *
     * Panics if the row does not exist in this source.
     */
    #[track_caller]
    pub fn from(source: &S, row: Row) -> RowReferenceIterator<T, S> {
        assert!(
            source.index_is_valid(row, 0),
            "Expected ({},{}) to be in range",
            row,
            0
        );
        RowReferenceIterator {
            matrix: source,
            row,
            range: 0..source.view_columns(),
            _type: PhantomData,
        }
    }
}

impl<'a, T, S: MatrixRef<T>> Iterator for RowReferenceIterator<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(column) => unsafe {
                // Safety: We initialised the range to 0..matrix.view_columns(), and
                // checked we can read from this row at creation, hence if
                // the view_columns have not changed this read is in bounds and if
                // they are able to be changed, then the MatrixRef implementation is
                // required to bounds check for us.
                Some(self.matrix.get_reference_unchecked(self.row, column))
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}
impl<'a, T, S: MatrixRef<T>> FusedIterator for RowReferenceIterator<'a, T, S> {}
impl<'a, T, S: MatrixRef<T>> ExactSizeIterator for RowReferenceIterator<'a, T, S> {}

/**
 * A column major iterator over references to all values in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as &1, &3, &2, &4
 */
#[derive(Debug)]
pub struct ColumnMajorReferenceIterator<'a, T, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column_counter: Column,
    columns: Column,
    row_counter: Row,
    rows: Row,
    finished: bool,
    _type: PhantomData<&'a T>,
}

impl<'a, T> ColumnMajorReferenceIterator<'a, T> {
    /**
     * Constructs a column major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> ColumnMajorReferenceIterator<T> {
        ColumnMajorReferenceIterator::from(matrix)
    }
}

impl<'a, T, S: MatrixRef<T>> ColumnMajorReferenceIterator<'a, T, S> {
    /**
     * Constructs a column major iterator over this source.
     */
    pub fn from(source: &S) -> ColumnMajorReferenceIterator<T, S> {
        ColumnMajorReferenceIterator {
            matrix: source,
            column_counter: 0,
            columns: source.view_columns(),
            row_counter: 0,
            rows: source.view_rows(),
            finished: !source.index_is_valid(0, 0),
            _type: PhantomData,
        }
    }

    /**
     * Constructors an iterator which also yields the row and column index of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T, S: MatrixRef<T>> Iterator for ColumnMajorReferenceIterator<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        column_major_iter(
            &mut self.finished,
            self.rows,
            self.columns,
            &mut self.row_counter,
            &mut self.column_counter,
        )
        .map(|(row, column)| unsafe {
            // Safety: We checked on creation that 0,0 is in range, and after getting
            // our next value we check if we hit the end of the matrix and will avoid
            // calling this on our next loop if we finished. Hence if the view size has
            // not changed this read is in bounds and if they are able to be changed,
            // then the MatrixRef implementation is required to bounds check for us.
            self.matrix.get_reference_unchecked(row, column)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        column_major_size_hint(
            self.rows,
            self.columns,
            self.row_counter,
            self.column_counter,
        )
    }
}

impl<'a, T, S: MatrixRef<T>> FusedIterator for ColumnMajorReferenceIterator<'a, T, S> {}
impl<'a, T, S: MatrixRef<T>> ExactSizeIterator for ColumnMajorReferenceIterator<'a, T, S> {}

impl<'a, T, S: MatrixRef<T>> Iterator for WithIndex<ColumnMajorReferenceIterator<'a, T, S>> {
    type Item = ((Row, Column), &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let (row, column) = (self.iterator.row_counter, self.iterator.column_counter);
        self.iterator.next().map(|x| ((row, column), x))
    }
}
impl<'a, T, S: MatrixRef<T>> FusedIterator for WithIndex<ColumnMajorReferenceIterator<'a, T, S>> {}
#[rustfmt::skip]
impl<'a, T, S: MatrixRef<T>> ExactSizeIterator for WithIndex<ColumnMajorReferenceIterator<'a, T, S>> {}

/**
 * A row major iterator over references to all values in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as &1, &2, &3, &4
 */
#[derive(Debug)]
pub struct RowMajorReferenceIterator<'a, T, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column_counter: Column,
    columns: Column,
    row_counter: Row,
    rows: Row,
    finished: bool,
    _type: PhantomData<&'a T>,
}

impl<'a, T> RowMajorReferenceIterator<'a, T> {
    /**
     * Constructs a column major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> RowMajorReferenceIterator<T> {
        RowMajorReferenceIterator::from(matrix)
    }
}

impl<'a, T, S: MatrixRef<T>> RowMajorReferenceIterator<'a, T, S> {
    /**
     * Constructs a column major iterator over this source.
     */
    pub fn from(source: &S) -> RowMajorReferenceIterator<T, S> {
        RowMajorReferenceIterator {
            matrix: source,
            column_counter: 0,
            columns: source.view_columns(),
            row_counter: 0,
            rows: source.view_rows(),
            finished: !source.index_is_valid(0, 0),
            _type: PhantomData,
        }
    }

    /**
     * Constructors an iterator which also yields the row and column index of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T, S: MatrixRef<T>> Iterator for RowMajorReferenceIterator<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        row_major_iter(
            &mut self.finished,
            self.rows,
            self.columns,
            &mut self.row_counter,
            &mut self.column_counter,
        )
        .map(|(row, column)| unsafe {
            // Safety: We checked on creation that 0,0 is in range, and after getting
            // our next value we check if we hit the end of the matrix and will avoid
            // calling this on our next loop if we finished. Hence if the view size has
            // not changed this read is in bounds and if they are able to be changed,
            // then the MatrixRef implementation is required to bounds check for us.
            self.matrix.get_reference_unchecked(row, column)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        row_major_size_hint(
            self.rows,
            self.columns,
            self.row_counter,
            self.column_counter,
        )
    }
}
impl<'a, T, S: MatrixRef<T>> FusedIterator for RowMajorReferenceIterator<'a, T, S> {}
impl<'a, T, S: MatrixRef<T>> ExactSizeIterator for RowMajorReferenceIterator<'a, T, S> {}

impl<'a, T, S: MatrixRef<T>> Iterator for WithIndex<RowMajorReferenceIterator<'a, T, S>> {
    type Item = ((Row, Column), &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let (row, column) = (self.iterator.row_counter, self.iterator.column_counter);
        self.iterator.next().map(|x| ((row, column), x))
    }
}
impl<'a, T, S: MatrixRef<T>> FusedIterator for WithIndex<RowMajorReferenceIterator<'a, T, S>> {}
impl<'a, T, S: MatrixRef<T>> ExactSizeIterator for WithIndex<RowMajorReferenceIterator<'a, T, S>> {}

/**
 * An iterator over the main diagonal in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as 1, 4
 *
 * If the matrix is not square this will stop at whichever row/colum is shorter.
 */
#[derive(Debug)]
pub struct DiagonalIterator<'a, T: Clone, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    range: Range<usize>,
    _type: PhantomData<&'a T>,
}

impl<'a, T: Clone> DiagonalIterator<'a, T> {
    /**
     * Constructs a diagonal iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> DiagonalIterator<T> {
        DiagonalIterator::from(matrix)
    }
}

impl<'a, T: Clone, S: MatrixRef<T>> DiagonalIterator<'a, T, S> {
    /**
     * Constructs a diagonal iterator over this source.
     */
    pub fn from(source: &S) -> DiagonalIterator<T, S> {
        DiagonalIterator {
            matrix: source,
            range: 0..std::cmp::min(source.view_rows(), source.view_columns()),
            _type: PhantomData,
        }
    }
}

impl<'a, T: Clone, S: MatrixRef<T>> Iterator for DiagonalIterator<'a, T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(i) => unsafe {
                // Safety: We initialised the range to 0..min(rows/columns), hence if the
                // view size has not changed this read is in bounds and if they are able to
                // be changed, then the MatrixRef implementation is required to bounds
                // check for us.
                Some(self.matrix.get_reference_unchecked(i, i).clone())
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}
impl<'a, T: Clone, S: MatrixRef<T>> FusedIterator for DiagonalIterator<'a, T, S> {}
impl<'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for DiagonalIterator<'a, T, S> {}

/**
 * An iterator over references to the main diagonal in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as &1, &4
 *
 * If the matrix is not square this will stop at whichever row/colum is shorter.
 */
#[derive(Debug)]
pub struct DiagonalReferenceIterator<'a, T, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    range: Range<usize>,
    _type: PhantomData<&'a T>,
}

impl<'a, T> DiagonalReferenceIterator<'a, T> {
    /**
     * Constructs a diagonal iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> DiagonalReferenceIterator<T> {
        DiagonalReferenceIterator::from(matrix)
    }
}

impl<'a, T, S: MatrixRef<T>> DiagonalReferenceIterator<'a, T, S> {
    /**
     * Constructs a diagonal iterator over this source.
     */
    pub fn from(source: &S) -> DiagonalReferenceIterator<T, S> {
        DiagonalReferenceIterator {
            matrix: source,
            range: 0..std::cmp::min(source.view_rows(), source.view_columns()),
            _type: PhantomData,
        }
    }
}

impl<'a, T, S: MatrixRef<T>> Iterator for DiagonalReferenceIterator<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(i) => unsafe {
                // Safety: We initialised the range to 0..min(rows/columns), hence if the
                // view size has not changed this read is in bounds and if they are able to
                // be changed, then the MatrixRef implementation is required to bounds
                // check for us.
                Some(self.matrix.get_reference_unchecked(i, i))
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}
impl<'a, T, S: MatrixRef<T>> FusedIterator for DiagonalReferenceIterator<'a, T, S> {}
impl<'a, T, S: MatrixRef<T>> ExactSizeIterator for DiagonalReferenceIterator<'a, T, S> {}

/**
 * A column major iterator over mutable references to all values in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as &mut 1, &mut 3, &mut 2, &mut 4
 */
#[derive(Debug)]
#[rustfmt::skip]
pub struct ColumnMajorReferenceMutIterator<'a, T, S: MatrixMut<T> + NoInteriorMutability = Matrix<T>> {
    matrix: &'a mut S,
    column_counter: Column,
    columns: Column,
    row_counter: Row,
    rows: Row,
    finished: bool,
    _type: PhantomData<&'a mut T>,
}

impl<'a, T> ColumnMajorReferenceMutIterator<'a, T> {
    /**
     * Constructs a column major iterator over this matrix.
     */
    pub fn new(matrix: &mut Matrix<T>) -> ColumnMajorReferenceMutIterator<T> {
        ColumnMajorReferenceMutIterator::from(matrix)
    }
}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> ColumnMajorReferenceMutIterator<'a, T, S> {
    /**
     * Constructs a column major iterator over this source.
     */
    pub fn from(source: &mut S) -> ColumnMajorReferenceMutIterator<T, S> {
        ColumnMajorReferenceMutIterator {
            column_counter: 0,
            columns: source.view_columns(),
            row_counter: 0,
            rows: source.view_rows(),
            finished: !source.index_is_valid(0, 0),
            matrix: source,
            _type: PhantomData,
        }
    }

    /**
     * Constructors an iterator which also yields the row and column index of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> Iterator
    for ColumnMajorReferenceMutIterator<'a, T, S>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        column_major_iter(
            &mut self.finished,
            self.rows,
            self.columns,
            &mut self.row_counter,
            &mut self.column_counter,
        )
        .map(|(row, column)| unsafe {
            // Safety: We checked on creation that 0,0 is in range, and after getting
            // our next value we check if we hit the end of the matrix and will avoid
            // calling this on our next loop if we finished. Since the view size may not
            // change due to NoInteriorMutability and our exclusive reference this read
            // is in bounds.
            // Safety: We are not allowed to give out overlapping mutable references,
            // but since we will always increment the counter on every call to next()
            // and stop when we reach the end no references will overlap*.
            // The compiler doesn't know this, so transmute the lifetime for it.
            // *We also require the source matrix to be NoInteriorMutability to additionally
            // make illegal any edge cases where some extremely exotic matrix rotates its data
            // inside the buffer around though a shared reference while we were iterating that
            // could otherwise make our cursor read the same data twice.
            std::mem::transmute(self.matrix.get_reference_unchecked_mut(row, column))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        column_major_size_hint(
            self.rows,
            self.columns,
            self.row_counter,
            self.column_counter,
        )
    }
}
#[rustfmt::skip]
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> FusedIterator for ColumnMajorReferenceMutIterator<'a, T, S> {}
#[rustfmt::skip]
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> ExactSizeIterator for ColumnMajorReferenceMutIterator<'a, T, S> {}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> Iterator
    for WithIndex<ColumnMajorReferenceMutIterator<'a, T, S>>
{
    type Item = ((Row, Column), &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let (row, column) = (self.iterator.row_counter, self.iterator.column_counter);
        self.iterator.next().map(|x| ((row, column), x))
    }
}
#[rustfmt::skip]
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> FusedIterator for WithIndex<ColumnMajorReferenceMutIterator<'a, T, S>> {}
#[rustfmt::skip]
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> ExactSizeIterator for WithIndex<ColumnMajorReferenceMutIterator<'a, T, S>> {}

/**
 * A row major iterator over mutable references to all values in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as &mut 1, &mut 2, &mut 3, &mut 4
 */
#[derive(Debug)]
pub struct RowMajorReferenceMutIterator<'a, T, S: MatrixMut<T> + NoInteriorMutability = Matrix<T>> {
    matrix: &'a mut S,
    column_counter: Column,
    columns: Column,
    row_counter: Row,
    rows: Row,
    finished: bool,
    _type: PhantomData<&'a mut T>,
}

impl<'a, T> RowMajorReferenceMutIterator<'a, T> {
    /**
     * Constructs a row major iterator over this matrix.
     */
    pub fn new(matrix: &mut Matrix<T>) -> RowMajorReferenceMutIterator<T> {
        RowMajorReferenceMutIterator::from(matrix)
    }
}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> RowMajorReferenceMutIterator<'a, T, S> {
    /**
     * Constructs a row major iterator over this source.
     */
    pub fn from(source: &mut S) -> RowMajorReferenceMutIterator<T, S> {
        RowMajorReferenceMutIterator {
            column_counter: 0,
            columns: source.view_columns(),
            row_counter: 0,
            rows: source.view_rows(),
            finished: !source.index_is_valid(0, 0),
            matrix: source,
            _type: PhantomData,
        }
    }

    /**
     * Constructors an iterator which also yields the row and column index of each element in
     * this iterator.
     */
    pub fn with_index(self) -> WithIndex<Self> {
        WithIndex { iterator: self }
    }
}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> Iterator
    for RowMajorReferenceMutIterator<'a, T, S>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        row_major_iter(
            &mut self.finished,
            self.rows,
            self.columns,
            &mut self.row_counter,
            &mut self.column_counter,
        )
        .map(|(row, column)| unsafe {
            // Safety: We checked on creation that 0,0 is in range, and after getting
            // our next value we check if we hit the end of the matrix and will avoid
            // calling this on our next loop if we finished. Since the view size may not
            // change due to NoInteriorMutability and our exclusive reference this read
            // is in bounds.
            // then the MatrixRef implementation is required to bounds check for us.
            // Safety: We are not allowed to give out overlapping mutable references,
            // but since we will always increment the counter on every call to next()
            // and stop when we reach the end no references will overlap*.
            // The compiler doesn't know this, so transmute the lifetime for it.
            // *We also require the source matrix to be NoInteriorMutability to additionally
            // make illegal any edge cases where some extremely exotic matrix rotates its data
            // inside the buffer around through a shared reference while we were iterating that
            // could otherwise make our cursor read the same data twice.
            std::mem::transmute(self.matrix.get_reference_unchecked_mut(row, column))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        row_major_size_hint(
            self.rows,
            self.columns,
            self.row_counter,
            self.column_counter,
        )
    }
}
#[rustfmt::skip]
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> FusedIterator for RowMajorReferenceMutIterator<'a, T, S> {}
#[rustfmt::skip]
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> ExactSizeIterator for RowMajorReferenceMutIterator<'a, T, S> {}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> Iterator
    for WithIndex<RowMajorReferenceMutIterator<'a, T, S>>
{
    type Item = ((Row, Column), &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let (row, column) = (self.iterator.row_counter, self.iterator.column_counter);
        self.iterator.next().map(|x| ((row, column), x))
    }
}
#[rustfmt::skip]
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> FusedIterator for WithIndex<RowMajorReferenceMutIterator<'a, T, S>> {}
#[rustfmt::skip]
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> ExactSizeIterator for WithIndex<RowMajorReferenceMutIterator<'a, T, S>> {}

/**
 * An iterator over mutable references to the main diagonal in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as &mut 1, &mut 4
 *
 * If the matrix is not square this will stop at whichever row/colum is shorter.
 */
#[derive(Debug)]
pub struct DiagonalReferenceMutIterator<'a, T, S: MatrixMut<T> + NoInteriorMutability = Matrix<T>> {
    matrix: &'a mut S,
    range: Range<usize>,
    _type: PhantomData<&'a mut T>,
}

impl<'a, T> DiagonalReferenceMutIterator<'a, T> {
    /**
     * Constructs a diagonal iterator over this matrix.
     */
    pub fn new(matrix: &mut Matrix<T>) -> DiagonalReferenceMutIterator<T> {
        DiagonalReferenceMutIterator::from(matrix)
    }
}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> DiagonalReferenceMutIterator<'a, T, S> {
    /**
     * Constructs a diagonal iterator over this source.
     */
    pub fn from(source: &mut S) -> DiagonalReferenceMutIterator<T, S> {
        DiagonalReferenceMutIterator {
            range: 0..std::cmp::min(source.view_rows(), source.view_columns()),
            matrix: source,
            _type: PhantomData,
        }
    }
}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> Iterator
    for DiagonalReferenceMutIterator<'a, T, S>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(i) => unsafe {
                // Safety: We initialised the range to 0..min(rows/columns), hence this read is
                // in bounds because the source is NoInteriorMutability and we hold an exclusive
                // reference to it, so the valid bounds cannot change in size.
                // Safety: We are not allowed to give out overlapping mutable references,
                // but since we will always increment the counter on every call to next()
                // and stop when we reach the end no references will overlap*.
                // The compiler doesn't know this, so transmute the lifetime for it.
                // *We also require the source matrix to be NoInteriorMutability to additionally
                // make illegal any edge cases where some extremely exotic matrix rotates its data
                // inside the buffer around through a shared reference while we were iterating that
                // could otherwise make our cursor read the same data twice.
                std::mem::transmute(self.matrix.get_reference_unchecked_mut(i, i))
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}
#[rustfmt::skip]
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> FusedIterator for DiagonalReferenceMutIterator<'a, T, S> {}
#[rustfmt::skip]
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> ExactSizeIterator for DiagonalReferenceMutIterator<'a, T, S> {}

/**
 * An iterator over mutable references to a column in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * Depending on the row iterator you want to obtain,
 * can either iterate through &mut 1, &mut 3 or &mut 2, &mut 4.
 */
#[derive(Debug)]
pub struct ColumnReferenceMutIterator<'a, T, S: MatrixMut<T> + NoInteriorMutability = Matrix<T>> {
    matrix: &'a mut S,
    column: Column,
    range: Range<Row>,
    _type: PhantomData<&'a mut T>,
}

impl<'a, T> ColumnReferenceMutIterator<'a, T> {
    /**
     * Constructs a column iterator over this matrix.
     *
     * # Panics
     *
     * Panics if the column does not exist in this matrix.
     */
    #[track_caller]
    pub fn new(matrix: &mut Matrix<T>, column: Column) -> ColumnReferenceMutIterator<T> {
        ColumnReferenceMutIterator::from(matrix, column)
    }
}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> ColumnReferenceMutIterator<'a, T, S> {
    /**
     * Constructs a column iterator over this source.
     *
     * # Panics
     *
     * Panics if the column does not exist in this source.
     */
    #[track_caller]
    pub fn from(source: &mut S, column: Column) -> ColumnReferenceMutIterator<T, S> {
        assert!(
            source.index_is_valid(0, column),
            "Expected ({},{}) to be in range",
            0,
            column
        );
        ColumnReferenceMutIterator {
            range: 0..source.view_rows(),
            matrix: source,
            column,
            _type: PhantomData,
        }
    }
}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> Iterator for ColumnReferenceMutIterator<'a, T, S> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(row) => unsafe {
                // Safety: We initialised the range to 0..matrix.view_rows(), and
                // checked we can read from this column at creation, hence this read is
                // in bounds because the source is NoInteriorMutability and we hold an exclusive
                // reference to it, so the valid bounds cannot change in size.
                // Safety: We are not allowed to give out overlapping mutable references,
                // but since we will always increment the counter on every call to next()
                // and stop when we reach the end no references will overlap*.
                // The compiler doesn't know this, so transmute the lifetime for it.
                // *We also require the source matrix to be NoInteriorMutability to additionally
                // make illegal any edge cases where some extremely exotic matrix rotates its data
                // inside the buffer around through a shared reference while we were iterating that
                // could otherwise make our cursor read the same data twice.
                std::mem::transmute(self.matrix.get_reference_unchecked_mut(row, self.column))
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> FusedIterator for ColumnReferenceMutIterator<'a, T, S> {}
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> ExactSizeIterator for ColumnReferenceMutIterator<'a, T, S> {}

/**
 * An iterator over mutable references to a row in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * Depending on the row iterator you want to obtain,
 * can either iterate through &mut 1, &mut 2 or &mut 3, &mut 4.
 */
#[derive(Debug)]
pub struct RowReferenceMutIterator<'a, T, S: MatrixMut<T> + NoInteriorMutability = Matrix<T>> {
    matrix: &'a mut S,
    row: Row,
    range: Range<Column>,
    _type: PhantomData<&'a mut T>,
}

impl<'a, T> RowReferenceMutIterator<'a, T> {
    /**
     * Constructs a row iterator over this matrix.
     *
     * # Panics
     *
     * Panics if the row does not exist in this matrix.
     */
    #[track_caller]
    pub fn new(matrix: &mut Matrix<T>, row: Row) -> RowReferenceMutIterator<T> {
        RowReferenceMutIterator::from(matrix, row)
    }
}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> RowReferenceMutIterator<'a, T, S> {
    /**
     * Constructs a row iterator over this source.
     *
     * # Panics
     *
     * Panics if the row does not exist in this source.
     */
    #[track_caller]
    pub fn from(source: &mut S, row: Row) -> RowReferenceMutIterator<T, S> {
        assert!(
            source.index_is_valid(row, 0),
            "Expected ({},{}) to be in range",
            row,
            0
        );
        RowReferenceMutIterator {
            range: 0..source.view_columns(),
            matrix: source,
            row,
            _type: PhantomData,
        }
    }
}

impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> Iterator for RowReferenceMutIterator<'a, T, S> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(column) => unsafe {
                // Safety: We initialised the range to 0..matrix.view_columns(), and
                // checked we can read from this row at creation, hence this read is
                // in bounds because the source is NoInteriorMutability and we hold an exclusive
                // reference to it, so the valid bounds cannot change in size.
                // Safety: We are not allowed to give out overlapping mutable references,
                // but since we will always increment the counter on every call to next()
                // and stop when we reach the end no references will overlap*.
                // The compiler doesn't know this, so transmute the lifetime for it.
                // *We also require the source matrix to be NoInteriorMutability to additionally
                // make illegal any edge cases where some extremely exotic matrix rotates its data
                // inside the buffer around through a shared reference while we were iterating that
                // could otherwise make our cursor read the same data twice.
                std::mem::transmute(self.matrix.get_reference_unchecked_mut(self.row, column))
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> FusedIterator for RowReferenceMutIterator<'a, T, S> {}
impl<'a, T, S: MatrixMut<T> + NoInteriorMutability> ExactSizeIterator for RowReferenceMutIterator<'a, T, S> {}
