/*!
 * Ways to transform and access matrices
 *
 * At the moment slicing is not very usable and can only be used to downsize
 * matrices with [retain](../struct.Matrix.html#method.retain) and
 * [retain_mut](../struct.Matrix.html#method.retain_mut). In the future it will be available
 * for iterating through matrices as well.
 */

use std::ops::Range;

use crate::matrices::{Row, Column};

/**
 * A kind of slice that can be taken on a matrix. This enum is not
 * stabilised and may gain new variants over time.
 */
#[derive(Clone)]
pub enum Slice {
    /**
     * A slice into a single row.
     */
    SingleRow(Row),
    /**
     * A slice into a single column.
     */
    SingleColumn(Column),
    /**
     * A slice into some range over the columns.
     */
    ColumnRange(Range<Column>),
    /**
     * A slice into some range over the rows.
     */
    RowRange(Range<Row>),
    /**
     * A slice into a range of rows and a range of columns.
     */
    RowColumnRange(Range<Row>, Range<Column>),
    /**
     * The negation of a slice, transforming all rows and columns previously
     * accepted by the slice into ones not accepted, and all rows and columns
     * previously rejected into ones accepted.
     *
     * This may not do what you expect! Rows and Columns are handled separately
     * to prevent the construction of jagged slices.
     *
     * For a 3x3 array and an existing slice accepting only the middle element,
     * negating this slice will not return a new slice accepting everything
     * but the middle element, but will instead take a minor accepting only
     * the elements that are not in the row or column of the previously accepted
     * element.
     *
     * <pre><code>  [
     *     ✗, ✗, ✗
     *     ✗, ✓, ✗
     *     ✗. ✗. ✗
     *   ]
     * </code></pre>
     *
     * <pre><code>  [
     *     ✓, ✗, ✓
     *     ✗, ✗, ✗
     *     ✓. ✗. ✓
     *   ]
     * </code></pre>
     *
     * This can loosly be considered `!R & !C` in terms of what is negated.
     */
    Not(Box<Slice>),
    // NotRow(Box<Slice>),
    // NotColumn(Box<Slice>),
    /**
     * The logical and of two slices, accepting only rows and columns accepted
     * by both.
     *
     * This can loosly be considered `R1 & R2 & C1 & C2` in terms of what is accepted.
     */
    And(Box<Slice>, Box<Slice>),
    /**
     * The logical or of two slices, accepting any row and column where one of the two
     * slices accept that row and one of the two slices accept that column.
     *
     * This may not do what you expect! Rows and Columns are handled separately
     * to prevent the construction of jagged slices.
     *
     * For the slices of the 3x3 matrix:
     *
     * <pre><code>  [                [
     *     ✓, ✓, ✓         ✗, ✗, ✗
     *     ✗, ✗, ✗         ✗, ✗, ✗
     *     ✗. ✗. ✗         ✓, ✗, ✗
     *   ]                ]
     * </code></pre>
     *
     * The or of them is
     *
     * <pre><code>  [
     *     ✓, ✓, ✓
     *     ✗, ✗, ✗
     *     ✓. ✓. ✓
     *   ]
     * </code></pre>
     *
     * This can loosly be considered `(R1 || R2) & (C1 || C2)` in terms of what is accepted.
     */
    Or(Box<Slice>, Box<Slice>),
}

impl Slice {
    /**
     * Checks this slice as a filter for some row and column.
     */
    pub fn accepts(&self, row: Row, column: Column) -> bool {
        match self {
            Slice::SingleRow(r) => r == &row,
            Slice::SingleColumn(c) => c == &column,
            Slice::ColumnRange(column_range) => column_range.contains(&column),
            Slice::RowRange(row_range) => row_range.contains(&row),
            Slice::RowColumnRange(row_range, column_range) => {
                row_range.contains(&row) && column_range.contains(&column)
            },
            Slice::Not(slice) => (!slice.accepts_row(row)) && (!slice.accepts_column(column)),
            Slice::And(slice1, slice2) => {
                slice1.accepts(row, column) && slice2.accepts(row, column)
            },
            Slice::Or(slice1, slice2) => {
                (slice1.accepts_row(row) || slice2.accepts_row(row))
                && (slice1.accepts_column(column) || slice2.accepts_column(column))
            },
        }
    }

    fn accepts_row(&self, row: Row) -> bool {
        match self {
            Slice::SingleRow(r) => r == &row,
            Slice::SingleColumn(_) => true,
            Slice::ColumnRange(_) => true,
            Slice::RowRange(row_range) => row_range.contains(&row),
            Slice::RowColumnRange(row_range, _) => row_range.contains(&row),
            Slice::Not(slice) => !slice.accepts_row(row),
            Slice::And(slice1, slice2) => {
                slice1.accepts_row(row) && slice2.accepts_row(row)
            },
            Slice::Or(slice1, slice2) => {
                slice1.accepts_row(row) || slice2.accepts_row(row)
            },
        }
    }

    fn accepts_column(&self, column: Column) -> bool {
        match self {
            Slice::SingleRow(_) => true,
            Slice::SingleColumn(c) => c == &column,
            Slice::ColumnRange(column_range) => column_range.contains(&column),
            Slice::RowRange(_) => true,
            Slice::RowColumnRange(_, column_range) => column_range.contains(&column),
            Slice::Not(slice) => !slice.accepts_column(column),
            Slice::And(slice1, slice2) => {
                slice1.accepts_column(column) && slice2.accepts_column(column)
            },
            Slice::Or(slice1, slice2) => {
                slice1.accepts_column(column) || slice2.accepts_column(column)
            },
        }
    }

    /**
     * Returns the negation of this slice
     */
    pub fn not(self) -> Slice {
        Slice::Not(Box::new(self))
    }

    /**
     * Returns the and of this slice and the other one
     */
    pub fn and(self, other: Slice) -> Slice {
        Slice::And(Box::new(self), Box::new(other))
    }

    /**
     * Returns the or of this slice and the other one
     */
    pub fn or(self, other: Slice) -> Slice {
        Slice::Or(Box::new(self), Box::new(other))
    }
}
