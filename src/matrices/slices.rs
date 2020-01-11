/*!
 * Ways to transform and access matrices
 */

use std::ops::Range;

use crate::matrices::{Row, Column};

/**
 * A kind of slice that can be taken on a matrix. This enum is not
 * stabalised and may gain new variants over time.
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
    // These can be used to construct jagged slices!
    // need to work out a way to accomodate for this or change design
    // /**
    //  * The negation of a slice, transforming all rows and columns previously
    //  * accepted by the slice into ones not accepted, and all rows and columns
    //  * previously rejected into ones accepted.
    //  */
    // Not(Box<Slice>),
    // /**
    //  * The logical and of two slices, accepting only rows and columns accepted
    //  * by both.
    //  */
    //  And(Box<Slice>, Box<Slice>),
    //  /**
    //   * The logical or of two slices, accepting rows and columns accepted
    //   * by either .
    //   */
    //  Or(Box<Slice>, Box<Slice>)
}

impl Slice {
    /**
     * Checks this slice as a filter for some row and column.
     */
    pub fn is_in(&self, row: Row, column: Column) -> bool {
        match self {
            Slice::SingleRow(r) => r == &row,
            Slice::SingleColumn(c) => c == &column,
            Slice::ColumnRange(column_range) => column_range.contains(&column),
            Slice::RowRange(row_range) => row_range.contains(&row),
            Slice::RowColumnRange(row_range, column_range) => {
                row_range.contains(&row) && column_range.contains(&column)
            },
            // Slice::Not(slice) => !slice.is_in(row, column),
            // Slice::And(slice1, slice2) => slice1.is_in(row, column) && slice2.is_in(row, column),
            // Slice::Or(slice1, slice2) => slice1.is_in(row, column) || slice2.is_in(row, column),
        }
    }
    //
    // /**
    //  * Returns the negation of this slice
    //  */
    // pub fn not(self) -> Slice {
    //     Slice::Not(Box::new(self))
    // }
    //
    // /**
    //  * Returns the and of this slice and the other one
    //  */
    // pub fn add(self, other: Slice) -> Slice {
    //     Slice::And(Box::new(self), Box::new(other))
    // }
    //
    // /**
    //  * Returns the or of this slice and the other one
    //  */
    // pub fn or(self, other: Slice) -> Slice {
    //     Slice::Or(Box::new(self), Box::new(other))
    // }
}
