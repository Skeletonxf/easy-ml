/*!
 * Ways to transform and access matrices
 *
 * At the moment slicing is not very usable and can only be used to downsize
 * matrices with [retain](super::Matrix::retain()) and
 * [retain_mut](super::Matrix::retain_mut()). In the future it will be available
 * for iterating through matrices and indexing into matrices to get values.
 */

use std::ops::Range;

use crate::matrices::{Row, Column};

/**
 * A slice defines across one dimension what values are accepted,
 * it can act like a filter. Slices can also be constructed via
 * boolean logic operations in the same way as in predicate logic expressions.
 */
#[non_exhaustive]
pub enum Slice {
    /** A slice that accepts all indexes */
    All(),
    /** A slice that accepts no indexes */
    None(),
    /** A slice that accepts only the provided index */
    Single(usize),
    /** A slice that accepts only indexes within the range */
    Range(Range<usize>),
    /**
     * A slice which rejects all indexes accepted by the argument, and accepts all indexes
     * rejected by the argument.
     */
    Not(Box<Slice>),
    /**
     * A slice which accepts only indexes accepted by both arguments, and rejects all others.
     */
    And(Box<Slice>, Box<Slice>),
    /**
     * A slice which accepts indexes accepted by either arguments, and rejects only
     * indexes accepted by neither. This is an inclusive or.
     *
     * You could construct an exclusive or by using combinations of AND, OR and NOT as
     * (a AND (NOT b)) OR ((NOT a) AND b) = a XOR b.
     */
    Or(Box<Slice>, Box<Slice>),
}

/**
 * A kind of slice that can be taken on a matrix, over its rows and columns.
 */
pub struct Slice2D {
    pub(crate) rows: Slice,
    pub(crate) columns: Slice,
}

impl Slice {
    /**
     * Checks if this slice accepts some index.
     */
    pub fn accepts(&self, index: usize) -> bool {
        match self {
            Slice::All() => true,
            Slice::None() => false,
            Slice::Single(i) => i == &index,
            Slice::Range(range) => range.contains(&index),
            Slice::Not(slice) => !slice.accepts(index),
            Slice::And(slice1, slice2) => slice1.accepts(index) && slice2.accepts(index),
            Slice::Or(slice1, slice2) => slice1.accepts(index) || slice2.accepts(index),
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

/**
 * A builder object to create a slice. This exists to make forgetting to specify rows
 * *and* columns a compilation error rather than a runtime one.
 */
pub struct EmptySlice2DBuilder {}
/**
 * A builder object to create a slice. This exists to make forgetting to specify rows
 * *and* columns a compilation error rather than a runtime one.
 */
pub struct RowSlice2DBuilder { rows: Slice }
/**
 * A builder object to create a slice. This exists to make forgetting to specify rows
 * *and* columns a compilation error rather than a runtime one.
 */
pub struct ColumnSlice2DBuilder { columns: Slice }

/**
 * Constructs a builder object to create a 2d slice
 *
 * The full syntax to create a `Slice2D` is like so:
 *
 * ```
 * use easy_ml::matrices::slices;
 * use easy_ml::matrices::slices::Slice;
 * slices::new()
 *      .rows(Slice::All())
 *      .columns(Slice::Single(1));
 * ```
 *
 * Rows and Column slices can be specified in either order but both must be given.
 */
pub fn new() -> EmptySlice2DBuilder {
    Slice2D::new()
}

impl Slice2D {
    /**
     * Constructs a builder object to create a 2d slice
     *
     * The full syntax to create a `Slice2D` is like so:
     *
     * ```
     * use easy_ml::matrices::slices::{Slice2D, Slice};
     * Slice2D::new()
     *      .rows(Slice::All())
     *      .columns(Slice::Single(1));
     * ```
     *
     * Rows and Column slices can be specified in either order but both must be given.
     */
    pub fn new() -> EmptySlice2DBuilder {
        EmptySlice2DBuilder { }
    }

    /**
     * Checks if this 2 dimensional slice accepts some index. The row and column
     * slices it is composed from must accept the row and column respectively.
     */
    pub fn accepts(&self, row: Row, column: Column) -> bool {
        self.rows.accepts(row) && self.columns.accepts(column)
    }
}

impl EmptySlice2DBuilder {
    /**
     * Constructs a new builder object with the rows defined first.
     */
    pub fn rows(self, rows: Slice) -> RowSlice2DBuilder {
        RowSlice2DBuilder {
            rows,
        }
    }

    /**
     * Constructs a new builder object with the columns defined first.
     */
    pub fn columns(self, columns: Slice) -> ColumnSlice2DBuilder {
        ColumnSlice2DBuilder {
            columns,
        }
    }
}

impl RowSlice2DBuilder {
    /**
     * Constructs a 2d slice with rows and columns defined.
     */
    pub fn columns(self, columns: Slice) -> Slice2D {
        Slice2D {
            rows: self.rows,
            columns,
        }
    }
}

impl ColumnSlice2DBuilder {
    /**
     * Constructs a 2d slice with rows and columns defined.
     */
    pub fn rows(self, rows: Slice) -> Slice2D {
        Slice2D {
            rows,
            columns: self.columns,
        }
    }
}
