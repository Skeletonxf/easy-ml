use crate::matrices::views::{DataLayout, MatrixMut, MatrixRef, NoInteriorMutability};
use crate::matrices::{Column, Row};

use std::marker::PhantomData;
use std::ops::Range;

/**
 * A 2 dimensional range over a matrix, hiding the rest of the matrix data from view.
 *
 * The entire source is still owned by the MatrixRange however, so this does not permit
 * creating multiple mutable ranges into a single matrix even if they wouldn't overlap.
 *
 * For non overlapping mutable ranges into a single matrix see
 * [`partition`](crate::matrices::Matrix::partition).
 */
#[derive(Clone, Debug)]
pub struct MatrixRange<T, S> {
    source: S,
    rows: IndexRange,
    columns: IndexRange,
    _type: PhantomData<T>,
}

impl<T, S> MatrixRange<T, S>
where
    S: MatrixRef<T>,
{
    /**
     * Creates a new MatrixRange giving a view of only the data within the row and column
     * [IndexRange](IndexRange)s.
     *
     * # Examples
     *
     * Creating a view and manipulating a matrix from it.
     * ```
     * use easy_ml::matrices::Matrix;
     * use easy_ml::matrices::views::{MatrixView, MatrixRange};
     * let mut matrix = Matrix::from(vec![
     *     vec![ 2, 3, 4 ],
     *     vec![ 5, 1, 8 ]]);
     * {
     *     let mut view = MatrixView::from(MatrixRange::from(&mut matrix, 0..1, 1..3));
     *     assert_eq!(vec![3, 4], view.row_major_iter().collect::<Vec<_>>());
     *     view.map_mut(|x| x + 10);
     * }
     * assert_eq!(matrix, Matrix::from(vec![
     *     vec![ 2, 13, 14 ],
     *     vec![ 5,  1,  8 ]]));
     * ```
     *
     * Various ways to construct a MatrixRange
     * ```
     * use easy_ml::matrices::Matrix;
     * use easy_ml::matrices::views::{IndexRange, MatrixRange};
     * let matrix = Matrix::from(vec![vec![1]]);
     * let index_range = MatrixRange::from(&matrix, IndexRange::new(0, 4), IndexRange::new(1, 3));
     * let tuple = MatrixRange::from(&matrix, (0, 4), (1, 3));
     * let array = MatrixRange::from(&matrix, [0, 4], [1, 3]);
     * // Note std::ops::Range is start..end not start and length!
     * let range = MatrixRange::from(&matrix, 0..4, 1..4);
     * ```
     */
    pub fn from<R>(source: S, rows: R, columns: R) -> MatrixRange<T, S>
    where
        R: Into<IndexRange>,
    {
        // FIXME: Clamp rows and columns to our source's length! We could report a length we
        // don't actually have otherwise!
        MatrixRange {
            source,
            rows: rows.into(),
            columns: columns.into(),
            _type: PhantomData,
        }
    }
}

/**
 * A range bounded between `start` inclusive and `start + length` exclusive.
 *
 * # Examples
 *
 * Converting between [Range](std::ops::Range) and IndexRange.
 * ```
 * use std::ops::Range;
 * use easy_ml::matrices::views::IndexRange;
 * assert_eq!(IndexRange::new(3, 2), (3..5).into());
 * assert_eq!(IndexRange::new(1, 5), (1..6).into());
 * assert_eq!(IndexRange::new(0, 4), (0..4).into());
 * ```
 */
// TODO: Document all ways of constructing an IndexRange via From/Into impls here
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexRange {
    pub(crate) start: usize,
    pub(crate) length: usize,
}

impl IndexRange {
    pub fn new(start: usize, length: usize) -> IndexRange {
        IndexRange { start, length }
    }

    // TODO: If we make these public we need to disambiguate Range from Mask behaviour better
    /**
     * Maps from a coordinate space of the ith index accessible by this range to the actual index
     * into the entire dimension's data.
     */
    #[inline]
    pub(crate) fn map(&self, index: usize) -> Option<usize> {
        if index < self.length {
            Some(index + self.start)
        } else {
            None
        }
    }

    // NOTE: This doesn't perform bounds checks, adding the length of the mask could push
    // the index out of the valid bounds of the dimension it is for, but if we performed
    // bounds checks here they would be redundant since performing the get with the masked index
    // will bounds check if required
    #[inline]
    pub(crate) fn mask(&self, index: usize) -> usize {
        if index < self.start {
            index
        } else {
            index + self.length
        }
    }

    // Clips the range or mask to not exceed a length. Note, this may yield 0 length ranges
    // that have non zero starting positions, however map and mask will still calculate correctly.
    pub(crate) fn clip(&mut self, to_length: usize) {
        let end = self.start + self.length;
        let end = std::cmp::min(end, to_length);
        let length = end.saturating_sub(self.start);
        self.length = length;
    }
}

/** Converts from a range of start..end to an IndexRange of start and length */
impl From<Range<usize>> for IndexRange {
    fn from(range: Range<usize>) -> IndexRange {
        IndexRange::new(range.start, range.end.saturating_sub(range.start))
    }
}

/** Converts from an IndexRange of start and length to a range of start..end */
impl From<IndexRange> for Range<usize> {
    fn from(range: IndexRange) -> Range<usize> {
        Range {
            start: range.start,
            end: range.start + range.length,
        }
    }
}

/**
 * Converts from a tuple of start and length to an IndexRange
 *
 * NOTE: In previous versions, this was erroneously implemented as conversion from a tuple of
 * start and end, not start and length as documented.
 */
impl From<(usize, usize)> for IndexRange {
    fn from(range: (usize, usize)) -> IndexRange {
        let (start, length) = range;
        IndexRange::new(start, length)
    }
}

/**
 * Converts from an array of start and length to an IndexRange
 *
 * NOTE: In previous versions, this was erroneously implemented as conversion from an array of
 * start and end, not start and length as documented.
 */
impl From<[usize; 2]> for IndexRange {
    fn from(range: [usize; 2]) -> IndexRange {
        let [start, length] = range;
        IndexRange::new(start, length)
    }
}

#[test]
fn test_index_range_clipping() {
    let mut range: IndexRange = (0..6).into();
    range.clip(4);
    assert_eq!(range, (0..4).into());
    let mut range: IndexRange = (1..4).into();
    range.clip(5);
    assert_eq!(range, (1..4).into());
    range.clip(2);
    assert_eq!(range, (1..2).into());
    let mut range: IndexRange = (3..5).into();
    range.clip(2);
    assert_eq!(range, (3..2).into());
    assert_eq!(range.map(0), None);
    assert_eq!(range.map(1), None);
    assert_eq!(range.mask(0), 0);
    assert_eq!(range.mask(1), 1);
}

// # Safety
//
// Since the MatrixRef we own must implement MatrixRef correctly, so do we by delegating to it,
// as we don't introduce any interior mutability.
/**
 * A MatrixRange of a MatrixRef type implements MatrixRef.
 */
unsafe impl<T, S> MatrixRef<T> for MatrixRange<T, S>
where
    S: MatrixRef<T>,
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        let row = self.rows.map(row)?;
        let column = self.columns.map(column)?;
        self.source.try_get_reference(row, column)
    }

    fn view_rows(&self) -> Row {
        self.rows.length
    }

    fn view_columns(&self) -> Column {
        self.columns.length
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        // It is the caller's responsibiltiy to always call with row/column indexes in range,
        // therefore the unwrap() case should never happen because on an arbitary MatrixRef
        // it would be undefined behavior.
        let row = self.rows.map(row).unwrap();
        let column = self.columns.map(column).unwrap();
        self.source.get_reference_unchecked(row, column)
    }

    fn data_layout(&self) -> DataLayout {
        self.source.data_layout()
    }
}

// # Safety
//
// Since the MatrixMut we own must implement MatrixMut correctly, so do we by delegating to it,
// as we don't introduce any interior mutability.
/**
 * A MatrixRange of a MatrixMut type implements MatrixMut.
 */
unsafe impl<T, S> MatrixMut<T> for MatrixRange<T, S>
where
    S: MatrixMut<T>,
{
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        let row = self.rows.map(row)?;
        let column = self.columns.map(column)?;
        self.source.try_get_reference_mut(row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        // It is the caller's responsibiltiy to always call with row/column indexes in range,
        // therefore the unwrap() case should never happen because on an arbitary MatrixRef
        // it would be undefined behavior.
        let row = self.rows.map(row).unwrap();
        let column = self.columns.map(column).unwrap();
        self.source.get_reference_unchecked_mut(row, column)
    }
}

// # Safety
//
// Since the NoInteriorMutability we own must implement NoInteriorMutability correctly, so
// do we by delegating to it, as we don't introduce any interior mutability.
/**
 * A MatrixRange of a NoInteriorMutability type implements NoInteriorMutability.
 */
unsafe impl<T, S> NoInteriorMutability for MatrixRange<T, S> where S: NoInteriorMutability {}
