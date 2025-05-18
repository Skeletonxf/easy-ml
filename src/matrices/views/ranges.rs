use crate::matrices::views::{DataLayout, MatrixMut, MatrixRef, NoInteriorMutability};
use crate::matrices::{Column, Row};

use std::marker::PhantomData;
use std::ops::Range;

/**
 * A 2 dimensional range over a matrix, hiding the values **outside** the range from view.
 *
 * The entire source is still owned by the MatrixRange however, so this does not permit
 * creating multiple mutable ranges into a single matrix even if they wouldn't overlap.
 *
 * For non overlapping mutable ranges into a single matrix see
 * [`partition`](crate::matrices::Matrix::partition).
 *
 * See also: [MatrixMask](MatrixMask)
 */
#[derive(Clone, Debug)]
pub struct MatrixRange<T, S> {
    source: S,
    rows: IndexRange,
    columns: IndexRange,
    _type: PhantomData<T>,
}

/**
 * A 2 dimensional mask over a matrix, hiding the values **inside** the range from view.
 *
 * The entire source is still owned by the MatrixMask however, so this does not permit
 * creating multiple mutable masks into a single matrix even if they wouldn't overlap.
 *
 * See also: [MatrixRange](MatrixRange)
 */
#[derive(Clone, Debug)]
pub struct MatrixMask<T, S> {
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
     *
     * NOTE: In previous versions (<=1.8.1), this erroneously did not clip the IndexRange input to
     * not exceed the rows and columns of the source, which led to the possibility to create
     * MatrixRanges that reported a greater number of rows and columns in their shape than their
     * actual data. This function will now correctly clip any ranges that exceed their sources.
     */
    pub fn from<R>(source: S, rows: R, columns: R) -> MatrixRange<T, S>
    where
        R: Into<IndexRange>,
    {
        let max_rows = source.view_rows();
        let max_columns = source.view_columns();
        MatrixRange {
            source,
            rows: {
                let mut rows = rows.into();
                rows.clip(max_rows);
                rows
            },
            columns: {
                let mut columns = columns.into();
                columns.clip(max_columns);
                columns
            },
            _type: PhantomData,
        }
    }

    /**
     * Consumes the MatrixRange, yielding the source it was created from.
     */
    #[allow(dead_code)]
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Gives a reference to the MatrixRange's source (in which the data is not clipped).
     */
    // # Safety
    //
    // Giving out a mutable reference to our source could allow it to be changed out from under us
    // and make our range checks invalid. However, since the source implements MatrixRef
    // interior mutability is not allowed, so we can give out shared references without breaking
    // our own integrity.
    #[allow(dead_code)]
    pub fn source_ref(&self) -> &S {
        &self.source
    }
}

impl<T, S> MatrixMask<T, S>
where
    S: MatrixRef<T>,
{
    /**
     * Creates a new MatrixMask giving a view of only the data outside the row and column
     * [IndexRange](IndexRange)s. If the index range given for rows or columns exceeds the
     * size of the matrix, they will be clipped to fit the actual size without an error.
     *
     * # Examples
     *
     * Creating a view and manipulating a matrix from it.
     * ```
     * use easy_ml::matrices::Matrix;
     * use easy_ml::matrices::views::{MatrixView, MatrixMask};
     * let mut matrix = Matrix::from(vec![
     *     vec![ 2, 3, 4 ],
     *     vec![ 5, 1, 8 ]]);
     * {
     *     let mut view = MatrixView::from(MatrixMask::from(&mut matrix, 0..1, 2..3));
     *     assert_eq!(vec![5, 1], view.row_major_iter().collect::<Vec<_>>());
     *     view.map_mut(|x| x + 10);
     * }
     * assert_eq!(matrix, Matrix::from(vec![
     *     vec![ 2,   3,  4 ],
     *     vec![ 15, 11,  8 ]]));
     * ```
     *
     * Various ways to construct a MatrixMask
     * ```
     * use easy_ml::matrices::Matrix;
     * use easy_ml::matrices::views::{IndexRange, MatrixMask};
     * let matrix = Matrix::from(vec![vec![1]]);
     * let index_range = MatrixMask::from(&matrix, IndexRange::new(0, 4), IndexRange::new(1, 3));
     * let tuple = MatrixMask::from(&matrix, (0, 4), (1, 3));
     * let array = MatrixMask::from(&matrix, [0, 4], [1, 3]);
     * // Note std::ops::Range is start..end not start and length!
     * let range = MatrixMask::from(&matrix, 0..4, 1..4);
     * ```
     */
    pub fn from<R>(source: S, rows: R, columns: R) -> MatrixMask<T, S>
    where
        R: Into<IndexRange>,
    {
        let max_rows = source.view_rows();
        let max_columns = source.view_columns();
        MatrixMask {
            source,
            rows: {
                let mut rows = rows.into();
                rows.clip(max_rows);
                rows
            },
            columns: {
                let mut columns = columns.into();
                columns.clip(max_columns);
                columns
            },
            _type: PhantomData,
        }
    }

    /**
     * Consumes the MatrixMask, yielding the source it was created from.
     */
    #[allow(dead_code)]
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Gives a reference to the MatrixMask's source (in which the data is not masked).
     */
    // # Safety
    //
    // Giving out a mutable reference to our source could allow it to be changed out from under us
    // and make our mask checks invalid. However, since the source implements MatrixRef
    // interior mutability is not allowed, so we can give out shared references without breaking
    // our own integrity.
    #[allow(dead_code)]
    pub fn source_ref(&self) -> &S {
        &self.source
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
 *
 * Creating a Range
 *
 * ```
 * use easy_ml::matrices::views::IndexRange;
 * let range = IndexRange::new(3, 2);
 * let also_range: IndexRange = (3, 2).into();
 * let also_also_range: IndexRange = [3, 2].into();
 * ```
 *
 * NB: You can construct an IndexRange where start+length exceeds isize::MAX or even
 * usize::MAX, however matrices and tensors themselves cannot contain more than isize::MAX
 * elements. Concerned readers should note that on a 64 bit computer this maximum
 * value is 9,223,372,036,854,775,807 so running out of memory is likely to occur first.
 */
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

    // Clips the range or mask to not exceed an index. Note, this may yield 0 length ranges
    // that have non zero starting positions, however map and mask will still calculate correctly.
    pub(crate) fn clip(&mut self, max_index: usize) {
        let end = self.start + self.length;
        let end = std::cmp::min(end, max_index);
        let length = end.saturating_sub(self.start);
        self.length = length;
    }
}

/**
 * Converts from a range of start..end to an IndexRange of start and length
 *
 * NOTE: In previous versions (<=1.8.1) this did not saturate when attempting to subtract the
 * start of the range from the end to calculate the length. It will now correctly produce an
 * IndexRange with a length of 0 if the end is before or equal to the start.
 */
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
 * NOTE: In previous versions (<=1.8.1), this was erroneously implemented as conversion from a
 * tuple of start and end, not start and length as documented.
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
 * NOTE: In previous versions (<=1.8.1), this was erroneously implemented as conversion from an
 * array of start and end, not start and length as documented.
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
        // It is the caller's responsibility to always call with row/column indexes in range,
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

#[test]
fn test_matrix_range_shape_clips() {
    use crate::matrices::Matrix;
    let matrix = Matrix::from(vec![vec![1, 2, 3], vec![4, 5, 6]]);
    let range = MatrixRange::from(&matrix, 0..7, 1..4);
    assert_eq!(2, range.view_rows());
    assert_eq!(2, range.view_columns());
    assert_eq!(2, range.rows.length);
    assert_eq!(2, range.columns.length);
}

// # Safety
//
// Since the MatrixRef we own must implement MatrixRef correctly, so do we by delegating to it,
// as we don't introduce any interior mutability.
/**
 * A MatrixMask of a MatrixRef type implements MatrixRef.
 */
unsafe impl<T, S> MatrixRef<T> for MatrixMask<T, S>
where
    S: MatrixRef<T>,
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        let row = self.rows.mask(row);
        let column = self.columns.mask(column);
        self.source.try_get_reference(row, column)
    }

    fn view_rows(&self) -> Row {
        // We enforce in the constructor that the mask is clipped to the size of our actual
        // matrix, hence the mask cannot be longer than our data in either dimension. If the
        // mask is the same length as our data, we'd return 0 which for MatrixRef is allowed.
        self.source.view_rows() - self.rows.length
    }

    fn view_columns(&self) -> Column {
        // We enforce in the constructor that the mask is clipped to the size of our actual
        // matrix, hence the mask cannot be longer than our data in either dimension. If the
        // mask is the same length as our data, we'd return 0 which for MatrixRef is allowed.
        self.source.view_columns() - self.columns.length
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        // It is the caller's responsibility to always call with row/column indexes in range,
        // therefore calling get_reference_unchecked with indexes beyond the size of the matrix
        // should never happen because on an arbitary MatrixRef it would be undefined behavior.
        let row = self.rows.mask(row);
        let column = self.columns.mask(column);
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
 * A MatrixMask of a MatrixMut type implements MatrixMut.
 */
unsafe impl<T, S> MatrixMut<T> for MatrixMask<T, S>
where
    S: MatrixMut<T>,
{
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        let row = self.rows.mask(row);
        let column = self.columns.mask(column);
        self.source.try_get_reference_mut(row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        // It is the caller's responsibility to always call with row/column indexes in range,
        // therefore calling get_reference_unchecked with indexes beyond the size of the matrix
        // should never happen because on an arbitary MatrixRef it would be undefined behavior.
        let row = self.rows.mask(row);
        let column = self.columns.mask(column);
        self.source.get_reference_unchecked_mut(row, column)
    }
}

// # Safety
//
// Since the NoInteriorMutability we own must implement NoInteriorMutability correctly, so
// do we by delegating to it, as we don't introduce any interior mutability.
/**
 * A MatrixMask of a NoInteriorMutability type implements NoInteriorMutability.
 */
unsafe impl<T, S> NoInteriorMutability for MatrixMask<T, S> where S: NoInteriorMutability {}
