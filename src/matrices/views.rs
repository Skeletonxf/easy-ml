/*!
 * Ways to slice into a Matrix, viewing only part of the whole data.
 *
 * Not remotely stable API yet
 */

// use std::rc::Rc;
// use std::sync::Arc;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use crate::matrices::{Row, Column, Matrix};
use crate::matrices::slices::Slice;

// TODO: Expose a non panicking API on MatrixRef/MatrixMut so don't have to deal with arbitary
// levels of #[track_caller] and can make an easy to use panicking API on the MatrixView instead

/**
 * A shared/immutable reference to a matrix of some type.
 */
pub trait MatrixRef<T> {
    fn get_reference(&self, row: Row, column: Column) -> &T;
    fn rows(&self) -> Row;
    fn columns(&self) -> Column;

    fn size(&self) -> (Row, Column) {
        (self.rows(), self.columns())
    }
}

/**
 * A unique/mutable reference to a matrix of some type.
 */
pub trait MatrixMut<T>: MatrixRef<T> {
    fn set(&mut self, row: Row, column: Column, value: T);
}

impl <'source, T> MatrixRef<T> for &'source Matrix<T> {
    #[track_caller]
    fn get_reference(&self, row: Row, column: Column) -> &T {
        Matrix::get_reference(self, row, column)
    }

    fn rows(&self) -> Row {
        Matrix::rows(self)
    }

    fn columns(&self) -> Column {
        Matrix::columns(self)
    }
}

// impl <T> MatrixRef<T> for Rc<Matrix<T>> {
//     #[track_caller]
//     fn get_reference(&self, row: Row, column: Column) -> &T {
//         (self.as_ref()).get_reference(row, column)
//     }
//
//     fn rows(&self) -> Row {
//         (self.as_ref()).rows()
//     }
//
//     fn columns(&self) -> Column {
//         (self.as_ref()).columns()
//     }
// }
//
// impl <T> MatrixRef<T> for Arc<Matrix<T>> {
//     #[track_caller]
//     fn get_reference(&self, row: Row, column: Column) -> &T {
//         (self.as_ref()).get_reference(row, column)
//     }
//
//     fn rows(&self) -> Row {
//         (self.as_ref()).rows()
//     }
//
//     fn columns(&self) -> Column {
//         (self.as_ref()).columns()
//     }
// }
//
impl <'source, T> MatrixRef<T> for &'source mut Matrix<T> {
    #[track_caller]
    fn get_reference(&self, row: Row, column: Column) -> &T {
        Matrix::get_reference(self, row, column)
    }

    fn rows(&self) -> Row {
        Matrix::rows(self)
    }

    fn columns(&self) -> Column {
        Matrix::columns(self)
    }
}

impl <'source, T> MatrixMut<T> for &'source mut Matrix<T> {
    #[track_caller]
    fn set(&mut self, row: Row, column: Column, value: T) {
        Matrix::set(self, row, column, value)
    }
}

impl <T> MatrixRef<T> for Matrix<T> {
    #[track_caller]
    fn get_reference(&self, row: Row, column: Column) -> &T {
        Matrix::get_reference(self, row, column)
    }

    fn rows(&self) -> Row {
        Matrix::rows(self)
    }

    fn columns(&self) -> Column {
        Matrix::columns(self)
    }
}

impl <T> MatrixMut<T> for Matrix<T> {
    #[track_caller]
    fn set(&mut self, row: Row, column: Column, value: T) {
        Matrix::set(self, row, column, value)
    }
}

impl <T, S> MatrixRef<T> for Box<S>
where
    S: MatrixRef<T>
{
    #[track_caller]
    fn get_reference(&self, row: Row, column: Column) -> &T {
        self.as_ref().get_reference(row, column)
    }

    fn rows(&self) -> Row {
        self.as_ref().rows()
    }

    fn columns(&self) -> Column {
        self.as_ref().columns()
    }
}

impl <T, S> MatrixMut<T> for Box<S>
where
    S: MatrixMut<T>
{
    #[track_caller]
    fn set(&mut self, row: Row, column: Column, value: T) {
        self.as_mut().set(row, column, value)
    }
}

impl <T> MatrixRef<T> for Box<dyn MatrixRef<T>> {
    #[track_caller]
    fn get_reference(&self, row: Row, column: Column) -> &T {
        self.as_ref().get_reference(row, column)
    }

    fn rows(&self) -> Row {
        self.as_ref().rows()
    }

    fn columns(&self) -> Column {
        self.as_ref().columns()
    }
}

impl <T> MatrixRef<T> for Box<dyn MatrixMut<T>> {
    #[track_caller]
    fn get_reference(&self, row: Row, column: Column) -> &T {
        self.as_ref().get_reference(row, column)
    }

    fn rows(&self) -> Row {
        self.as_ref().rows()
    }

    fn columns(&self) -> Column {
        self.as_ref().columns()
    }
}

impl <T> MatrixMut<T> for Box<dyn MatrixMut<T>> {
    #[track_caller]
    fn set(&mut self, row: Row, column: Column, value: T) {
        self.as_mut().set(row, column, value)
    }
}

/**
 * A view into some or all of a matrix.
 *
 * A MatrixView is to a Matrix what an slice is to a Vec. Just as a Vec is resizeable
 * and a slice into it is not, and a slice may span only a portion of the total of a Vec,
 * a MatrixView cannot resize its source, and may span only a portion of the total Matrix in
 * each dimension. The main difference is that a slice is more primitive than a Vec,
 * whereas a Matrix is more primitive than a MatrixView. A MatrixView is generic not only
 * over the type of the data in the Matrix, but also over the way the Matrix is 'sliced'
 * and the two are orthogonal to each other.
 */
 #[derive(Debug)]
struct MatrixView<T, S> {
    source: S,
    _type: PhantomData<T>,
    // TODO: Transposition
}

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
     *
     * TODO: Uncomment once APIs are public to let doc test run
     * //```
     * //use easy_ml::matrices::Matrix;
     * //use easy_ml::matrices::views::MatrixView;
     * //use easy_ml::matrices::slices::{Slice, Slice2D};
     * //let matrix = Matrix::from(vec![vec![1.0]]);
     * //let _ = MatrixView::from(&matrix, Slice2D::new().rows(Slice::All()).columns(Slice::All()));
     * //let mut matrix = Matrix::from(vec![vec![1.0]]);
     * //let _ = MatrixView::from(&mut matrix, Slice2D::new().rows(Slice::All()).columns(Slice::All()));
     * //```
     */
    fn from(source: S) -> MatrixView<T, S> {
        MatrixView {
            source,
            _type: PhantomData,
        }
    }

    /**
     * Consumes the matrix view, yielding the source it was created from.
     */
    fn source(self) -> S {
        self.source
    }

    fn get_reference(&self, row: Row, column: Column) -> &T {
        self.source.get_reference(row, column)
    }

    fn rows(&self) -> Row {
        self.source.rows()
    }

    fn columns(&self) -> Column {
        self.source.columns()
    }

    fn size(&self) -> (Row, Column) {
        (self.rows(), self.columns())
    }
}

impl <T, S> MatrixView<T, S>
where
    S: MatrixRef<T>,
    T: Clone,
{
    fn get(&self, row: Row, column: Column) -> T {
        self.get_reference(row, column).clone()
    }
}

impl <T, S> MatrixView<T, S>
where
    S: MatrixMut<T> {
    fn set(&mut self, row: Row, column: Column, value: T) {
        self.source.set(row, column, value)
    }
}

/**
 * A range of a matrix, hiding the rest of the matrix data from view. The entire source
 * is still owned by the MatrixRange however, so this does not permit creating multiple
 * mutable ranges into a single matrix.
 */
struct MatrixRange<S> {
    source: S,
    rows: IndexRange,
    columns: IndexRange,
}

struct IndexRange {
    start: usize,
    length: usize,
}

impl IndexRange {
    /**
     * Maps from a coordinate space of the ith index accessible by this range to the actual index
     * into the entire matrix data.
     */
    #[inline]
    fn map(&self, index: usize) -> Option<usize> {
        if index < self.length {
            Some(index + self.start)
        } else {
            None
        }
    }
}

impl <T, S> MatrixRef<T> for MatrixRange<S>
where
    S: MatrixRef<T>
{
    #[track_caller]
    fn get_reference(&self, row: Row, column: Column) -> &T {
        let row = self.rows.map(row).expect("Row out of index");
        let column = self.columns.map(column).expect("Column out of index");
        self.source.get_reference(row, column)
    }

    fn rows(&self) -> Row {
        self.rows.length
    }

    fn columns(&self) -> Column {
        self.columns.length
    }
}

impl <T, S> MatrixMut<T> for MatrixRange<S>
where
    S: MatrixMut<T>
{
    #[track_caller]
    fn set(&mut self, row: Row, column: Column, value: T) {
        let row = self.rows.map(row).expect("Row out of index");
        let column = self.columns.map(column).expect("Column out of index");
        self.source.set(row, column, value)
    }
}

// TODO: Make MatrixQuadrant able to be 4 different MatrixMut that can be mutated independently
pub(crate) struct MatrixQuadrant<'source, T> {
    top_left: Vec<&'source mut [T]>,
    top_right: Vec<&'source mut [T]>,
    bottom_left: Vec<&'source mut [T]>,
    bottom_right: Vec<&'source mut [T]>,
    rows: Row,
    columns: Column,
    row: Row,
    column: Column,
}

impl <'source, T> MatrixQuadrant<'source, T> {
    pub(crate) fn from_slices(
        top_left: Vec<&'source mut [T]>,
        top_right: Vec<&'source mut [T]>,
        bottom_left: Vec<&'source mut [T]>,
        bottom_right: Vec<&'source mut [T]>,
        rows: Row,
        columns: Column,
        row: Row,
        column: Column,
    ) -> MatrixQuadrant<'source, T> {
        MatrixQuadrant {
            top_left,
            top_right,
            bottom_left,
            bottom_right,
            rows,
            columns,
            row,
            column,
        }
    }
}

/**
 * An error indicating failure to convert a slice to an IndexRange.
 */
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct SliceToIndexRangeError;

impl Error for SliceToIndexRangeError {}

impl fmt::Display for SliceToIndexRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Slice cannot be converted to an IndexRange")
    }
}

/**
 * Converts a basic slice into an IndexRange, erroring if the slice is composed of
 * other slices as these are not representable by a single contiguous range in the
 * general case.
 */
fn slice_to_index_range(length: usize, slice: Slice) -> Result<IndexRange, SliceToIndexRangeError> {
    match slice {
        // Anything which can be expressed as a range of accepted
        // indexes (likely most actual use cases of slices) can be
        // turned into an arithmetic lookup
        Slice::All() => Ok(IndexRange {
            start: 0,
            length,
        }),
        Slice::None() => Ok(IndexRange {
            start: 0,
            length: 0,
        }),
        Slice::Single(i) => Ok(IndexRange {
            start: i,
            length: 1,
        }),
        Slice::Range(range) => Ok(IndexRange {
            start: range.start,
            length: range.end - range.start,
        }),
        _ => Err(SliceToIndexRangeError),
    }
}

#[test]
fn creating_matrix_views_ref() {
    let matrix = Matrix::from(vec![vec![1.0]]);
    let _ = MatrixView::from(MatrixRange {
        source: &matrix,
        rows: IndexRange {
            start: 0,
            length: 1,
        },
        columns: IndexRange {
            start: 0,
            length: 1,
        },
    });
}

#[test]
fn creating_matrix_views_mut() {
    let mut matrix = Matrix::from(vec![vec![1.0]]);
    let _ = MatrixView::from(MatrixRange {
        source: &mut matrix,
        rows: IndexRange {
            start: 0,
            length: 1,
        },
        columns: IndexRange {
            start: 0,
            length: 1,
        },
    });
}

#[test]
fn creating_matrix_views_erased() {
    let matrix = Matrix::from(vec![vec![1.0]]);
    let boxed: Box<dyn MatrixMut<f32>> = Box::new(matrix);
    let mut view = MatrixView::from(boxed);
    view.get(0, 0);
    view.set(0, 0, 2.0);
}
