/*!
 * Ways to slice into a Matrix, viewing only part of the whole data.
 *
 * Not stable or public API yet
 */

// use std::ops::{Deref, DerefMut};
// use std::rc::Rc;
// use std::sync::Arc;
use std::marker::PhantomData;

use crate::matrices::{Row, Column, Matrix};
//use crate::matrices::slices::{Slice, Slice2D};

/**
 * A shared/immutable reference to a matrix of some type.
 */
trait MatrixRef<T> {
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
trait MatrixMut<T>: MatrixRef<T> {
    fn set(&mut self, row: Row, column: Column, value: T);
}
//
// // /**
// //  * An owned matrix such as Box<Matrix<f64>>.
// //  */
// // // TODO: Make a Matrix wrapper type that doesn't have the overheads of a Box as implementing
// // // Deref<Matrix<T>> for Matrix<T> seems wrong.
// // trait MatrixOwned<T>: MatrixMut<T> {
// //     fn get_owned(self) -> Matrix<T>;
// // }
//
// impl <'source, T> MatrixRef<T> for &'source Matrix<T> {
//     #[track_caller]
//     fn get_reference(&self, row: Row, column: Column) -> &T {
//         (*self).get_reference(row, column)
//     }
//
//     fn rows(&self) -> Row {
//         (*self).rows()
//     }
//
//     fn columns(&self) -> Column {
//         (*self).columns()
//     }
// }
//
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
// impl <'source, T> MatrixRef<T> for &'source mut Matrix<T> {
//     #[track_caller]
//     fn get_reference(&self, row: Row, column: Column) -> &T {
//         let matrix = (*self);
//         matrix.get_reference(row, column)
//     }
//
//     fn rows(&self) -> Row {
//         self.rows()
//     }
//
//     fn columns(&self) -> Column {
//         self.columns()
//     }
// }
//
// impl <'source, T> MatrixMut<T> for &'source mut Matrix<T> {
//     #[track_caller]
//     fn set(&mut self, row: Row, column: Column, value: T) {
//         (*self).set(row, column, value)
//     }
// }
//
// impl <T> MatrixRef<T> for Box<Matrix<T>> {
//     #[track_caller]
//     fn get_reference(&self, row: Row, column: Column) -> &T {
//         self.get_reference(row, column)
//     }
//
//     fn rows(&self) -> Row {
//         self.rows()
//     }
//
//     fn columns(&self) -> Column {
//         self.columns()
//     }
// }
//
// impl <T> MatrixMut<T> for Box<Matrix<T>> {
//     #[track_caller]
//     fn set(&mut self, row: Row, column: Column, value: T) {
//         self.set(row, column, value)
//     }
// }
//
// impl <T> MatrixOwned<T> for Box<Matrix<T>> {
//     fn get_owned(self) -> Matrix<T> {
//         *self
//     }
// }

/**
 * A view into some or all of a matrix.
 *
 * A MatrixView is to a Matrix what an slice is to a Vec. Just as a Vec is resizeable
 * and a slice into it is not, and a slice may span only a portion of the total of a Vec,
 * a MatrixView cannot resize its source, and may span only a portion of the total Matrix in
 * each dimension. The main difference is that a slice is more primitive than a Vec,
 * whereas a Matrix is more primitive than a MatrixView.
 */
struct MatrixView<T, S> {
    source: S,
    _type: PhantomData<T>,
}

impl <T, S: MatrixRef<T>> MatrixView<T, S> {
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

impl <T, S: MatrixMut<T>> MatrixView<T, S> {
    fn set(&mut self, row: Row, column: Column, value: T) {
        self.source.set(row, column, value)
    }
}

// enum Lookup {
//     Transpose {
//         rows: IndexLookup,
//         columns: IndexLookup,
//     },
//     Normal {
//         rows: IndexLookup,
//         columns: IndexLookup,
//     }
// }

struct MatrixRange<M> {
    matrix: M,
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

impl <'source, T> MatrixRef<T> for MatrixRange<&'source Matrix<T>> {
    #[track_caller]
    fn get_reference(&self, row: Row, column: Column) -> &T {
        let row = self.rows.map(row).expect("Row out of index");
        let column = self.columns.map(column).expect("Column out of index");
        self.matrix.get_reference(row, column)
    }

    fn rows(&self) -> Row {
        self.rows.length
    }

    fn columns(&self) -> Column {
        self.columns.length
    }
}

impl <'source, T> MatrixRef<T> for MatrixRange<&'source mut Matrix<T>> {
    #[track_caller]
    fn get_reference(&self, row: Row, column: Column) -> &T {
        let row = self.rows.map(row).expect("Row out of index");
        let column = self.columns.map(column).expect("Column out of index");
        self.matrix.get_reference(row, column)
    }

    fn rows(&self) -> Row {
        self.rows.length
    }

    fn columns(&self) -> Column {
        self.columns.length
    }
}

impl <'source, T> MatrixMut<T> for MatrixRange<&'source mut Matrix<T>> {
    #[track_caller]
    fn set(&mut self, row: Row, column: Column, value: T) {
        let row = self.rows.map(row).expect("Row out of index");
        let column = self.columns.map(column).expect("Column out of index");
        self.matrix.set(row, column, value)
    }
}




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

// TODO: Make MatrixQuadrant able to be 4 different MatrixViews that can be mutated independently




// enum IndexLookup {
//     Range {
//         start: usize,
//         length: usize,
//     },
//     Mapping(Vec<usize>),
// }

// impl IndexLookup {
//     /**
//      * Maps from a coordinate space of the ith index accessible by this view to the actual index
//      * into the entire matrix data.
//      */
//     #[inline]
//     fn map(&self, index: usize) -> Option<usize> {
//         match self {
//             IndexLookup::Range {
//                 start,
//                 length,
//             } => {
//                 if index < *length {
//                     Some(index + start)
//                 } else {
//                     None
//                 }
//             },
//             IndexLookup::Mapping(lookup) => {
//                 if index < lookup.len() {
//                     Some(lookup[index])
//                 } else {
//                     None
//                 }
//             },
//         }
//     }
// }

// impl Lookup {
//     /**
//      * Maps from a row and column in the coordinate space of the i'th row/column accessibile by
//      * this view into the actual indexes into the entire matrix data.
//      */
//     #[inline]
//     fn map(&self, row: Row, column: Column) -> Option<(Row, Column)> {
//         match self {
//             Lookup::Normal {
//                 rows,
//                 columns,
//             } => {
//                 let r = rows.map(row);
//                 let c = columns.map(column);
//                 match (r, c) {
//                     (Some(r), Some(c)) => Some((r, c)),
//                     _ => None
//                 }
//             },
//             Lookup::Transpose {
//                 rows,
//                 columns,
//             } => {
//                 let r = rows.map(column);
//                 let c = columns.map(row);
//                 match (r, c) {
//                     (Some(r), Some(c)) => Some((r, c)),
//                     _ => None
//                 }
//             }
//         }
//     }
// }

// fn slice_to_lookup(length: usize, slice: Slice) -> IndexLookup {
//     match slice {
//         // Anything which can be expressed as a range of accepted
//         // indexes (likely most actual use cases of slices) can be
//         // turned into an arithmetic lookup
//         Slice::All() => IndexLookup::Range {
//             start: 0,
//             length,
//         },
//         Slice::None() => IndexLookup::Range {
//             start: 0,
//             length: 0,
//         },
//         Slice::Single(i) => IndexLookup::Range {
//             start: i,
//             length: 1,
//         },
//         Slice::Range(range) => IndexLookup::Range {
//             start: range.start,
//             length: range.end - range.start,
//         },
//         slice => {
//             // For the general case create a lookup table
//             let mut lookup = Vec::with_capacity(length);
//             for i in 0..length {
//                 if slice.accepts(i) {
//                     lookup.push(i);
//                 }
//             }
//             IndexLookup::Mapping(lookup)
//         },
//     }
// }

impl <'source, T: 'source, S: 'source> MatrixView<T, S> {
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
    fn from(source: S) -> MatrixView<T, S>
    where
        S: MatrixRef<T>
    {
        MatrixView {
            source,
            _type: PhantomData,
        }
    }
}

#[test]
fn creating_matrix_views_ref() {
    let matrix = Matrix::from(vec![vec![1.0]]);
    let _ = MatrixView::from(MatrixRange {
        matrix: &matrix,
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
        matrix: &mut matrix,
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

// #[test]
// fn creating_matrix_views_box() {
//     let matrix = Matrix::from(vec![vec![1.0]]);
//     let _ = MatrixView::from(Box::new(matrix), Slice2D::new().rows(Slice::All()).columns(Slice::All()));
// }
