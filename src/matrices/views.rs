/*!
 * Ways to slice into a Matrix, viewing only part of the whole data.
 *
 * Not stable or public API yet
 */

use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::Arc;

use crate::matrices::{Row, Column, Matrix};
use crate::matrices::slices::{Slice, Slice2D};

/**
 * A shared/immutable reference to a matrix of some type such as &Matrix<f64>.
 */
trait MatrixRef<T>: Deref<Target = Matrix<T>> {
    fn get(&self) -> &Matrix<T>;
}

/**
 * A unique/mutable reference to a matrix of some type such as &mut Matrix<f64>.
 */
trait MatrixMut<T>: MatrixRef<T> + DerefMut<Target = Matrix<T>> {
    fn get_mut(&mut self) -> &mut Matrix<T>;
}

/**
 * An owned matrix such as Box<Matrix<f64>>.
 */
// TODO: Make a Matrix wrapper type that doesn't have the overheads of a Box as implementing
// Deref<Matrix<T>> for Matrix<T> seems wrong.
trait MatrixOwned<T>: MatrixMut<T> {
    fn get_owned(self) -> Matrix<T>;
}

impl <'source, T> MatrixRef<T> for &'source Matrix<T> {
    fn get(&self) -> &Matrix<T> {
        self
    }
}

impl <T> MatrixRef<T> for Rc<Matrix<T>> {
    fn get(&self) -> &Matrix<T> {
        self
    }
}

impl <T> MatrixRef<T> for Arc<Matrix<T>> {
    fn get(&self) -> &Matrix<T> {
        self
    }
}

impl <'source, T> MatrixRef<T> for &'source mut Matrix<T> {
    fn get(&self) -> &Matrix<T> {
        self
    }
}

impl <'source, T> MatrixMut<T> for &'source mut Matrix<T> {
    fn get_mut(&mut self) -> &mut Matrix<T> {
        self
    }
}

impl <T> MatrixRef<T> for Box<Matrix<T>> {
    fn get(&self) -> &Matrix<T> {
        self
    }
}

impl <T> MatrixMut<T> for Box<Matrix<T>> {
    fn get_mut(&mut self) -> &mut Matrix<T> {
        self
    }
}

impl <T> MatrixOwned<T> for Box<Matrix<T>> {
    fn get_owned(self) -> Matrix<T> {
        *self
    }
}

/**
 * A view into some or all of a matrix.
 *
 * A MatrixView is to a Matrix what an slice is to a Vec. Just as a Vec is resizeable
 * and a slice into it is not, and a slice may span only a portion of the total of a Vec,
 * a MatrixView cannot resize its source, and may span only a portion of the total Matrix in
 * each dimension. The main difference is that a slice is more primitive than a Vec,
 * whereas a Matrix is more primitive than a MatrixView.
 */
struct MatrixView<M> {
    source: M,
    lookup: Lookup,
}

enum Lookup {
    Transpose {
        rows: IndexLookup,
        columns: IndexLookup,
    },
    Normal {
        rows: IndexLookup,
        columns: IndexLookup,
    }
}

enum IndexLookup {
    Range {
        start: usize,
        length: usize,
    },
    Mapping(Vec<usize>),
}

impl IndexLookup {
    /**
     * Maps from a coordinate space of the ith index accessible by this view to the actual index
     * into the entire matrix data.
     */
    #[inline]
    fn map(&self, index: usize) -> Option<usize> {
        match self {
            IndexLookup::Range {
                start,
                length,
            } => {
                if index < *length {
                    Some(index + start)
                } else {
                    None
                }
            },
            IndexLookup::Mapping(lookup) => {
                if index < lookup.len() {
                    Some(lookup[index])
                } else {
                    None
                }
            },
        }
    }
}

impl Lookup {
    /**
     * Maps from a row and column in the coordinate space of the i'th row/column accessibile by
     * this view into the actual indexes into the entire matrix data.
     */
    #[inline]
    fn map(&self, row: Row, column: Column) -> Option<(Row, Column)> {
        match self {
            Lookup::Normal {
                rows,
                columns,
            } => {
                let r = rows.map(row);
                let c = columns.map(column);
                match (r, c) {
                    (Some(r), Some(c)) => Some((r, c)),
                    _ => None
                }
            },
            Lookup::Transpose {
                rows,
                columns,
            } => {
                let r = rows.map(column);
                let c = columns.map(row);
                match (r, c) {
                    (Some(r), Some(c)) => Some((r, c)),
                    _ => None
                }
            }
        }
    }
}

fn slice_to_lookup(length: usize, slice: Slice) -> IndexLookup {
    match slice {
        // Anything which can be expressed as a range of accepted
        // indexes (likely most actual use cases of slices) can be
        // turned into an arithmetic lookup
        Slice::All() => IndexLookup::Range {
            start: 0,
            length,
        },
        Slice::None() => IndexLookup::Range {
            start: 0,
            length: 0,
        },
        Slice::Single(i) => IndexLookup::Range {
            start: i,
            length: 1,
        },
        Slice::Range(range) => IndexLookup::Range {
            start: range.start,
            length: range.end - range.start,
        },
        slice => {
            // For the general case create a lookup table
            let mut lookup = Vec::with_capacity(length);
            for i in 0..length {
                if slice.accepts(i) {
                    lookup.push(i);
                }
            }
            IndexLookup::Mapping(lookup)
        },
    }
}

impl <'source, M: 'source> MatrixView<M> {
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
    fn from<T>(source: M, slice: Slice2D) -> MatrixView<M>
    where
        T: 'source,
        M: MatrixRef<T>
    {
        let rows = source.rows();
        let columns = source.columns();
        MatrixView {
            source,
            lookup: Lookup::Normal {
                rows: slice_to_lookup(rows, slice.rows),
                columns: slice_to_lookup(columns, slice.columns),
            },
        }
    }
}

#[test]
fn creating_matrix_views_ref() {
    let matrix = Matrix::from(vec![vec![1.0]]);
    let _ = MatrixView::from(&matrix, Slice2D::new().rows(Slice::All()).columns(Slice::All()));
}

#[test]
fn creating_matrix_views_mut() {
    let mut matrix = Matrix::from(vec![vec![1.0]]);
    let _ = MatrixView::from(&mut matrix, Slice2D::new().rows(Slice::All()).columns(Slice::All()));
}

#[test]
fn creating_matrix_views_box() {
    let matrix = Matrix::from(vec![vec![1.0]]);
    let _ = MatrixView::from(Box::new(matrix), Slice2D::new().rows(Slice::All()).columns(Slice::All()));
}
