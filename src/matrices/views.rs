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

use crate::matrices::{Column, Row};
use std::marker::PhantomData;

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
* to check `view_rows`/`view_columns`/`view_size` and request only indexes in range.
*
* 2 - `self.view_size()` must equal `(self.view_rows(), self.view_columns())` - as in the provided implementation.
*
* 3 - Either the `view_rows`/`view_columns`/`view_size` that define which indexes are valid may not
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
     * The size of the matrix that this reference can view. This may be smaller than the actual
     * size of the data stored in the matrix.
     */
    fn view_size(&self) -> (Row, Column) {
        (self.view_rows(), self.view_columns())
    }

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
 */
 #[derive(Debug)]
pub struct MatrixView<T, S> {
    source: S,
    _type: PhantomData<T>,
    // TODO: Transposition?
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

    // fn get_reference(&self, row: Row, column: Column) -> &T {
    //     self.source.try_get_reference(row, column)
    // }

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
     * Returns the dimensionality of this matrix view in Row, Column format
     */
    pub fn size(&self) -> (Row, Column) {
        self.source.view_size()
    }
}

/**
 * MatrixView methods which require only read access via a [MatrixRef](MatrixRef) source.
 */
impl <T, S> MatrixView<T, S>
where
    S: MatrixRef<T>,
    T: Clone,
{
    // fn get(&self, row: Row, column: Column) -> T {
    //     self.get_reference(row, column).clone()
    // }
}

/**
 * MatrixView methods which require mutable access via a [MatrixMut](MatrixMut) source.
 */
impl <T, S> MatrixView<T, S>
where
    S: MatrixMut<T> {
    // fn set(&mut self, row: Row, column: Column, value: T) {
    //     self.source.set(row, column, value)
    // }
}

#[test]
fn creating_matrix_views_erased() {
    let matrix = Matrix::from(vec![vec![1.0]]);
    let boxed: Box<dyn MatrixMut<f32>> = Box::new(matrix);
    MatrixView::from(boxed);
}
