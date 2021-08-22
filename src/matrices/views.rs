/*!
 * Generic views into a matrix.
 */

use crate::matrices::{Column, Matrix, Row};

/**
* A shared/immutable reference to a matrix (or a portion of it) of some type.
*
* # Indexing
*
* Valid indexes into a MatrixRef range from 0 inclusive to `view_rows` exclusive for rows and
* from 0 inclusive to `view_columns` for columns. Even if a 4x4 matrix creates some 2x2 MatrixRef
* that can view only its center, the indexes used on the MatrixRef would be 0,0 to 1,1, not
* 1,1 to 2,2 as corresponding on the matrix.
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
* 2 - `self.view_size()` must equal `(self.view_rows(), self.view_columns())` - the provided implementation.
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
