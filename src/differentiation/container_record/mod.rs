use crate::differentiation::functions::{Division, FunctionDerivative, Multiplication};
use crate::differentiation::iterators::{
    AsRecords, InconsistentHistory, InvalidRecordIteratorError,
};
use crate::differentiation::record_operations;
use crate::differentiation::Record;
use crate::differentiation::{Derivatives, Index, Primitive, WengertList};
use crate::matrices::iterators::{
    ColumnMajorIterator, RowMajorIterator, RowMajorOwnedIterator, RowMajorReferenceMutIterator,
};
use crate::matrices::views::{MatrixMut, MatrixRef, MatrixView, NoInteriorMutability};
use crate::matrices::{Column, Matrix, Row};
use crate::numeric::{Numeric, NumericRef};
use crate::tensors::indexing::{
    TensorAccess, TensorIterator, TensorOwnedIterator, TensorReferenceMutIterator,
};
use crate::tensors::views::{DataLayout, TensorMut, TensorRef, TensorRename, TensorView};
use crate::tensors::{Dimension, Tensor};

mod container_operations;
pub mod iterators;

/**
 * A pluralisation of [Record](crate::differentiation::Record) that groups together a
 * **s**ource of numbers of type T and stores the WengertList only once.
 *
 * Typically you would refer to one of the type aliases to disambiguate the type of `S` and
 * use more succinct generics: [RecordMatrix](RecordMatrix), [RecordTensor](RecordTensor).
 *
 * For both Matrix and Tensor source types, the containers implement [`+`](std::ops::Add) and
 * [`-`](std::ops::Sub) and have the methods `elementwise_multiply` and `elementwise_divide`.
 * In all cases the containers must have the same size for the operation and will panic if
 * mismatched.
 * [`*`](std::ops::Mul) is also implemented for 2 dimensional tensors and matrices as matrix
 * multiplication.
 *
 * For convenience, as with Trace and Record, many unary operations including
 * [Cos](crate::numeric::extra::Cos), [Exp](crate::numeric::extra::Exp),
 * [Ln](crate::numeric::extra::Ln), [Neg](std::ops::Neg), [Pow](crate::numeric::extra::Pow),
 * [Sin](crate::numeric::extra::Sin), and [Sqrt](crate::numeric::extra::Sqrt) are implemented as
 * well, applying the unary function to each element in the tensor.
 *
 * `+`, `-`, `*` and `/` operations with a RecordContainer and a scalar are also implemented,
 * treating the right hand side scalar as a constant. These are also unary functions in terms of
 * the derivative tracking, for example `X + 5` applies the function `+5` to each element in
 * `X`. Due to the orphan rule, the standard library scalars cannot be implemented for a left hand
 * side scalar, see [SwappedOperations](crate::differentiation::record_operations::SwappedOperations).
 *
 * Both [RecordMatrix](RecordMatrix) and [RecordTensor](RecordTensor) implement
 * [MatrixRef](MatrixRef) and [TensorRef](TensorRef) respectively, which provide read and write
 * access to the underlying numbers and indexes into the WengertList. These APIs along with the
 * `from_existing` constructors for RecordMatrix, RecordTensor, and Record allow arbitary
 * manipulation of specific elements in a record container if needed. However, any arithmetic
 * performed on the raw data won't be tracked on the WengertList and overwriting data within a
 * record container already tracked on the WengertList could result in nonsense. These APIs exist
 * for easy read access to check the data in a record container and for read/write access when
 * manipulating the shape of a record container, and are designed to be used only for moving data
 * around - you should put it back unchanged in a RecordContainer or Record before doing further
 * arithmetic that needs to be tracked on the WengertList.
 *
 * If you just want to manipulate the data in record containers as if they were Records you can
 * use the iterator APIs of [AsRecords](AsRecords) instead and collect them back into containers
 * when you're done.
 */
// TODO: Docs on here for making shapes match up.
#[derive(Debug)]
pub struct RecordContainer<'a, T: Primitive, S, const D: usize> {
    // Opted to store the indexes alongside each number (T, Index) for a number of reasons, the
    // main factor being it makes implementing TensorRef feasible so can utilise the range of
    // existing APIs for Tensor manipulation. It's theoretically possible to only store the first
    // index and calculate the rest, since most of the time all indexes are ascending entries in
    // the WengertList but this would also massively complicate the implementation, especially for
    // handling non row-major operations such as matrix multiplication. It's also not super clear
    // that this would be more efficient because it turns reads into more arithmetic rather
    // than avoiding any work. Just lifting the WengertList out of the tensor should have
    // meaningful improvements to cache line efficiency, and failing that still disallows
    // very questionable states from existing.
    numbers: S,
    history: Option<&'a WengertList<T>>,
}

/**
 * Alias for succinctly referring to RecordContainers backed by a matrix.
 */
pub type RecordMatrix<'a, T, S> = RecordContainer<'a, T, MatrixView<(T, Index), S>, 2>;

/**
 * Alias for succinctly referring to RecordContainers backed by a tensor.
 */
pub type RecordTensor<'a, T, S, const D: usize> =
    RecordContainer<'a, T, TensorView<(T, Index), S, D>, D>;

fn calculate_incrementing_indexes(starting_index: usize, total: usize) -> Vec<Index> {
    let mut indexes = vec![0; total];
    for (i, x) in indexes.iter_mut().enumerate() {
        *x = starting_index + i;
    }
    indexes
}

impl<'a, T, const D: usize> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
{
    /**
     * Creates multiple untracked Records which have no backing WengertList.
     *
     * This is provided for using constants along with Records in operations.
     *
     * For example with `Y = X + 4` the computation graph could be conceived as many
     * `Y[i,j]` nodes with parent nodes of `X[i,j]` and 4 combined with the operation `+`.
     * However there is no need to record the derivatives of a constant, so
     * instead the computation graph can be conceived as `Y[i,j]` nodes each with a single
     * parent node of `X[i,j]` and the unary operation of `+4`.
     */
    pub fn constants<S>(c: S) -> Self
    where
        S: TensorMut<T, D>,
    {
        RecordContainer {
            numbers: TensorView::from(Tensor::from(
                c.view_shape(),
                TensorOwnedIterator::from_numeric(c)
                    .map(|x| (x, 0))
                    .collect(),
            )),
            history: None,
        }
    }

    /**
     * Creates multiple records backed by the provided WengertList.
     *
     * The records cannot live longer than the WengertList, hence
     * the following example does not compile
     *
     * ```compile_fail
     * use easy_ml::differentiation::RecordTensor;
     * use easy_ml::differentiation::WengertList;
     * use easy_ml::tensors::Tensor;
     * let record = {
     *     let list = WengertList::new();
     *     RecordTensor::variables(
     *         &list,
     *         Tensor::from([("r", 2), ("c", 2)], vec![ 1.0, 2.0, 3.0, 4.0 ])
     *     )
     * }; // list no longer in scope
     * ```
     */
    pub fn variables<S>(history: &'a WengertList<T>, x: S) -> Self
    where
        S: TensorMut<T, D>,
    {
        let total = crate::tensors::dimensions::elements(&x.view_shape());
        let starting_index = history.append_nullary_repeating(total);
        RecordContainer {
            numbers: TensorView::from(Tensor::from(
                x.view_shape(),
                TensorOwnedIterator::from_numeric(x)
                    .zip(calculate_incrementing_indexes(starting_index, total))
                    .collect(),
            )),
            history: Some(history),
        }
    }
}

impl<'a, T, S, const D: usize> RecordTensor<'a, T, S, D>
where
    T: Numeric + Primitive,
    S: TensorRef<(T, Index), D>,
{
    /**
     * Returns the number of elements stored by this container's source.
     *
     * For a 2 x 3 Tensor, this would return 6, and for a 2 x 3 x 4 Tensor this would return 24
     * and so on.
     *
     * see also [dimensions::elements](crate::tensors::dimensions::elements)
     */
    pub fn elements(&self) -> usize {
        crate::tensors::dimensions::elements(&self.numbers.shape())
    }

    /**
     * The shape of this container's source.
     */
    pub fn shape(&self) -> [(Dimension, usize); D] {
        self.numbers.shape()
    }

    /**
     * Creates a container from constants/variables directly, most likely obtained by getting a
     * tensor view of an existing container. **The inputs are not checked for validity**. It is
     * possible to pass in the wrong Wengert list here or even numbers with indexes that aren't
     * tracked on the WengertList.
     *
     * It is recommended to use this constructor only in conjunction with
     * resizing or masking an existing container and not for creating new variables. Any variables
     * created outside of `RecordContainer::variables` would have to be manually added to the
     * correct Wengert list, and any arithmetic operations would also need tracking correctly.
     *
     * ```
     * use easy_ml::differentiation::RecordTensor;
     * use easy_ml::differentiation::WengertList;
     * use easy_ml::tensors::Tensor;
     * use easy_ml::tensors::views::{TensorView, TensorRange};
     *
     * let list = WengertList::new();
     * let x = RecordTensor::variables(
     *     &list,
     *     Tensor::from_fn([("x", 2), ("y", 2)], |[r, c]| ((r + 3) * (c + 2)) as f64)
     * );
     * // oh no wrong shape!
     * let fixed = TensorView::from(TensorRange::from(x, [("y", 0..1)]).unwrap()); // we can unwrap here because we know the range is valid
     * let x = RecordTensor::from_existing(Some(&list), fixed);
     * assert_eq!([("x", 2), ("y", 1)], x.shape());
     * ```
     */
    pub fn from_existing(
        history: Option<&'a WengertList<T>>,
        numbers: TensorView<(T, Index), S, D>,
    ) -> Self {
        RecordContainer { numbers, history }
    }

    /**
     * Returns a record tensor with the dimension names of the shape renamed to the provided
     * dimensions. The data of this container and the dimension lengths and order remain unchanged.
     *
     * This is a shorthand for constructing the RecordTensor via manipulating this TensorView. See
     * [`RecordTensor::from_existing`](RecordTensor::from_existing).
     *
     * # Panics
     *
     * If a dimension name is not unique
     *
     * ```
     * use easy_ml::differentiation::RecordTensor;
     * use easy_ml::differentiation::WengertList;
     * use easy_ml::tensors::Tensor;
     * use easy_ml::tensors::views::{TensorView, TensorRename};
     *
     * let list = WengertList::new();
     * let x = RecordTensor::variables(
     *     &list,
     *     Tensor::from_fn([("x", 2), ("y", 2)], |[r, c]| ((r + 3) * (c + 2)) as f64)
     * );
     * // oh no wrong dimension names!
     * let x = x.rename_view(["a", "b"]);
     * assert_eq!([("a", 2), ("b", 2)], x.shape());
     * ```
     */
    #[track_caller]
    pub fn rename_view(
        self,
        dimensions: [Dimension; D],
    ) -> RecordTensor<'a, T, TensorRename<(T, Index), S, D>, D> {
        RecordTensor::from_existing(
            self.history,
            TensorView::from(TensorRename::from(self.numbers.source(), dimensions)),
        )
    }

    /**
     * Returns a TensorView of this record container, both the `T` for each record element and
     * also the index for that record's entry in the WengertList. These can be parsed back into
     * a RecordTensor with [`from_existing`](RecordTensor::from_existing) or individually into
     * [Record](Record)s with [`Record::from_existing`](Record::from_existing) to continue tracking
     * numerical operations on the data.
     */
    pub fn view(&self) -> TensorView<(T, usize), &RecordTensor<'a, T, S, D>, D> {
        TensorView::from(self)
    }

    /**
     * Returns a TensorAccess which can be indexed in the order of the supplied dimensions
     * to read values from this record container, both the `T` for each record element and
     * also the index for that record's entry in the WengertList. These can be parsed back into
     * a RecordTensor with [`from_existing`](RecordTensor::from_existing) or individually into
     * [Record](Record)s with [`Record::from_existing`](Record::from_existing) to continue tracking
     * numerical operations on the data.
     *
     * # Panics
     *
     * If the set of dimensions supplied do not match the set of dimensions in this tensor's shape.
     */
    #[track_caller]
    pub fn index_by(
        &self,
        dimensions: [Dimension; D],
    ) -> TensorAccess<(T, Index), &RecordTensor<'a, T, S, D>, D> {
        TensorAccess::from(self, dimensions)
    }

    /**
     * Creates a TensorAccess which will index into the dimensions this record was created with
     * in the same order as they were provided, both the `T` for each record element and
     * also the index for that record's entry in the WengertList. These can be parsed back into
     * a RecordTensor with [`from_existing`](RecordTensor::from_existing) or individually into
     * [Record](Record)s with [`Record::from_existing`](Record::from_existing) to continue tracking
     * numerical operations on the data.. See [TensorAccess::from_source_order].
     */
    pub fn index(&self) -> TensorAccess<(T, Index), &RecordTensor<'a, T, S, D>, D> {
        TensorAccess::from_source_order(self)
    }

    /**
     * Returns an iterator over this record tensor as [Record]s instead of the raw `(T, Index)`
     * data. After manipulating the iterator it can be collected back into a RecordTensor with
     * [RecordTensor::from_iter](RecordTensor::from_iter).
     *
     * This is a shorthand for `AsRecords::from(tensor.history(), TensorIterator::from(&tensor))`
     */
    pub fn iter_as_records<'b>(
        &'b self,
    ) -> AsRecords<'a, TensorIterator<'b, (T, Index), RecordTensor<'a, T, S, D>, D>, T> {
        AsRecords::from_tensor(self)
    }
}

impl<'a, T, S, const D: usize> RecordTensor<'a, T, S, D>
where
    T: Numeric + Primitive,
    S: TensorMut<(T, Index), D>,
{
    /**
     * Resets all of the records to place them back on the WengertList, for use
     * in performing another derivation after clearing the WengertList.
     */
    pub fn reset(&mut self) {
        match self.history {
            None => (), // noop
            Some(history) => {
                let total = self.elements();
                let starting_index = history.append_nullary_repeating(total);
                for (x, i) in self
                    .numbers
                    .iter_reference_mut()
                    .zip(calculate_incrementing_indexes(starting_index, total))
                {
                    let (_, ref mut old_index) = x;
                    *old_index = i;
                }
            }
        };
    }

    /**
     * A convenience helper function which takes a RecordContainer by value and
     * calls [reset](RecordTensor::reset()) on it.
     */
    pub fn do_reset(mut x: Self) -> Self {
        x.reset();
        x
    }
}

impl<'a, T> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
{
    /**
     * Creates multiple untracked Records which have no backing WengertList.
     *
     * This is provided for using constants along with Records in operations.
     *
     * For example with `Y = X + 4` the computation graph could be conceived as many
     * `Y[i,j]` nodes with parent nodes of `X[i,j]` and 4 combined with the operation `+`.
     * However there is no need to record the derivatives of a constant, so
     * instead the computation graph can be conceived as `Y[i,j]` nodes each with a single
     * parent node of `X[i,j]` and the unary operation of `+4`.
     */
    pub fn constants<S>(c: S) -> Self
    where
        S: MatrixMut<T> + NoInteriorMutability,
    {
        RecordContainer {
            numbers: MatrixView::from(Matrix::from_flat_row_major(
                (c.view_rows(), c.view_columns()),
                RowMajorOwnedIterator::from_numeric(c)
                    .map(|x| (x, 0))
                    .collect(),
            )),
            history: None,
        }
    }

    /**
     * Creates multiple records backed by the provided WengertList.
     *
     * The records cannot live longer than the WengertList, hence
     * the following example does not compile
     *
     * ```compile_fail
     * use easy_ml::differentiation::RecordMatrix;
     * use easy_ml::differentiation::WengertList;
     * use easy_ml::matrices::Matrix;
     * let record = {
     *     let list = WengertList::new();
     *     RecordMatrix::variables(
     *         &list,
     *         Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
     *     )
     * }; // list no longer in scope
     * ```
     */
    pub fn variables<S>(history: &'a WengertList<T>, x: S) -> Self
    where
        S: MatrixMut<T> + NoInteriorMutability,
    {
        let total = x.view_rows() * x.view_columns();
        let starting_index = history.append_nullary_repeating(total);
        RecordContainer {
            numbers: MatrixView::from(Matrix::from_flat_row_major(
                (x.view_rows(), x.view_columns()),
                RowMajorOwnedIterator::from_numeric(x)
                    .zip(calculate_incrementing_indexes(starting_index, total))
                    .collect(),
            )),
            history: Some(history),
        }
    }
}

impl<'a, T, S> RecordMatrix<'a, T, S>
where
    T: Numeric + Primitive,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    /**
     * Returns the number of elements stored by this container's source.
     *
     * For a 2 x 3 Matrix, this would return 6, and for a 3 x 4 Matrix this would return 12
     * and so on.
     */
    pub fn elements(&self) -> usize {
        self.numbers.rows() * self.numbers.columns()
    }

    /**
     * Returns the dimensionality of this matrix container in Row, Column format
     */
    pub fn size(&self) -> (Row, Column) {
        self.numbers.size()
    }

    /**
     * Creates a container from constants/variables directly, most likely obtained by getting a
     * matrix view of an existing container. **The inputs are not checked for validity**. It is
     * possible to pass in the wrong Wengert list here or even numbers with indexes that aren't
     * tracked on the WengertList.
     *
     * It is recommended to use this constructor only in conjunction with
     * resizing or masking an existing container and not for creating new variables. Any variables
     * created outside of `RecordContainer::variables` would have to be manually added to the
     * correct Wengert list, and any arithmetic operations would also need tracking correctly.
     *
     * ```
     * use easy_ml::differentiation::RecordMatrix;
     * use easy_ml::differentiation::WengertList;
     * use easy_ml::matrices::Matrix;
     * use easy_ml::matrices::views::{MatrixView, MatrixRange};
     *
     * let list = WengertList::new();
     * let x = RecordMatrix::variables(
     *     &list,
     *     Matrix::from_fn((2, 2), |(r, c)| ((r + 3) * (c + 2)) as f64)
     * );
     * // oh no wrong shape!
     * let fixed = MatrixView::from(MatrixRange::from(x, 0..2, 0..1));
     * let x = RecordMatrix::from_existing(Some(&list), fixed);
     * assert_eq!((2, 1), x.size());
     * ```
     */
    pub fn from_existing(
        history: Option<&'a WengertList<T>>,
        numbers: MatrixView<(T, Index), S>,
    ) -> Self {
        RecordContainer { numbers, history }
    }

    /**
     * Returns a MatrixView of this record container, both the `T` for each record element and
     * also the index for that record's entry in the WengertList. These can be parsed back into
     * a RecordMatrix with [`from_existing`](RecordMatrix::from_existing) or individually into
     * [Record](Record)s with [`Record::from_existing`](Record::from_existing) to continue tracking
     * numerical operations on the data.
     */
    pub fn view(&self) -> MatrixView<(T, usize), &RecordMatrix<'a, T, S>> {
        MatrixView::from(self)
    }

    /**
     * Returns an iterator over this record matrix as [Record]s instead of the raw `(T, Index)`
     * data. After manipulating the iterator it can be collected back into a RecordMatrix with
     * [RecordMatrix::from_iter](RecordMatrix::from_iter).
     *
     * This is a shorthand for `AsRecords::from(matrix.history(), RowMajorIterator::from(&matrix))`
     */
    pub fn iter_row_major_as_records<'b>(
        &'b self,
    ) -> AsRecords<'a, RowMajorIterator<'b, (T, Index), RecordMatrix<'a, T, S>>, T> {
        AsRecords::from_matrix_row_major(self)
    }

    /**
     * Returns an iterator over this record matrix as [Record]s instead of the raw `(T, Index)`
     * data. After manipulating the iterator it can be collected back into a RecordMatrix with
     * [RecordMatrix::from_iter](RecordMatrix::from_iter).
     *
     * This is a shorthand for
     * `AsRecords::from(matrix.history(), ColumnMajorIterator::from(&matrix))`
     */
    pub fn iter_column_major_as_records<'b>(
        &'b self,
    ) -> AsRecords<'a, ColumnMajorIterator<'b, (T, Index), RecordMatrix<'a, T, S>>, T> {
        AsRecords::from_matrix_column_major(self)
    }
}

impl<'a, T, S> RecordMatrix<'a, T, S>
where
    T: Numeric + Primitive,
    S: MatrixMut<(T, Index)> + NoInteriorMutability,
{
    /**
     * Resets all of the records to place them back on the WengertList, for use
     * in performing another derivation after clearing the WengertList.
     */
    pub fn reset(&mut self) {
        match self.history {
            None => (), // noop
            Some(history) => {
                let total = self.elements();
                let starting_index = history.append_nullary_repeating(total);
                for (x, i) in self
                    .numbers
                    .row_major_reference_mut_iter()
                    .zip(calculate_incrementing_indexes(starting_index, total))
                {
                    let (_, ref mut old_index) = x;
                    *old_index = i;
                }
            }
        };
    }

    /**
     * A convenience helper function which takes a RecordContainer by value and
     * calls [reset](RecordMatrix::reset()) on it.
     */
    pub fn do_reset(mut x: Self) -> Self {
        x.reset();
        x
    }
}

impl<'a, T, S, const D: usize> RecordContainer<'a, T, S, D>
where
    T: Primitive,
{
    /**
     * Gets the WengertList these records are backed by if variables, and [None](None) if constants.
     */
    pub fn history(&self) -> Option<&'a WengertList<T>> {
        self.history
    }
}

/// Returns the vec of indexes and vec of ys for Y = unary(X), not checking but assuming that the
/// length of the iterator matches the total.
fn unary<'a, T, I>(
    total: usize,
    history: &WengertList<T>,
    records: I,
    fx: impl Fn(T) -> T,
    dfx_dx: impl Fn(T) -> T,
) -> Vec<(T, usize)>
where
    I: Iterator<Item = (T, Index)>,
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    let mut ys = vec![(T::zero(), 0); total];
    history.borrow(|history| {
        // shadow the name so we can't accidentally try to use history while holding
        // the borrow
        // use enumerate not with_index because we need the 1D index for indexing
        // indexes
        for (i, (x, parent)) in records.enumerate() {
            let y = fx(x.clone());
            let derivative = dfx_dx(x);
            let new_index = history.append_unary(parent, derivative);
            ys[i] = (y, new_index)
        }
    }); // drop borrow on history
    ys
}

/// Returns the vec of indexes and vec of zs for Z = binary(X, Y), not checking but assuming that
/// the length of the iterators match the total. Also assumes both inputs have the same shape
fn binary_both_history<'a, T, I1, I2>(
    total: usize,
    history: &WengertList<T>,
    x_records: I1,
    y_records: I2,
    fxy: impl Fn(T, T) -> T,
    dfxy_dx: impl Fn(T, T) -> T,
    dfxy_dy: impl Fn(T, T) -> T,
) -> Vec<(T, usize)>
where
    I1: Iterator<Item = (T, Index)>,
    I2: Iterator<Item = (T, Index)>,
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    let mut zs = vec![(T::zero(), 0); total];
    history.borrow(|history| {
        // shadow the name so we can't accidentally try to use history while holding
        // the borrow
        // use enumerate not with_index because we need the 1D index for indexing
        // indexes
        for (i, ((x, parent1), (y, parent2))) in (x_records.zip(y_records)).enumerate() {
            let z = fxy(x.clone(), y.clone());
            let derivative1 = dfxy_dx(x.clone(), y.clone());
            let derivative2 = dfxy_dy(x, y);
            let new_index = history.append_binary(parent1, derivative1, parent2, derivative2);
            zs[i] = (z, new_index);
        }
    }); // drop borrow on history
    zs
}

/// Returns the vec of indexes and vec of zs for Z = binary(X, Y), as with binary_both_history,
/// but only tracking the derivatives for X, not Y.
fn binary_x_history<'a, T, I1, I2>(
    total: usize,
    history: &WengertList<T>,
    x_records: I1,
    y_records: I2,
    fxy: impl Fn(T, T) -> T,
    dfxy_dx: impl Fn(T, T) -> T,
) -> Vec<(T, usize)>
where
    I1: Iterator<Item = (T, Index)>,
    I2: Iterator<Item = (T, Index)>,
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    let mut zs = vec![(T::zero(), 0); total];
    history.borrow(|history| {
        // shadow the name so we can't accidentally try to use history while holding
        // the borrow
        // use enumerate not with_index because we need the 1D index for indexing
        // indexes
        for (i, ((x, parent1), (y, _))) in (x_records.zip(y_records)).enumerate() {
            let z = fxy(x.clone(), y.clone());
            // if rhs didn't have a history, don't track that derivative
            let derivative1 = dfxy_dx(x, y);
            let new_index = history.append_unary(parent1, derivative1);
            zs[i] = (z, new_index);
        }
    }); // drop borrow on history
    zs
}

/// Returns the vec of indexes and vec of zs for Z = binary(X, Y), as with binary_both_history,
/// but only tracking the derivatives for Y, not X.
fn binary_y_history<'a, T, I1, I2>(
    total: usize,
    history: &WengertList<T>,
    x_records: I1,
    y_records: I2,
    fxy: impl Fn(T, T) -> T,
    dfxy_dy: impl Fn(T, T) -> T,
) -> Vec<(T, usize)>
where
    I1: Iterator<Item = (T, Index)>,
    I2: Iterator<Item = (T, Index)>,
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    let mut zs = vec![(T::zero(), 0); total];
    history.borrow(|history| {
        // shadow the name so we can't accidentally try to use history while holding
        // the borrow
        // use enumerate not with_index because we need the 1D index for indexing
        // indexes
        for (i, ((x, _), (y, parent2))) in (x_records.zip(y_records)).enumerate() {
            let z = fxy(x.clone(), y.clone());
            // if self didn't have a history, don't track that derivative
            let derivative2 = dfxy_dy(x, y);
            let new_index = history.append_unary(parent2, derivative2);
            zs[i] = (z, new_index);
        }
    }); // drop borrow on history
    zs
}

impl<'a, T, S, const D: usize> RecordTensor<'a, T, S, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: TensorRef<(T, Index), D>,
{
    /**
     * Creates a new RecordContainer from a reference to an existing RecordContainer by applying
     * some unary function from `T` to `T` to every element in the container.
     *
     * To compute the new records, the unary function of some input x to some
     * output y is needed along with its derivative with respect to its input x.
     *
     * For example, tanh is a commonly used activation function, but the Real trait
     * does not include this operation and Record has no operations for it specifically.
     * However, you can use this function to compute the tanh for a record container like so:
     *
     * ```
     * use easy_ml::differentiation::{RecordTensor, WengertList};
     * use easy_ml::tensors::Tensor;
     * let list = WengertList::new();
     * let X = RecordTensor::variables(
     *     &list,
     *     Tensor::from_fn(
     *         [("rows", 2), ("columns", 2)],
     *         |[r, c]| 0.15 * ((1 + r + c) as f32)
     *     )
     * );
     * // the derivative of tanh(x) is sech(x) * sech(x) which is equivalent to
     * // 1 / (cosh(x) * cosh(x))
     * let Y = X.unary(|x| x.tanh(), |x| 1.0 / (x.cosh() * x.cosh()));
     *
     * // we can unwrap here because we know Y contains variables not constants
     * let derivatives = Y.derivatives().unwrap();
     * let derivatives_indexing = derivatives.index_by(["rows", "columns"]);
     * assert_eq!(
     *     derivatives_indexing.get_ref([0, 0]).at_tensor(&X),
     *     Tensor::from(
     *         [("rows", 2), ("columns", 2)],
     *         // [0, 0] element in Y only had the one input variable [0, 0] in X
     *         vec![
     *             0.9778332, 0.0,
     *             0.0,       0.0
     *        ]
     *     ),
     * );
     * assert_eq!(
     *     derivatives_indexing.get_ref([0, 1]).at_tensor(&X),
     *     Tensor::from(
     *         [("rows", 2), ("columns", 2)],
     *         vec![
     *             0.0, 0.915137,
     *             0.0, 0.0
     *        ]
     *     ),
     * );
     * assert_eq!(
     *     // [0, 1] and [1, 0] elements in X had the same starting value so end up with the same
     *     // derivative for their corresponding input variable in X
     *     derivatives_indexing.get_ref([0, 1]).at_tensor(&X).index().get([0, 1]),
     *     derivatives_indexing.get_ref([1, 0]).at_tensor(&X).index().get([1, 0]),
     * );
     * assert_eq!(
     *     derivatives_indexing.get_ref([1, 1]).at_tensor(&X),
     *     Tensor::from(
     *         [("rows", 2), ("columns", 2)],
     *         vec![
     *             0.0, 0.0,
     *             0.0, 0.8220013
     *        ]
     *     ),
     * );
     * ```
     */
    #[track_caller]
    pub fn unary(
        &self,
        fx: impl Fn(T) -> T,
        dfx_dx: impl Fn(T) -> T,
    ) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D> {
        let total = self.elements();
        match self.history {
            None => RecordTensor::constants(self.numbers.map(|(x, _)| fx(x))),
            Some(history) => {
                let ys = unary::<T, _>(total, history, self.numbers.iter(), fx, dfx_dx);
                RecordContainer {
                    numbers: self.numbers.new_with_same_shape(ys),
                    history: Some(history),
                }
            }
        }
    }

    /**
     * Creates a new RecordContainer from two RecordContainers by applying
     * some binary function from `T` to `T` to every element pair in the containers. Both
     * containers must have the same shape.
     *
     * To compute the new records, the binary function of some inputs x and y to some
     * output z is needed along with its derivative with respect to its first input x and
     * its derivative with respect to its second input y.
     *
     * For example, atan2 takes two arguments, but the Real trait
     * does not include this operation and Record has no operations for it specifically.
     * However, you can use this function to compute the atan2 for two record containers like so:
     *
     * ```
     * use easy_ml::differentiation::{RecordTensor, WengertList};
     * use easy_ml::tensors::Tensor;
     * let list = WengertList::new();
     * let X = RecordTensor::variables(
     *     &list,
     *     Tensor::from_fn(
     *         [("rows", 2), ("columns", 2)],
     *         |[r, c]| ((1 + r + c) as f32)
     *     )
     * );
     * let Y = RecordTensor::variables(
     *     &list,
     *     Tensor::from_fn(
     *         [("rows", 2), ("columns", 2)],
     *         |[r, c]| ((1 + r + c) as f32)
     *     )
     * );
     * // the derivative of atan2 with respect to x is y/(x*x + y*y)
     * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdx
     * // the derivative of atan2 with respect to y is -x/(x*x + y*y)
     * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdy
     * let Z = X.binary(&Y,
     *     |x, y| x.atan2(y),
     *     |x, y| y/((x*x) + (y*y)),
     *     |x, y| -x/((x*x) + (y*y))
     * );
     *
     *
     * // we can unwrap here because we know Z contains variables not constants
     * let derivatives = Z.derivatives().unwrap();
     * // Just as in the unary example, only one pair of the four inputs in X and Y influence the
     * // outputs in Z, so we have a lot of 0.0 derivatives, and the inputs in [0, 1] and [1, 0]
     * // are identical so we see the same derivative.
     * let dZ_dX = derivatives.map(|d| d.at_tensor(&X));
     * assert_eq!(
     *     dZ_dX,
     *     Tensor::from([("rows", 2), ("columns", 2)], vec![
     *         Tensor::from([("rows", 2), ("columns", 2)], vec![
     *             0.5, 0.0,
     *             0.0, 0.0
     *         ]),
     *         Tensor::from([("rows", 2), ("columns", 2)], vec![
     *             0.0, 0.25,
     *             0.0, 0.0
     *         ]),
     *         Tensor::from([("rows", 2), ("columns", 2)], vec![
     *             0.0, 0.0,
     *             0.25, 0.0
     *         ]),
     *         Tensor::from([("rows", 2), ("columns", 2)], vec![
     *             0.0, 0.0,
     *             0.0, 0.16666667
     *         ])
     *     ])
     * );
     * let dZ_dY = derivatives.map(|d| d.at_tensor(&Y));
     * assert_eq!(
     *     dZ_dY,
     *     Tensor::from([("rows", 2), ("columns", 2)], vec![
     *         Tensor::from([("rows", 2), ("columns", 2)], vec![
     *             -0.5, 0.0,
     *             0.0, 0.0
     *         ]),
     *         Tensor::from([("rows", 2), ("columns", 2)], vec![
     *             0.0, -0.25,
     *             0.0, 0.0
     *         ]),
     *         Tensor::from([("rows", 2), ("columns", 2)], vec![
     *             0.0, 0.0,
     *             -0.25, 0.0
     *         ]),
     *         Tensor::from([("rows", 2), ("columns", 2)], vec![
     *             0.0, 0.0,
     *             0.0, -0.16666667
     *         ])
     *     ])
     * );
     * ```
     *
     * # Panics
     *
     * - If both record containers have a WengertList that are different to each other
     * - If the record containers have different shapes
     */
    #[track_caller]
    pub fn binary<S2>(
        &self,
        rhs: &RecordTensor<'a, T, S2, D>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
    where
        S2: TensorRef<(T, Index), D>,
    {
        {
            let left_shape = self.numbers.shape();
            let right_shape = rhs.numbers.shape();
            if left_shape != right_shape {
                panic!(
                    "Record containers must have the same shape for a binary operation: (left: {:?}, right: {:?})",
                    left_shape,
                    right_shape
                );
            }
        }
        let total = self.elements();
        match (self.history, rhs.history) {
            (None, None) => RecordTensor::constants(
                // use direct_from here maybe?
                Tensor::from(
                    self.numbers.shape(),
                    self.numbers
                        .iter()
                        .zip(rhs.numbers.iter())
                        .map(|((x, _), (y, _))| fxy(x, y))
                        .collect(),
                ),
            ),
            (Some(history), None) => {
                let zs = binary_x_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.iter(),
                    rhs.numbers.iter(),
                    fxy,
                    dfxy_dx,
                );
                RecordContainer {
                    numbers: self.numbers.new_with_same_shape(zs),
                    history: Some(history),
                }
            }
            (None, Some(history)) => {
                let zs = binary_y_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.iter(),
                    rhs.numbers.iter(),
                    fxy,
                    dfxy_dy,
                );
                RecordContainer {
                    numbers: self.numbers.new_with_same_shape(zs),
                    history: Some(history),
                }
            }
            (Some(history), Some(h)) => {
                assert!(
                    record_operations::same_lists(history, h),
                    "Record containers must be using the same WengertList"
                );
                let zs = binary_both_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.iter(),
                    rhs.numbers.iter(),
                    fxy,
                    dfxy_dx,
                    dfxy_dy,
                );
                RecordContainer {
                    numbers: self.numbers.new_with_same_shape(zs),
                    history: Some(history),
                }
            }
        }
    }

    /**
     * Creates a new RecordContainer from a reference to an existing RecordContainer by applying
     * some unary function on `Record<T>` to `Record<T>` to every element in the container. This
     * will fail if the function would create records with inconsistent histories.
     *
     * When used with pure functions that can't return different histories for different inputs
     * unwrapping with always succeed.
     *
     * This API can allow you to call a generic function that operates on
     * [Numeric](crate::numeric::Numeric) numbers and apply all the correct derivative tracking
     * during the intermediate calculations for you, without having to resort to storing the
     * Record types.
     *
     * ```
     * use easy_ml::numeric::Numeric;
     * use easy_ml::numeric::extra::Real;
     * use easy_ml::tensors::Tensor;
     * use easy_ml::differentiation::{RecordTensor, WengertList};
     *
     * fn sigmoid<T: Numeric + Real+ Copy>(x: T) -> T {
     *     T::one() / (T::one() + (-x).exp())
     * }
     *
     * let history = WengertList::new();
     * let layer = RecordTensor::variables(&history, Tensor::from([("x", 2)], vec![ 0.2, 0.6 ]));
     * let after = layer.map(sigmoid).unwrap(); // sigmoid can't introduce inconsistent histories
     * ```
     *
     * NB: Mapping a RecordTensor of constants to variables is not inconsistent, the history
     * after mapping doesn't have to be the same as before, only must be the same for every
     * mapped element.
     */
    #[track_caller]
    pub fn map(
        &self,
        fx: impl Fn(Record<'a, T>) -> Record<'a, T>,
    ) -> Result<RecordTensor<'a, T, Tensor<(T, Index), D>, D>, InconsistentHistory<'a, T>> {
        let result = RecordTensor::from_iter(self.shape(), self.iter_as_records().map(fx));
        RecordTensor::<'a, T, S, D>::map_collection(result, self.shape())
    }

    /**
     * Creates a new RecordContainer from a reference to an existing RecordContainer by applying
     * some unary function on `Record<T>` and each index of that position in the Record to
     * `Record<T>` to every element in the container. This will fail if the function would create
     * records with inconsistent histories.
     *
     * When used with pure functions that can't return different histories for different inputs
     * unwrapping with always succeed.
     *
     * This API can allow you to call a generic function that operates on
     * [Numeric](crate::numeric::Numeric) numbers and apply all the correct derivative tracking
     * during the intermediate calculations for you, without having to resort to storing the
     * Record types.
     *
     * NB: Mapping a RecordTensor of constants to variables is not inconsistent, the history
     * after mapping doesn't have to be the same as before, only must be the same for every
     * mapped element.
     */
    #[track_caller]
    pub fn map_with_index(
        &self,
        fx: impl Fn([usize; D], Record<'a, T>) -> Record<'a, T>,
    ) -> Result<RecordTensor<'a, T, Tensor<(T, Index), D>, D>, InconsistentHistory<'a, T>> {
        let result = RecordTensor::from_iter(
            self.shape(),
            self.iter_as_records().with_index().map(|(i, x)| fx(i, x)),
        );
        RecordTensor::<'a, T, S, D>::map_collection(result, self.shape())
    }

    #[track_caller]
    fn map_collection(
        result: Result<
            RecordTensor<'a, T, Tensor<(T, usize), D>, D>,
            InvalidRecordIteratorError<'a, T, D>,
        >,
        shape: [(Dimension, usize); D],
    ) -> Result<RecordTensor<'a, T, Tensor<(T, usize), D>, D>, InconsistentHistory<'a, T>> {
        use InvalidRecordIteratorError as Error;
        match result {
            Ok(tensor) => Ok(tensor),
            Err(error) => match error {
                // These first two should be 100% impossible but provide a sensible error just
                // in case some weird things break our invariants
                Error::Empty => panic!(
                    "Illegal state, record tensor was empty {:?}",
                    shape
                ),
                Error::Shape { requested, length } => panic!(
                    "Illegal state, record tensor shape was inconsistent: requested: {:?}, length of data: {:?}",
                    requested,
                    length
                ),
                // This one is theoretically possible but in practise shouldn't happen by accident
                // However, it can't implement Debug unless T is debug so to avoid having to
                // restrict our function signature we return a Result anyway - this also encourages
                // the user to make sure their function isn't going to cause this case, which
                // with some of the other variants like with_index might come up more easily
                Error::InconsistentHistory(h) => Err(h),
            }
        }
    }

    /**
     * For each record in the container, peforms a backward pass up its WengertList from it
     * as the output, computing all the derivatives for the inputs involving this output.
     *
     * If this container has no backing WengertList, ie was created as constants, then None is
     * returned instead. Otherwise the returned Tensor will have the same shape as this container,
     * with the respective derivatives matching each element in this container.
     *
     * If you have N inputs x<sub>1</sub> to x<sub>N</sub>, and this output is Y with M outputs,
     * then this computes all the derivatives δy<sub>j</sub>/δx<sub>i</sub> for i = 1 to N and
     * j = 1 to M.
     *
     * If you have a lot of outputs this could be very expensive! Reverse auto diff is optimised
     * for domains where there are many more inputs than outputs.
     *
     * If you only need some of the derivatives then
     * [derivatives_for](RecordTensor::derivatives_for) can be used instead to avoid
     * calculating the rest.
     */
    pub fn derivatives(&self) -> Option<Tensor<Derivatives<T>, D>> {
        self.history.map(|history| {
            self.numbers.map(|(x, i)| {
                Record {
                    number: x,
                    history: Some(history),
                    index: i,
                }
                .derivatives()
            })
        })
    }

    /**
     * For the record at the index, peforms a backward pass up its WengertList from it
     * as the output, computing all the derivatives for the inputs involving this output.
     *
     * If the index is invalid or this container has no backing WengertList, ie was created
     * as constants, then None is returned instead.
     *
     * If you have N inputs x<sub>1</sub> to x<sub>N</sub>, and this output is y,
     * then this computes all the derivatives δy/δx<sub>i</sub> for i = 1 to N.
     */
    pub fn derivatives_for(&self, indexes: [usize; D]) -> Option<Derivatives<T>> {
        let (number, index) = match self.get_reference(indexes).map(|(x, i)| (x.clone(), *i)) {
            Some(tuple) => tuple,
            None => return None,
        };
        // The nature of reverse autodiff is that we expect to only have a few outputs from
        // which we calculate all the derivatives we care about. Therefore just call Record and
        // reuse the implementation instead of trying to do anything clever like calculate all
        // derivatives for every number in this container.
        Record {
            number,
            history: self.history,
            index,
        }
        .try_derivatives()
    }

    /**
     * Performs elementwise multiplication for two record tensors of the same shape.
     *
     * # Panics
     *
     * - If both record containers have a WengertList that are different to each other
     * - If the record containers have different shapes
     */
    // TODO: Assign variants?
    pub fn elementwise_multiply<S2>(
        &self,
        other: &RecordTensor<'a, T, S2, D>,
    ) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
    where
        S2: TensorRef<(T, Index), D>,
    {
        self.binary(
            other,
            Multiplication::<T>::function,
            Multiplication::<T>::d_function_dx,
            Multiplication::<T>::d_function_dy,
        )
    }

    /**
     * Performs elementwise division for two record tensors of the same shape.
     *
     * # Panics
     *
     * - If both record containers have a WengertList that are different to each other
     * - If the record containers have different shapes
     */
    pub fn elementwise_divide<S2>(
        &self,
        other: &RecordTensor<'a, T, S2, D>,
    ) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
    where
        S2: TensorRef<(T, Index), D>,
    {
        self.binary(
            other,
            Division::<T>::function,
            Division::<T>::d_function_dx,
            Division::<T>::d_function_dy,
        )
    }
}

impl<T: Clone + Primitive> Derivatives<T> {
    /**
     * Queries the derivative at the provided index into the record tensor as input.
     *
     * If you construct a Derivatives object for some output y,
     * and call .at_tensor_index(i, &xs) on it for some input container xs and index i, this
     * returns dy/dx where x = xs\[i\].
     *
     * If the index into the tensor is invalid, returns None instead.
     */
    pub fn at_tensor_index<S, const D: usize>(
        &self,
        indexes: [usize; D],
        input: &RecordTensor<T, S, D>,
    ) -> Option<T>
    where
        S: TensorRef<(T, Index), D>,
    {
        let index = match input.get_reference(indexes).map(|(_, i)| *i) {
            Some(i) => i,
            None => return None,
        };
        Some(self.derivatives[index].clone())
    }

    /**
     * Queries the derivatives at every element in the record tensor input.
     *
     * If you construct a Derivatives object for some output y,
     * and call .at_tensor(&xs) on it for some input container xs this
     * returns dy/dx for every x in xs.
     */
    pub fn at_tensor<S, const D: usize>(&self, input: &RecordTensor<T, S, D>) -> Tensor<T, D>
    where
        S: TensorRef<(T, Index), D>,
    {
        input.numbers.map(|(_, i)| self.derivatives[i].clone())
    }

    /**
     * Queries the derivative at the provided index into the record matrix as input.
     *
     * If you construct a Derivatives object for some output y,
     * and call .at_matrix_index(i, j, &xs) on it for some input container xs and indexes i and j,
     * this returns dy/dx where x = xs\[i, j\].
     *
     * If the index into the tensor is invalid, returns None instead.
     */
    pub fn at_matrix_index<S>(
        &self,
        row: Row,
        column: Column,
        input: &RecordMatrix<T, S>,
    ) -> Option<T>
    where
        S: MatrixRef<(T, Index)> + NoInteriorMutability,
    {
        let index = match input.try_get_reference(row, column).map(|(_, i)| *i) {
            Some(i) => i,
            None => return None,
        };
        Some(self.derivatives[index].clone())
    }

    /**
     * Queries the derivatives at every element in the record matrix input.
     *
     * If you construct a Derivatives object for some output y,
     * and call .at_matrix(&xs) on it for some input container xs this
     * returns dy/dx for every x in xs.
     */
    pub fn at_matrix<S>(&self, input: &RecordMatrix<T, S>) -> Matrix<T>
    where
        S: MatrixRef<(T, Index)> + NoInteriorMutability,
    {
        input.numbers.map(|(_, i)| self.derivatives[i].clone())
    }
}

impl<'a, T, S, const D: usize> RecordTensor<'a, T, S, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: TensorMut<(T, Index), D>,
{
    /**
     * Overwrites a RecordContainer by applying
     * some unary function from `T` to `T` to every element in the container.
     *
     * To compute the new records, the unary function of some input x to some
     * output y is needed along with its derivative with respect to its input x.
     */
    #[track_caller]
    pub fn unary_assign(&mut self, fx: impl Fn(T) -> T, dfx_dx: impl Fn(T) -> T) {
        let total = self.elements();
        match self.history {
            None => self.numbers.map_mut(|(x, i)| (fx(x), i)),
            Some(history) => {
                let ys = unary::<T, _>(total, history, self.numbers.iter(), fx, dfx_dx);
                for (element, result) in self.numbers.iter_reference_mut().zip(ys) {
                    *element = result;
                }
                self.history = Some(history);
            }
        }
    }

    /**
     * Overwrites the left hand side of a RecordContainer with the result of applying
     * some binary function from `T` to `T` to every element pair in the containers. Both
     * containers must have the same shape.
     * To compute the new records, the binary function of some inputs x and y to some
     * output z is needed along with its derivative with respect to its first input x and
     * its derivative with respect to its second input y.
     *
     * # Panics
     *
     * - If both record containers have a WengertList that are different to each other
     * - If the record containers have different shapes
     */
    #[track_caller]
    pub fn binary_left_assign<S2>(
        &mut self,
        rhs: &RecordTensor<'a, T, S2, D>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) where
        S2: TensorRef<(T, Index), D>,
    {
        {
            let left_shape = self.numbers.shape();
            let right_shape = rhs.numbers.shape();
            if left_shape != right_shape {
                panic!(
                    "Record containers must have the same shape for a binary operation: (left: {:?}, right: {:?})",
                    left_shape,
                    right_shape
                );
            }
        }
        let total = self.elements();
        match (self.history, rhs.history) {
            (None, None) => {
                for (x, y) in self.numbers.iter_reference_mut().zip(rhs.numbers.iter()) {
                    let (left, _) = x;
                    let (right, _) = y;
                    *x = (fxy(left.clone(), right), 0);
                }
            }
            (Some(history), None) => {
                let zs = binary_x_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.iter(),
                    rhs.numbers.iter(),
                    fxy,
                    dfxy_dx,
                );
                for (element, result) in self.numbers.iter_reference_mut().zip(zs) {
                    *element = result;
                }
                self.history = Some(history);
            }
            (None, Some(history)) => {
                let zs = binary_y_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.iter(),
                    rhs.numbers.iter(),
                    fxy,
                    dfxy_dy,
                );
                for (element, result) in self.numbers.iter_reference_mut().zip(zs) {
                    *element = result;
                }
                self.history = Some(history);
            }
            (Some(history), Some(h)) => {
                assert!(
                    record_operations::same_lists(history, h),
                    "Record containers must be using the same WengertList"
                );
                let zs = binary_both_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.iter(),
                    rhs.numbers.iter(),
                    fxy,
                    dfxy_dx,
                    dfxy_dy,
                );
                for (element, result) in self.numbers.iter_reference_mut().zip(zs) {
                    *element = result;
                }
                self.history = Some(history);
            }
        }
    }

    /**
     * A convenience helper function which takes the RecordContainer value and
     * calls [unary_assign](RecordTensor::unary_assign()) on it, returning
     * the record container which now contains the result of the operation.
     */
    #[track_caller]
    pub fn do_unary_assign(mut self, fx: impl Fn(T) -> T, dfx_dx: impl Fn(T) -> T) -> Self {
        self.unary_assign(fx, dfx_dx);
        self
    }

    /**
     * A convenience helper function which takes the left hand side by value and
     * calls [binary_left_assign](RecordTensor::binary_left_assign()) on it, returning
     * the left hand side which now contains the result of the operation.
     */
    #[track_caller]
    pub fn do_binary_left_assign<S2>(
        mut self,
        rhs: &RecordTensor<'a, T, S2, D>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) -> Self
    where
        S2: TensorRef<(T, Index), D>,
    {
        self.binary_left_assign(rhs, fxy, dfxy_dx, dfxy_dy);
        self
    }

    /**
     * Updates this RecordContainer in place by applying some unary function on `Record<T>` to
     * `Record<T>` to every element in the container. This will fail if the function would create
     * records with inconsistent histories.
     *
     * When used with pure functions that can't return different histories for different inputs
     * unwrapping with always succeed.
     *
     * Since this updates the container in place, if Err is returned then the data in this
     * RecordContainer is still available but it has been corrupted - at least one of the elements
     * should have a different history than what it will have because the mapping function created
     * inconsistent histories that couldn't be represented by the container as it only stores
     * one.
     *
     * This API can allow you to call a generic function that operates on
     * [Numeric](crate::numeric::Numeric) numbers and apply all the correct derivative tracking
     * during the intermediate calculations for you, without having to resort to storing the
     * Record types.
     *
     * NB: Mapping a RecordTensor of constants to variables is not inconsistent, the history
     * after mapping doesn't have to be the same as before, only must be the same for every
     * mapped element.
     */
    // TODO: Example
    #[track_caller]
    pub fn map_mut(
        &mut self,
        fx: impl Fn(Record<'a, T>) -> Record<'a, T>,
    ) -> Result<(), InconsistentHistory<'a, T>> {
        let history = self.history;
        let new_history =
            map_mut_base::<'a, T, _, _>(TensorReferenceMutIterator::from(self), |x| {
                let record = Record::from_existing(x.clone(), history);
                let result = fx(record);
                *x = (result.number, result.index);
                result.history
            })?;
        self.history = new_history;
        Ok(())
    }

    /**
     * Updates this RecordContainer in place by applying some unary function on `Record<T>` and
     * each index of that position in the Record to `Record<T>` to every element in the container.
     * This will fail if the function would create records with inconsistent histories.
     *
     * When used with pure functions that can't return different histories for different inputs
     * unwrapping with always succeed.
     *
     * Since this updates the container in place, if Err is returned then the data in this
     * RecordContainer is still available but it has been corrupted - at least one of the elements
     * should have a different history than what it will have because the mapping function created
     * inconsistent histories that couldn't be represented by the container as it only stores
     * one.
     *
     * This API can allow you to call a generic function that operates on
     * [Numeric](crate::numeric::Numeric) numbers and apply all the correct derivative tracking
     * during the intermediate calculations for you, without having to resort to storing the
     * Record types.
     *
     * NB: Mapping a RecordTensor of constants to variables is not inconsistent, the history
     * after mapping doesn't have to be the same as before, only must be the same for every
     * mapped element.
     */
    #[track_caller]
    pub fn map_mut_with_index(
        &mut self,
        fx: impl Fn([usize; D], Record<'a, T>) -> Record<'a, T>,
    ) -> Result<(), InconsistentHistory<'a, T>> {
        let history = self.history;
        let new_history = map_mut_base::<'a, T, _, _>(
            TensorReferenceMutIterator::from(self).with_index(),
            |(i, x)| {
                let record = Record::from_existing(x.clone(), history);
                let result = fx(i, record);
                *x = (result.number, result.index);
                result.history
            },
        )?;
        self.history = new_history;
        Ok(())
    }
}

#[track_caller]
fn map_mut_base<'a, T, I, X>(
    mut iter: I,
    fx: impl Fn(X) -> Option<&'a WengertList<T>>,
) -> Result<Option<&'a WengertList<T>>, InconsistentHistory<'a, T>>
where
    I: Iterator<Item = X>,
    T: Primitive,
{
    use crate::differentiation::record_operations::are_exact_same_list;
    #[rustfmt::skip]
    let first_history = fx(iter.next().expect("Illegal state, record container was empty"));
    let mut different_history: Option<Option<&WengertList<T>>> = None;
    for x in iter {
        let history = fx(x);
        if !are_exact_same_list(history, first_history) {
            different_history = Some(history);
        }
    }
    match different_history {
        None => Ok(first_history),
        Some(h) => Err(InconsistentHistory {
            first: first_history,
            later: h,
        }),
    }
}

impl<'a, T, S, const D: usize> RecordTensor<'a, T, S, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: TensorRef<(T, Index), D>,
{
    /**
     * Overwrites the right hand side of a RecordContainer with the result of applying
     * some binary function from `T` to `T` to every element pair in the containers. Both
     * containers must have the same shape.
     * To compute the new records, the binary function of some inputs x and y to some
     * output z is needed along with its derivative with respect to its first input x and
     * its derivative with respect to its second input y.
     *
     * # Panics
     *
     * - If both record containers have a WengertList that are different to each other
     * - If the record containers have different shapes
     */
    #[track_caller]
    pub fn binary_right_assign<S2>(
        &self,
        rhs: &mut RecordTensor<'a, T, S2, D>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) where
        S2: TensorMut<(T, Index), D>,
    {
        // x is lhs, y is rhs, so calling binary_left_assign on the rhs container
        // means we need to swap all the arguments
        rhs.binary_left_assign(
            self,
            |y, x| fxy(x, y),
            |y, x| dfxy_dy(x, y),
            |y, x| dfxy_dx(x, y),
        )
    }

    /**
     * A convenience helper function which takes the right hand side by value and
     * calls [binary_right_assign](RecordTensor::binary_right_assign()) on it, returning
     * the right hand side which now contains the result of the operation.
     */
    #[track_caller]
    pub fn do_binary_right_assign<S2>(
        &self,
        mut rhs: RecordTensor<'a, T, S2, D>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) -> RecordTensor<'a, T, S2, D>
    where
        S2: TensorMut<(T, Index), D>,
    {
        self.binary_right_assign(&mut rhs, fxy, dfxy_dx, dfxy_dy);
        rhs
    }
}

impl<'a, T, S> RecordMatrix<'a, T, S>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    /**
     * Creates a new RecordContainer from a reference to an existing RecordContainer by applying
     * some unary function from `T` to `T` to every element in the container.
     *
     * To compute the new records, the unary function of some input x to some
     * output y is needed along with its derivative with respect to its input x.
     *
     * For example, tanh is a commonly used activation function, but the Real trait
     * does not include this operation and Record has no operations for it specifically.
     * However, you can use this function to compute the tanh for a record container like so:
     *
     * ```
     * use easy_ml::differentiation::{RecordMatrix, WengertList};
     * use easy_ml::matrices::Matrix;
     * let list = WengertList::new();
     * let X = RecordMatrix::variables(
     *     &list,
     *     Matrix::from_fn((2, 2), |(r, c)| 0.15 * ((1 + r + c) as f32))
     * );
     * // the derivative of tanh(x) is sech(x) * sech(x) which is equivalent to
     * // 1 / (cosh(x) * cosh(x))
     * let Y = X.unary(|x| x.tanh(), |x| 1.0 / (x.cosh() * x.cosh()));
     *
     * // we can unwrap here because we know Y contains variables not constants
     * let derivatives = Y.derivatives().unwrap();
     * assert_eq!(
     *     derivatives.get_reference(0, 0).at_matrix(&X),
     *     Matrix::from(vec![
     *         // (0, 0) element in Y only had the one input variable (0, 0) in X
     *         vec![0.9778332, 0.0],
     *         vec![0.0,       0.0]
     *     ]),
     * );
     * assert_eq!(
     *     derivatives.get_reference(0, 1).at_matrix(&X),
     *     Matrix::from(vec![
     *         vec![0.0, 0.915137],
     *         vec![0.0,      0.0]
     *     ]),
     * );
     * assert_eq!(
     *     // (0, 1) and (1, 0) elements in X had the same starting value so end up with the same
     *     // derivative for their corresponding input variable in X
     *     derivatives.get_reference(0, 1).at_matrix(&X).get(0, 1),
     *     derivatives.get_reference(1, 0).at_matrix(&X).get(1, 0),
     * );
     * assert_eq!(
     *     derivatives.get_reference(1, 1).at_matrix(&X),
     *     Matrix::from(vec![
     *         vec![0.0, 0.0      ],
     *         vec![0.0, 0.8220013]
     *     ]),
     * );
     * ```
     */
    #[track_caller]
    pub fn unary(
        &self,
        fx: impl Fn(T) -> T,
        dfx_dx: impl Fn(T) -> T,
    ) -> RecordMatrix<'a, T, Matrix<(T, Index)>> {
        let total = self.elements();
        match self.history {
            None => RecordMatrix::constants(self.numbers.map(|(x, _)| fx(x))),
            Some(history) => {
                let ys = unary::<T, _>(total, history, self.numbers.row_major_iter(), fx, dfx_dx);
                RecordContainer {
                    numbers: MatrixView::from(Matrix::from_flat_row_major(self.numbers.size(), ys)),
                    history: Some(history),
                }
            }
        }
    }

    /**
     * Creates a new RecordContainer from two RecordContainers by applying
     * some binary function from `T` to `T` to every element pair in the containers. Both
     * containers must have the same shape.
     *
     * To compute the new records, the binary function of some inputs x and y to some
     * output z is needed along with its derivative with respect to its first input x and
     * its derivative with respect to its second input y.
     *
     * For example, atan2 takes two arguments, but the Real trait
     * does not include this operation and Record has no operations for it specifically.
     * However, you can use this function to compute the atan2 for two record containers like so:
     *
     * ```
     * use easy_ml::differentiation::{RecordMatrix, WengertList};
     * use easy_ml::matrices::Matrix;
     * let list = WengertList::new();
     * let X = RecordMatrix::variables(
     *     &list,
     *     Matrix::from_fn((2, 2), |(r, c)| ((1 + r + c) as f32))
     * );
     * let Y = RecordMatrix::variables(
     *     &list,
     *     Matrix::from_fn((2, 2), |(r, c)| ((1 + r + c) as f32))
     * );
     * // the derivative of atan2 with respect to x is y/(x*x + y*y)
     * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdx
     * // the derivative of atan2 with respect to y is -x/(x*x + y*y)
     * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdy
     * let Z = X.binary(&Y,
     *     |x, y| x.atan2(y),
     *     |x, y| y/((x*x) + (y*y)),
     *     |x, y| -x/((x*x) + (y*y))
     * );
     *
     * // we can unwrap here because we know Z contains variables not constants
     * let derivatives = Z.derivatives().unwrap();
     * // Just as in the unary example, only one pair of the four inputs in X and Y influence the
     * // outputs in Z, so we have a lot of 0.0 derivatives, and the inputs in [0, 1] and [1, 0]
     * // are identical so we see the same derivative.
     * let dZ_dX = derivatives.map(|d| d.at_matrix(&X));
     * assert_eq!(
     *     dZ_dX,
     *     Matrix::from(vec![
     *          vec![
     *              Matrix::from(vec![
     *                  vec![ 0.5, 0.0 ],
     *                  vec![ 0.0, 0.0 ]
     *              ]),
     *              Matrix::from(vec![
     *                  vec![ 0.0, 0.25 ],
     *                  vec![ 0.0, 0.0 ]
     *              ])
     *          ],
     *          vec![
     *              Matrix::from(vec![
     *                  vec![ 0.0, 0.0 ],
     *                  vec![ 0.25, 0.0 ]
     *              ]),
     *              Matrix::from(vec![
     *                  vec![ 0.0, 0.0 ],
     *                  vec![ 0.0, 0.16666667 ]
     *              ])
     *          ]
     *     ])
     * );
     * let dZ_dY = derivatives.map(|d| d.at_matrix(&Y));
     * assert_eq!(
     *     dZ_dY,
     *     Matrix::from(vec![
     *          vec![
     *              Matrix::from(vec![
     *                  vec![ -0.5, 0.0 ],
     *                  vec![ 0.0, 0.0 ]
     *              ]),
     *              Matrix::from(vec![
     *                  vec![ 0.0, -0.25 ],
     *                  vec![ 0.0, 0.0 ]
     *              ])
     *          ],
     *          vec![
     *              Matrix::from(vec![
     *                  vec![ 0.0, 0.0 ],
     *                  vec![ -0.25, 0.0 ]
     *              ]),
     *              Matrix::from(vec![
     *                  vec![ 0.0, 0.0 ],
     *                  vec![ 0.0, -0.16666667 ]
     *              ])
     *          ]
     *     ])
     * );
     * ```
     *
     * # Panics
     *
     * - If both record containers have a WengertList that are different to each other
     * - If the record containers have different shapes
     */
    #[track_caller]
    pub fn binary<S2>(
        &self,
        rhs: &RecordMatrix<'a, T, S2>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
    where
        S2: MatrixRef<(T, Index)> + NoInteriorMutability,
    {
        let shape = {
            let left_shape = self.numbers.size();
            let right_shape = rhs.numbers.size();
            if left_shape != right_shape {
                panic!(
                    "Record containers must have the same size for a binary operation: (left: {:?}, right: {:?})",
                    left_shape,
                    right_shape
                );
            }
            left_shape
        };
        let total = self.elements();
        match (self.history, rhs.history) {
            (None, None) => RecordMatrix::constants(Matrix::from_flat_row_major(
                shape,
                self.numbers
                    .row_major_iter()
                    .zip(rhs.numbers.row_major_iter())
                    .map(|((x, _), (y, _))| fxy(x, y))
                    .collect(),
            )),
            (Some(history), None) => {
                let zs = binary_x_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.row_major_iter(),
                    rhs.numbers.row_major_iter(),
                    fxy,
                    dfxy_dx,
                );
                RecordContainer {
                    numbers: MatrixView::from(Matrix::from_flat_row_major(shape, zs)),
                    history: Some(history),
                }
            }
            (None, Some(history)) => {
                let zs = binary_y_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.row_major_iter(),
                    rhs.numbers.row_major_iter(),
                    fxy,
                    dfxy_dy,
                );
                RecordContainer {
                    numbers: MatrixView::from(Matrix::from_flat_row_major(shape, zs)),
                    history: Some(history),
                }
            }
            (Some(history), Some(h)) => {
                assert!(
                    record_operations::same_lists(history, h),
                    "Record containers must be using the same WengertList"
                );
                let zs = binary_both_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.row_major_iter(),
                    rhs.numbers.row_major_iter(),
                    fxy,
                    dfxy_dx,
                    dfxy_dy,
                );
                RecordContainer {
                    numbers: MatrixView::from(Matrix::from_flat_row_major(shape, zs)),
                    history: Some(history),
                }
            }
        }
    }

    /**
     * Creates a new RecordContainer from a reference to an existing RecordContainer by applying
     * some unary function on `Record<T>` to `Record<T>` to every element in the container. This
     * will fail if the function would create records with inconsistent histories.
     *
     * When used with pure functions that can't return different histories for different inputs
     * unwrapping with always succeed.
     *
     * This API can allow you to call a generic function that operates on
     * [Numeric](crate::numeric::Numeric) numbers and apply all the correct derivative tracking
     * during the intermediate calculations for you, without having to resort to storing the
     * Record types.
     *
     * ```
     * use easy_ml::numeric::Numeric;
     * use easy_ml::numeric::extra::Real;
     * use easy_ml::matrices::Matrix;
     * use easy_ml::differentiation::{RecordMatrix, WengertList};
     *
     * fn sigmoid<T: Numeric + Real+ Copy>(x: T) -> T {
     *     T::one() / (T::one() + (-x).exp())
     * }
     *
     * let history = WengertList::new();
     * let layer = RecordMatrix::variables(&history, Matrix::from(vec![vec![ 0.2, 0.6 ]]));
     * let after = layer.map(sigmoid).unwrap(); // sigmoid can't introduce inconsistent histories
     * ```
     *
     * NB: Mapping a RecordMatrix of constants to variables is not inconsistent, the history
     * after mapping doesn't have to be the same as before, only must be the same for every
     * mapped element.
     */
    #[track_caller]
    pub fn map(
        &self,
        fx: impl Fn(Record<'a, T>) -> Record<'a, T>,
    ) -> Result<RecordMatrix<'a, T, Matrix<(T, Index)>>, InconsistentHistory<'a, T>> {
        let result = RecordMatrix::from_iter(self.size(), self.iter_row_major_as_records().map(fx));
        RecordMatrix::<'a, T, S>::map_collection(result, self.size())
    }

    /**
     * Creates a new RecordContainer from a reference to an existing RecordContainer by applying
     * some unary function on `Record<T>` and each index of that position in the Record to
     * `Record<T>` to every element in the container. This will fail if the function would
     * create records with inconsistent histories.
     *
     * When used with pure functions that can't return different histories for different inputs
     * unwrapping with always succeed.
     *
     * This API can allow you to call a generic function that operates on
     * [Numeric](crate::numeric::Numeric) numbers and apply all the correct derivative tracking
     * during the intermediate calculations for you, without having to resort to storing the
     * Record types.
     *
     * NB: Mapping a RecordMatrix of constants to variables is not inconsistent, the history
     * after mapping doesn't have to be the same as before, only must be the same for every
     * mapped element.
     */
    #[track_caller]
    pub fn map_with_index(
        &self,
        fx: impl Fn(Record<'a, T>, Row, Column) -> Record<'a, T>,
    ) -> Result<RecordMatrix<'a, T, Matrix<(T, Index)>>, InconsistentHistory<'a, T>> {
        let result = RecordMatrix::from_iter(
            self.size(),
            self.iter_row_major_as_records()
                .with_index()
                .map(|((r, c), x)| fx(x, r, c)),
        );
        RecordMatrix::<'a, T, S>::map_collection(result, self.size())
    }

    #[track_caller]
    fn map_collection(
        result: Result<
            RecordMatrix<'a, T, Matrix<(T, usize)>>,
            InvalidRecordIteratorError<'a, T, 2>,
        >,
        size: (Row, Column),
    ) -> Result<RecordMatrix<'a, T, Matrix<(T, usize)>>, InconsistentHistory<'a, T>> {
        use InvalidRecordIteratorError as Error;
        match result {
            Ok(matrix) => Ok(matrix),
            Err(error) => match error {
                // These first two should be 100% impossible but provide a sensible error just
                // in case some weird things break our invariants
                Error::Empty => panic!(
                    "Illegal state, record matrix was empty {:?}",
                    size
                ),
                Error::Shape { requested, length } => panic!(
                    "Illegal state, record matrix shape was inconsistent: requested: {:?}, length of data: {:?}",
                    requested,
                    length
                ),
                // This one is theoretically possible but in practise shouldn't happen by accident
                // However, it can't implement Debug unless T is debug so to avoid having to
                // restrict our function signature we return a Result anyway - this also encourages
                // the user to make sure their function isn't going to cause this case, which
                // with some of the other variants like with_index might come up more easily
                Error::InconsistentHistory(h) => Err(h),
            }
        }
    }

    /**
     * For each record in the container, peforms a backward pass up its WengertList from it
     * as the output, computing all the derivatives for the inputs involving this output.
     *
     * If this container has no backing WengertList, ie was created as constants, then None is
     * returned instead. Otherwise the returned Matrix will have the same size as this container,
     * with the respective derivatives matching each element in this container.
     *
     * If you have N inputs x<sub>1</sub> to x<sub>N</sub>, and this output is Y with M outputs,
     * then this computes all the derivatives δy<sub>j</sub>/δx<sub>i</sub> for i = 1 to N and
     * j = 1 to M.
     *
     * If you have a lot of outputs this could be very expensive! Reverse auto diff is optimised
     * for domains where there are many more inputs than outputs.
     *
     * If you only need some of the derivatives then
     * [derivatives_for](RecordMatrix::derivatives_for) can be used instead to avoid
     * calculating the rest.
     */
    pub fn derivatives(&self) -> Option<Matrix<Derivatives<T>>> {
        self.history.map(|history| {
            self.numbers.map(|(x, i)| {
                Record {
                    number: x,
                    history: Some(history),
                    index: i,
                }
                .derivatives()
            })
        })
    }

    /**
     * For the record at the index, peforms a backward pass up its WengertList from it
     * as the output, computing all the derivatives for the inputs involving this output.
     *
     * If the index is invalid or this container has no backing WengertList, ie was created
     * as constants, then None is returned instead.
     *
     * If you have N inputs x<sub>1</sub> to x<sub>N</sub>, and this output is y,
     * then this computes all the derivatives δy/δx<sub>i</sub> for i = 1 to N.
     */
    pub fn derivatives_for(&self, row: Row, column: Column) -> Option<Derivatives<T>> {
        let (number, index) = match self
            .try_get_reference(row, column)
            .map(|(x, i)| (x.clone(), *i))
        {
            Some(tuple) => tuple,
            None => return None,
        };
        // The nature of reverse autodiff is that we expect to only have a few outputs from
        // which we calculate all the derivatives we care about. Therefore just call Record and
        // reuse the implementation instead of trying to do anything clever like calculate all
        // derivatives for every number in this container.
        Record {
            number,
            history: self.history,
            index,
        }
        .try_derivatives()
    }

    /**
     * Performs elementwise multiplication for two record matrices of the same size.
     *
     * # Panics
     *
     * - If both record containers have a WengertList that are different to each other
     * - If the record containers have different shapes
     */
    // TODO: Assign variants?
    pub fn elementwise_multiply<S2>(
        &self,
        other: &RecordMatrix<'a, T, S2>,
    ) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
    where
        S2: MatrixRef<(T, Index)> + NoInteriorMutability,
    {
        self.binary(
            other,
            Multiplication::<T>::function,
            Multiplication::<T>::d_function_dx,
            Multiplication::<T>::d_function_dy,
        )
    }

    /**
     * Performs elementwise division for two record matrices of the same size.
     *
     * # Panics
     *
     * - If both record containers have a WengertList that are different to each other
     * - If the record containers have different shapes
     */
    pub fn elementwise_divide<S2>(
        &self,
        other: &RecordMatrix<'a, T, S2>,
    ) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
    where
        S2: MatrixRef<(T, Index)> + NoInteriorMutability,
    {
        self.binary(
            other,
            Division::<T>::function,
            Division::<T>::d_function_dx,
            Division::<T>::d_function_dy,
        )
    }
}

impl<'a, T, S> RecordMatrix<'a, T, S>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: MatrixMut<(T, Index)> + NoInteriorMutability,
{
    /**
     * Overwrites a RecordContainer by applying
     * some unary function from `T` to `T` to every element in the container.
     *
     * To compute the new records, the unary function of some input x to some
     * output y is needed along with its derivative with respect to its input x.
     */
    #[track_caller]
    pub fn unary_assign(&mut self, fx: impl Fn(T) -> T, dfx_dx: impl Fn(T) -> T) {
        let total = self.elements();
        match self.history {
            None => self.numbers.map_mut(|(x, i)| (fx(x), i)),
            Some(history) => {
                let ys = unary::<T, _>(total, history, self.numbers.row_major_iter(), fx, dfx_dx);
                for (element, result) in self.numbers.row_major_reference_mut_iter().zip(ys) {
                    *element = result;
                }
                self.history = Some(history);
            }
        }
    }

    /**
     * Overwrites the left hand side of a RecordContainer with the result of applying
     * some binary function from `T` to `T` to every element pair in the containers. Both
     * containers must have the same shape.
     * To compute the new records, the binary function of some inputs x and y to some
     * output z is needed along with its derivative with respect to its first input x and
     * its derivative with respect to its second input y.
     *
     * # Panics
     *
     * - If both record containers have a WengertList that are different to each other
     * - If the record containers have different shapes
     */
    #[track_caller]
    pub fn binary_left_assign<S2>(
        &mut self,
        rhs: &RecordMatrix<'a, T, S2>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) where
        S2: MatrixRef<(T, Index)> + NoInteriorMutability,
    {
        {
            let left_shape = self.numbers.size();
            let right_shape = rhs.numbers.size();
            if left_shape != right_shape {
                panic!(
                    "Record containers must have the same size for a binary operation: (left: {:?}, right: {:?})",
                    left_shape,
                    right_shape
                );
            }
        }
        let total = self.elements();
        match (self.history, rhs.history) {
            (None, None) => {
                for (x, y) in self
                    .numbers
                    .row_major_reference_mut_iter()
                    .zip(rhs.numbers.row_major_iter())
                {
                    let (left, _) = x;
                    let (right, _) = y;
                    *x = (fxy(left.clone(), right), 0);
                }
            }
            (Some(history), None) => {
                let zs = binary_x_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.row_major_iter(),
                    rhs.numbers.row_major_iter(),
                    fxy,
                    dfxy_dx,
                );
                for (element, result) in self.numbers.row_major_reference_mut_iter().zip(zs) {
                    *element = result;
                }
                self.history = Some(history);
            }
            (None, Some(history)) => {
                let zs = binary_y_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.row_major_iter(),
                    rhs.numbers.row_major_iter(),
                    fxy,
                    dfxy_dy,
                );
                for (element, result) in self.numbers.row_major_reference_mut_iter().zip(zs) {
                    *element = result;
                }
                self.history = Some(history);
            }
            (Some(history), Some(h)) => {
                assert!(
                    record_operations::same_lists(history, h),
                    "Record containers must be using the same WengertList"
                );
                let zs = binary_both_history::<T, _, _>(
                    total,
                    history,
                    self.numbers.row_major_iter(),
                    rhs.numbers.row_major_iter(),
                    fxy,
                    dfxy_dx,
                    dfxy_dy,
                );
                for (element, result) in self.numbers.row_major_reference_mut_iter().zip(zs) {
                    *element = result;
                }
                self.history = Some(history);
            }
        }
    }

    /**
     * A convenience helper function which takes the RecordContainer value and
     * calls [unary_assign](RecordMatrix::unary_assign()) on it, returning
     * the record container which now contains the result of the operation.
     */
    #[track_caller]
    pub fn do_unary_assign(mut self, fx: impl Fn(T) -> T, dfx_dx: impl Fn(T) -> T) -> Self {
        self.unary_assign(fx, dfx_dx);
        self
    }

    /**
     * A convenience helper function which takes the left hand side by value and
     * calls [binary_left_assign](RecordMatrix::binary_left_assign()) on it, returning
     * the left hand side which now contains the result of the operation.
     */
    #[track_caller]
    pub fn do_binary_left_assign<S2>(
        mut self,
        rhs: &RecordMatrix<'a, T, S2>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) -> Self
    where
        S2: MatrixRef<(T, Index)> + NoInteriorMutability,
    {
        self.binary_left_assign(rhs, fxy, dfxy_dx, dfxy_dy);
        self
    }

    /**
     * Updates this RecordContainer in place by applying some unary function on `Record<T>` to
     * `Record<T>` to every element in the container. This will fail if the function would create
     * records with inconsistent histories.
     *
     * When used with pure functions that can't return different histories for different inputs
     * unwrapping with always succeed.
     *
     * Since this updates the container in place, if Err is returned then the data in this
     * RecordContainer is still available but it has been corrupted - at least one of the elements
     * should have a different history than what it will have because the mapping function created
     * inconsistent histories that couldn't be represented by the container as it only stores
     * one.
     *
     * This API can allow you to call a generic function that operates on
     * [Numeric](crate::numeric::Numeric) numbers and apply all the correct derivative tracking
     * during the intermediate calculations for you, without having to resort to storing the
     * Record types.
     *
     * NB: Mapping a RecordMatrix of constants to variables is not inconsistent, the history
     * after mapping doesn't have to be the same as before, only must be the same for every
     * mapped element.
     */
    // TODO: Example
    #[track_caller]
    pub fn map_mut(
        &mut self,
        fx: impl Fn(Record<'a, T>) -> Record<'a, T>,
    ) -> Result<(), InconsistentHistory<'a, T>> {
        let history = self.history;
        let new_history =
            map_mut_base::<'a, T, _, _>(RowMajorReferenceMutIterator::from(self), |x| {
                let record = Record::from_existing(x.clone(), history);
                let result = fx(record);
                *x = (result.number, result.index);
                result.history
            })?;
        self.history = new_history;
        Ok(())
    }

    /**
     * Updates this RecordContainer in place by applying some unary function on `Record<T>` and
     * each index of that position in the Record to `Record<T>` to every element in the container.
     * This will fail if the function would create records with inconsistent histories.
     *
     * When used with pure functions that can't return different histories for different inputs
     * unwrapping with always succeed.
     *
     * Since this updates the container in place, if Err is returned then the data in this
     * RecordContainer is still available but it has been corrupted - at least one of the elements
     * should have a different history than what it will have because the mapping function created
     * inconsistent histories that couldn't be represented by the container as it only stores
     * one.
     *
     * This API can allow you to call a generic function that operates on
     * [Numeric](crate::numeric::Numeric) numbers and apply all the correct derivative tracking
     * during the intermediate calculations for you, without having to resort to storing the
     * Record types.
     *
     * NB: Mapping a RecordMatrix of constants to variables is not inconsistent, the history
     * after mapping doesn't have to be the same as before, only must be the same for every
     * mapped element.
     */
    #[track_caller]
    pub fn map_mut_with_index(
        &mut self,
        fx: impl Fn(Record<'a, T>, Row, Column) -> Record<'a, T>,
    ) -> Result<(), InconsistentHistory<'a, T>> {
        let history = self.history;
        let new_history = map_mut_base::<'a, T, _, _>(
            RowMajorReferenceMutIterator::from(self).with_index(),
            |((r, c), x)| {
                let record = Record::from_existing(x.clone(), history);
                let result = fx(record, r, c);
                *x = (result.number, result.index);
                result.history
            },
        )?;
        self.history = new_history;
        Ok(())
    }
}

impl<'a, T, S> RecordMatrix<'a, T, S>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    /**
     * Overwrites the right hand side of a RecordContainer with the result of applying
     * some binary function from `T` to `T` to every element pair in the containers. Both
     * containers must have the same shape.
     * To compute the new records, the binary function of some inputs x and y to some
     * output z is needed along with its derivative with respect to its first input x and
     * its derivative with respect to its second input y.
     *
     * # Panics
     *
     * - If both record containers have a WengertList that are different to each other
     * - If the record containers have different shapes
     */
    #[track_caller]
    pub fn binary_right_assign<S2>(
        &self,
        rhs: &mut RecordMatrix<'a, T, S2>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) where
        S2: MatrixMut<(T, Index)> + NoInteriorMutability,
    {
        // x is lhs, y is rhs, so calling binary_left_assign on the rhs container
        // means we need to swap all the arguments
        rhs.binary_left_assign(
            self,
            |y, x| fxy(x, y),
            |y, x| dfxy_dy(x, y),
            |y, x| dfxy_dx(x, y),
        )
    }

    /**
     * A convenience helper function which takes the right hand side by value and
     * calls [binary_right_assign](RecordMatrix::binary_right_assign()) on it, returning
     * the right hand side which now contains the result of the operation.
     */
    #[track_caller]
    pub fn do_binary_right_assign<S2>(
        &self,
        mut rhs: RecordMatrix<'a, T, S2>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) -> RecordMatrix<'a, T, S2>
    where
        S2: MatrixMut<(T, Index)> + NoInteriorMutability,
    {
        self.binary_right_assign(&mut rhs, fxy, dfxy_dx, dfxy_dy);
        rhs
    }
}

// # Safety
//
// Our inner `numbers` tensor has to implement TensorRef correctly so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorRef
// correctly as well.
/**
 * RecordTensor implements TensorRef when the source does, returning references to the tuples
 * of `T` and [`Index`](Index).
 */
unsafe impl<'a, T, S, const D: usize> TensorRef<(T, Index), D> for RecordTensor<'a, T, S, D>
where
    T: Primitive,
    S: TensorRef<(T, Index), D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&(T, Index)> {
        self.numbers.source_ref().get_reference(indexes)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        self.numbers.source_ref().view_shape()
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &(T, Index) {
        self.numbers.source_ref().get_reference_unchecked(indexes)
    }

    fn data_layout(&self) -> DataLayout<D> {
        self.numbers.source_ref().data_layout()
    }
}

// # Safety
//
// Our inner `numbers` tensor has to implement TensorMut correctly so by delegating to it
// without changing any indexes or introducing interior mutability, we implement TensorMut
// correctly as well.
/**
 * RecordTensor implements TensorMut when the source does, returning mutable references to the
 * tuples of `T` and [`Index`](Index).
 */
unsafe impl<'a, T, S, const D: usize> TensorMut<(T, Index), D> for RecordTensor<'a, T, S, D>
where
    T: Primitive,
    S: TensorMut<(T, Index), D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut (T, Index)> {
        self.numbers.source_ref_mut().get_reference_mut(indexes)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut (T, Index) {
        self.numbers
            .source_ref_mut()
            .get_reference_unchecked_mut(indexes)
    }
}

// # Safety
//
// Our inner `numbers` matrix has to implement MatrixRef correctly so by delegating to it
// without changing any indexes or introducing interior mutability, we implement MatrixRef
// correctly as well.
/**
 * RecordMatrix implements MatrixRef when the source does, returning references to the tuples
 * of `T` and [`Index`](Index).
 */
unsafe impl<'a, T, S> MatrixRef<(T, Index)> for RecordMatrix<'a, T, S>
where
    T: Primitive,
    S: MatrixRef<(T, Index)>,
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&(T, Index)> {
        self.numbers.source_ref().try_get_reference(row, column)
    }

    fn view_rows(&self) -> Row {
        self.numbers.source_ref().view_rows()
    }

    fn view_columns(&self) -> Column {
        self.numbers.source_ref().view_columns()
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &(T, Index) {
        self.numbers
            .source_ref()
            .get_reference_unchecked(row, column)
    }

    fn data_layout(&self) -> crate::matrices::views::DataLayout {
        self.numbers.source_ref().data_layout()
    }
}

// # Safety
//
// Our inner `numbers` matrix has to implement NoInteriorMutability correctly so by delegating to
// it without introducing interior mutability, we implement NoInteriorMutability
// correctly as well.
/**
 * RecordMatrix implements NoInteriorMutability when the source does.
 */
unsafe impl<'a, T, S> NoInteriorMutability for RecordMatrix<'a, T, S>
where
    T: Primitive,
    S: NoInteriorMutability,
{
}

// # Safety
//
// Our inner `numbers` matrix has to implement MatrixMut correctly so by delegating to it
// without changing any indexes or introducing interior mutability, we implement MatrixMut
// correctly as well.
/**
 * RecordMatrix implements MatrixMut when the source does, returning mutable references to the
 * tuples of `T` and [`Index`].
 */
unsafe impl<'a, T, S> MatrixMut<(T, Index)> for RecordMatrix<'a, T, S>
where
    T: Primitive,
    S: MatrixMut<(T, Index)>,
{
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut (T, Index)> {
        self.numbers
            .source_ref_mut()
            .try_get_reference_mut(row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut (T, Index) {
        self.numbers
            .source_ref_mut()
            .get_reference_unchecked_mut(row, column)
    }
}

// TODO: Delete these in version 2.0 as can add blanket impl to MatrixRef itself

// # Safety
//
// Our inner `numbers` matrix has to implement MatrixRef correctly so by delegating to it
// without changing any indexes or introducing interior mutability, we implement MatrixRef
// correctly as well.
/**
 * RecordMatrix implements MatrixRef when the source does, returning references to the tuples
 * of `T` and [`Index`].
 */
unsafe impl<'a, T, S> MatrixRef<(T, Index)> for &RecordMatrix<'a, T, S>
where
    T: Primitive,
    S: MatrixRef<(T, Index)>,
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&(T, Index)> {
        self.numbers.source_ref().try_get_reference(row, column)
    }

    fn view_rows(&self) -> Row {
        self.numbers.source_ref().view_rows()
    }

    fn view_columns(&self) -> Column {
        self.numbers.source_ref().view_columns()
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &(T, Index) {
        self.numbers
            .source_ref()
            .get_reference_unchecked(row, column)
    }

    fn data_layout(&self) -> crate::matrices::views::DataLayout {
        self.numbers.source_ref().data_layout()
    }
}

// # Safety
//
// Our inner `numbers` matrix has to implement MatrixRef correctly so by delegating to it
// without changing any indexes or introducing interior mutability, we implement MatrixRef
// correctly as well.
/**
 * RecordMatrix implements MatrixRef when the source does, returning references to the tuples
 * of `T` and [`Index`].
 */
unsafe impl<'a, T, S> MatrixRef<(T, Index)> for &mut RecordMatrix<'a, T, S>
where
    T: Primitive,
    S: MatrixRef<(T, Index)>,
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&(T, Index)> {
        self.numbers.source_ref().try_get_reference(row, column)
    }

    fn view_rows(&self) -> Row {
        self.numbers.source_ref().view_rows()
    }

    fn view_columns(&self) -> Column {
        self.numbers.source_ref().view_columns()
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &(T, Index) {
        self.numbers
            .source_ref()
            .get_reference_unchecked(row, column)
    }

    fn data_layout(&self) -> crate::matrices::views::DataLayout {
        self.numbers.source_ref().data_layout()
    }
}

// # Safety
//
// Our inner `numbers` matrix has to implement MatrixMut correctly so by delegating to it
// without changing any indexes or introducing interior mutability, we implement MatrixMut
// correctly as well.
/**
 * RecordMatrix implements MatrixMut when the source does, returning mutable references to the
 * tuples of `T` and [`Index`].
 */
unsafe impl<'a, T, S> MatrixMut<(T, Index)> for &mut RecordMatrix<'a, T, S>
where
    T: Primitive,
    S: MatrixMut<(T, Index)>,
{
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut (T, Index)> {
        self.numbers
            .source_ref_mut()
            .try_get_reference_mut(row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut (T, Index) {
        self.numbers
            .source_ref_mut()
            .get_reference_unchecked_mut(row, column)
    }
}

/**
 * A zero dimensional record tensor can be converted losslessly into a record.
 */
impl<'a, T, S> From<RecordTensor<'a, T, S, 0>> for Record<'a, T>
where
    T: Numeric + Primitive,
    S: TensorRef<(T, Index), 0>,
{
    /**
     * Converts the sole element in the zero dimensional record tensor into a record.
     */
    fn from(scalar: RecordTensor<'a, T, S, 0>) -> Record<'a, T> {
        // Not a good way to make this zero copy and just move the data out of the scalar because
        // TensorRef API doesn't have by value moves of the data and using TensorOwnedIterator
        // requires T: Default and a dummy value (at which point a clone is probably cheaper or
        // basically the same?)
        Record::from(&scalar)
    }
}

/**
 * A zero dimensional record tensor can be converted losslessly into a record.
 */
impl<'a, T, S> From<&RecordTensor<'a, T, S, 0>> for Record<'a, T>
where
    T: Numeric + Primitive,
    S: TensorRef<(T, Index), 0>,
{
    /**
     * Converts the sole element in the zero dimensional record tensor into a record.
     */
    fn from(scalar: &RecordTensor<'a, T, S, 0>) -> Record<'a, T> {
        Record::from_existing(scalar.view().scalar(), scalar.history)
    }
}

/**
 * A record can be converted losslessly into a zero dimensional record tensor.
 */
impl<'a, T> From<Record<'a, T>> for RecordTensor<'a, T, Tensor<(T, Index), 0>, 0>
where
    T: Numeric + Primitive,
{
    /**
     * Converts a record into a zero dimensional record tensor with the single element.
     */
    fn from(record: Record<'a, T>) -> RecordTensor<'a, T, Tensor<(T, Index), 0>, 0> {
        RecordTensor::from_existing(
            record.history,
            TensorView::from(Tensor::from([], vec![(record.number, record.index)])),
        )
    }
}

/**
 * A record can be converted losslessly into a zero dimensional record tensor.
 */
impl<'a, T> From<&Record<'a, T>> for RecordTensor<'a, T, Tensor<(T, Index), 0>, 0>
where
    T: Numeric + Primitive,
{
    /**
     * Converts a record into a zero dimensional record tensor with the single element.
     */
    fn from(record: &Record<'a, T>) -> RecordTensor<'a, T, Tensor<(T, Index), 0>, 0> {
        RecordTensor::from_existing(
            record.history,
            TensorView::from(Tensor::from(
                [],
                vec![(record.number.clone(), record.index)],
            )),
        )
    }
}

#[test]
fn matrix_multiplication_derivatives_are_the_same() {
    #[rustfmt::skip]
    let a = Tensor::from(
        [("r", 4), ("c", 3)],
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            0.0, 5.0, 2.0
        ]
    );
    let b = a.transpose(["c", "r"]);
    let history = WengertList::new();
    let also_history: WengertList<f64> = WengertList::new();
    let tensor_of_records_a = a.map(|x| Record::variable(x, &history));
    let tensor_of_records_b = b.map(|x| Record::variable(x, &history));
    let tensor_of_records_c = &tensor_of_records_a * &tensor_of_records_b;
    let record_tensor_a = RecordTensor::variables(&also_history, a);
    let record_tensor_b = RecordTensor::variables(&also_history, b);
    let record_tensor_c = &record_tensor_a * &record_tensor_b;

    // C should be calculated the same in terms of the actual number
    assert_eq!(
        tensor_of_records_c.map(|r| r.number),
        TensorView::from(&record_tensor_c).map(|(n, _)| n)
    );

    let tensor_of_records_derivatives = tensor_of_records_c.map(|r| r.derivatives());
    let tensor_of_records_a_derivatives =
        tensor_of_records_derivatives.map(|d| d.at_tensor(&record_tensor_a));
    let tensor_of_records_b_derivatives =
        tensor_of_records_derivatives.map(|d| d.at_tensor(&record_tensor_b));

    let record_tensor_derivatives = record_tensor_c.derivatives().unwrap();
    let record_tensor_a_derivatives =
        record_tensor_derivatives.map(|d| d.at_tensor(&record_tensor_a));
    let record_tensor_b_derivatives =
        record_tensor_derivatives.map(|d| d.at_tensor(&record_tensor_b));

    // Every calculated derivative should match exactly
    assert_eq!(tensor_of_records_a_derivatives, record_tensor_a_derivatives);
    assert_eq!(tensor_of_records_b_derivatives, record_tensor_b_derivatives);

    // Verify C is actually calculated correctly
    #[rustfmt::skip]
    assert_eq!(
        tensor_of_records_c.map(|r| r.number),
        Tensor::from(
            [("r", 4), ("c", 4)],
            vec![
                14.0, 32.0, 50.0, 16.0,
                32.0, 77.0, 122.0, 37.0,
                50.0, 122.0, 194.0, 58.0,
                16.0, 37.0, 58.0, 29.0
            ]
        )
    );
    #[rustfmt::skip]
    assert_eq!(
        tensor_of_records_c.map(|r| r.number),
        Tensor::from(
            [("r", 4), ("c", 4)],
            vec![
                (1.0 * 1.0) + (2.0 * 2.0) + (3.0 * 3.0),
                (1.0 * 4.0) + (2.0 * 5.0) + (3.0 * 6.0),
                (1.0 * 7.0) + (2.0 * 8.0) + (3.0 * 9.0),
                (1.0 * 0.0) + (2.0 * 5.0) + (3.0 * 2.0),

                (4.0 * 1.0) + (5.0 * 2.0) + (6.0 * 3.0),
                (4.0 * 4.0) + (5.0 * 5.0) + (6.0 * 6.0),
                (4.0 * 7.0) + (5.0 * 8.0) + (6.0 * 9.0),
                (4.0 * 0.0) + (5.0 * 5.0) + (6.0 * 2.0),

                (7.0 * 1.0) + (8.0 * 2.0) + (9.0 * 3.0),
                (7.0 * 4.0) + (8.0 * 5.0) + (9.0 * 6.0),
                (7.0 * 7.0) + (8.0 * 8.0) + (9.0 * 9.0),
                (7.0 * 0.0) + (8.0 * 5.0) + (9.0 * 2.0),

                (0.0 * 1.0) + (5.0 * 2.0) + (2.0 * 3.0),
                (0.0 * 4.0) + (5.0 * 5.0) + (2.0 * 6.0),
                (0.0 * 7.0) + (5.0 * 8.0) + (2.0 * 9.0),
                (0.0 * 0.0) + (5.0 * 5.0) + (2.0 * 2.0)
            ]
        )
    );

    let tensor_of_records_derivatives = history.operations.borrow().clone();
    let record_tensor_derivatives = also_history.operations.borrow().clone();
    assert_eq!(
        tensor_of_records_derivatives.len(),
        record_tensor_derivatives.len()
    );
}

#[test]
fn matrix_view_matrix_multiplication_derivatives_are_the_same() {
    #[rustfmt::skip]
    let a = Matrix::from(vec![
        vec![ 1.0, 2.0, 3.0 ],
        vec![ 4.0, 5.0, 6.0 ],
        vec![ 7.0, 8.0, 9.0 ],
        vec![ 0.0, 5.0, 2.0 ]
    ]);
    let b = a.transpose();
    let history = WengertList::new();
    let also_history: WengertList<f64> = WengertList::new();
    let matrix_of_records_a = a.map(|x| Record::variable(x, &history));
    let matrix_of_records_b = b.map(|x| Record::variable(x, &history));
    let matrix_of_records_c = &matrix_of_records_a * &matrix_of_records_b;
    let record_matrix_a = RecordMatrix::variables(&also_history, a);
    let record_matrix_b = RecordMatrix::variables(&also_history, b);
    let record_matrix_c = &record_matrix_a * &record_matrix_b;

    // C should be calculated the same in terms of the actual number
    assert_eq!(
        matrix_of_records_c.map(|r| r.number),
        MatrixView::from(&record_matrix_c).map(|(n, _)| n)
    );

    let matrix_of_records_derivatives = matrix_of_records_c.map(|r| r.derivatives());
    let matrix_of_records_a_derivatives =
        matrix_of_records_derivatives.map(|d| d.at_matrix(&record_matrix_a));
    let matrix_of_records_b_derivatives =
        matrix_of_records_derivatives.map(|d| d.at_matrix(&record_matrix_b));

    let record_matrix_derivatives = record_matrix_c.derivatives().unwrap();
    let record_matrix_a_derivatives =
        record_matrix_derivatives.map(|d| d.at_matrix(&record_matrix_a));
    let record_matrix_b_derivatives =
        record_matrix_derivatives.map(|d| d.at_matrix(&record_matrix_b));

    // Every calculated derivative should match exactly
    assert_eq!(matrix_of_records_a_derivatives, record_matrix_a_derivatives);
    assert_eq!(matrix_of_records_b_derivatives, record_matrix_b_derivatives);

    // Verify C is actually calculated correctly
    #[rustfmt::skip]
    assert_eq!(
        matrix_of_records_c.map(|r| r.number),
        Matrix::from(vec![
                vec![ 14.0, 32.0, 50.0, 16.0 ],
                vec![ 32.0, 77.0, 122.0, 37.0 ],
                vec![ 50.0, 122.0, 194.0, 58.0 ],
                vec![ 16.0, 37.0, 58.0, 29.0 ]
            ]
        )
    );
    #[rustfmt::skip]
    assert_eq!(
        matrix_of_records_c.map(|r| r.number),
        Matrix::from(vec![
                vec![
                    (1.0 * 1.0) + (2.0 * 2.0) + (3.0 * 3.0),
                    (1.0 * 4.0) + (2.0 * 5.0) + (3.0 * 6.0),
                    (1.0 * 7.0) + (2.0 * 8.0) + (3.0 * 9.0),
                    (1.0 * 0.0) + (2.0 * 5.0) + (3.0 * 2.0),
                ],
                vec![
                    (4.0 * 1.0) + (5.0 * 2.0) + (6.0 * 3.0),
                    (4.0 * 4.0) + (5.0 * 5.0) + (6.0 * 6.0),
                    (4.0 * 7.0) + (5.0 * 8.0) + (6.0 * 9.0),
                    (4.0 * 0.0) + (5.0 * 5.0) + (6.0 * 2.0),
                ],
                vec![
                    (7.0 * 1.0) + (8.0 * 2.0) + (9.0 * 3.0),
                    (7.0 * 4.0) + (8.0 * 5.0) + (9.0 * 6.0),
                    (7.0 * 7.0) + (8.0 * 8.0) + (9.0 * 9.0),
                    (7.0 * 0.0) + (8.0 * 5.0) + (9.0 * 2.0),
                ],
                vec![
                    (0.0 * 1.0) + (5.0 * 2.0) + (2.0 * 3.0),
                    (0.0 * 4.0) + (5.0 * 5.0) + (2.0 * 6.0),
                    (0.0 * 7.0) + (5.0 * 8.0) + (2.0 * 9.0),
                    (0.0 * 0.0) + (5.0 * 5.0) + (2.0 * 2.0)
                ]
        ])
    );

    let matrix_of_records_derivatives = history.operations.borrow().clone();
    let record_matrix_derivatives = also_history.operations.borrow().clone();
    assert_eq!(
        matrix_of_records_derivatives.len(),
        record_matrix_derivatives.len()
    );
}
