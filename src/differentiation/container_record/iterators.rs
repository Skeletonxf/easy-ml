/*!
 * Record container iterators, for manipulating iterators of Records and converting back to
 * Record containers.
 */

use crate::differentiation::{Index, Primitive, Record, WengertList};
use crate::differentiation::{RecordMatrix, RecordTensor};
use crate::matrices::iterators::{ColumnMajorIterator, RowMajorIterator, WithIndex};
use crate::matrices::views::{MatrixRef, MatrixView, NoInteriorMutability};
use crate::matrices::{Column, Matrix, Row};
use crate::numeric::Numeric;
use crate::tensors::indexing::TensorIterator;
use crate::tensors::views::{TensorRef, TensorView};
use crate::tensors::{Dimension, InvalidShapeError, Tensor};

use std::error::Error;
use std::fmt;
use std::fmt::Debug;
use std::iter::{ExactSizeIterator, FusedIterator};

/**
 * A wrapper around another iterator of record data and a history for that iterator's data
 * that iterates though each element in the iterator as a [Record] instead.
 *
 * The main purpose of this representation is to allow manipulating one or more
 * [RecordContainer](crate::differentiation::RecordContainer)s as iterators of Records then
 * collect the iterator back into a RecordContainer so containers of Records with a shared history
 * don't have to store the history for each element but the richer Record API can be used for
 * operations anyway.
 *
 * ```
 * use easy_ml::differentiation::{WengertList, Record, RecordTensor};
 * use easy_ml::tensors::Tensor;
 * use easy_ml::numeric::Numeric;
 * use easy_ml::numeric::extra::Real;
 *
 * let history = WengertList::new();
 * let A = RecordTensor::variables(
 *     &history,
 *     Tensor::from_fn([("r", 3), ("c", 2)], |[r, c]| ((5 * r) + c) as f32)
 * );
 * let B = RecordTensor::variables(
 *     &history,
 *     Tensor::from([("x", 6)], vec![ 0.2, 0.1, 0.5, 0.3, 0.7, 0.9 ])
 * );
 *
 * fn power<T: Numeric + Real+ Copy>(x: T, y: T) -> T {
 *     x.pow(y)
 * }
 *
 * let result: RecordTensor<_, _, 2> = RecordTensor::from_iter(
 *     A.shape(),
 *     // iterators of records don't need to have matching shapes as long as the number
 *     // of elements matches the final shape
 *     A.iter_as_records().zip(B.iter_as_records()).map(|(x, y)| power(x, y))
 * ).expect("result should have 6 elements");
 * ```
 */
pub struct AsRecords<'a, I, T> {
    numbers: I,
    history: Option<&'a WengertList<T>>,
}

/**
 * AsRecords can be created from a RecordTensor to manipulate the data as an iterator of Records
 * then streamed back into a RecordTensor with [from_iter](RecordTensor::from_iter)
 *
 * See also: [map](RecordTensor::map), [map_mut](RecordTensor::map_mut)
 *
 * ```
 * use easy_ml::differentiation::{WengertList, Record, RecordTensor};
 * use easy_ml::tensors::Tensor;
 *
 * let history = WengertList::new();
 * let X = RecordTensor::constants(
 *     Tensor::from_fn([("r", 2), ("c", 2)], |[r, c]| (r + c) as f32)
 * );
 * let y = Record::variable(1.0, &history);
 * let result = RecordTensor::from_iter(
 *     [("r", 2), ("c", 2)],
 *     // Here we create each variable z from the constant in X and the variable y.
 *     // If we just did X + 1.0 we'd still have only constants, and we can't do X + y
 *     // directly because those traits aren't implemented.
 *     X.iter_as_records().map(|x| x + y)
 * );
 * // we can unwrap here because we know the iterator still contains 4 elements and they all
 * // have the same WengertList so we can convert back to a RecordTensor (which is now
 * // variables instead of constants)
 * let Z = result.unwrap();
 * let Z_indexing = Z.index();
 * assert_eq!(1.0, Z_indexing.get([0, 0]).0);
 * assert_eq!(2.0, Z_indexing.get([0, 1]).0);
 * assert_eq!(3.0, Z_indexing.get([1, 1]).0);
 * ```
 */
impl<'a, 'b, T, S, const D: usize>
    AsRecords<'a, TensorIterator<'b, (T, Index), RecordTensor<'a, T, S, D>, D>, T>
where
    T: Numeric + Primitive,
    S: TensorRef<(T, Index), D>,
{
    /**
     * Given a record tensor returns an iterator of Records
     *
     * ```
     * use easy_ml::differentiation::{WengertList, Record, RecordTensor};
     * use easy_ml::differentiation::iterators::AsRecords;
     * use easy_ml::tensors::Tensor;
     *
     * let history = WengertList::new();
     * let X = RecordTensor::variables(
     *     &history,
     *     Tensor::from_fn([("r", 2), ("c", 2)], |[r, c]| (r + c) as f32)
     * );
     * let iter = X.iter_as_records(); // shorthand helper method
     * let also_iter = AsRecords::from_tensor(&X);
     * ```
     */
    pub fn from_tensor(tensor: &'b RecordTensor<'a, T, S, D>) -> Self {
        AsRecords::from(tensor.history, TensorIterator::from(tensor))
    }
}

/**
 * AsRecords can be created from a RecordMatrix to manipulate the data as an iterator of Records
 * then streamed back into a RecordMatrix with [from_iter](RecordMatrix::from_iter)
 *
 * See also: [map](RecordMatrix::map), [map_mut](RecordMatrix::map_mut)
 *
 * ```
 * use easy_ml::differentiation::{WengertList, Record, RecordMatrix};
 * use easy_ml::matrices::Matrix;
 *
 * let history = WengertList::new();
 * let X = RecordMatrix::constants(
 *     Matrix::from_fn((2, 2), |(r, c)| (r + c) as f32)
 * );
 * let y = Record::variable(1.0, &history);
 * let result = RecordMatrix::from_iter(
 *     (2, 2),
 *     // Here we create each variable z from the constant in X and the variable y.
 *     // If we just did X + 1.0 we'd still have only constants, and we can't do X + y
 *     // directly because those traits aren't implemented.
 *     X.iter_row_major_as_records().map(|x| x + y)
 * );
 * // we can unwrap here because we know the iterator still contains 4 elements and they all
 * // have the same WengertList so we can convert back to a RecordMatrix (which is now
 * // variables instead of constants)
 * let Z = result.unwrap();
 * let Z_view = Z.view();
 * assert_eq!(1.0, Z_view.get(0, 0).0);
 * assert_eq!(2.0, Z_view.get(0, 1).0);
 * assert_eq!(3.0, Z_view.get(1, 1).0);
 * ```
 */
impl<'a, 'b, T, S> AsRecords<'a, RowMajorIterator<'b, (T, Index), RecordMatrix<'a, T, S>>, T>
where
    T: Numeric + Primitive,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    /**
     * Given a record matrix returns a row major iterator of Records
     *
     * ```
     * use easy_ml::differentiation::{WengertList, Record, RecordMatrix};
     * use easy_ml::differentiation::iterators::AsRecords;
     * use easy_ml::matrices::Matrix;
     *
     * let history = WengertList::new();
     * let X = RecordMatrix::variables(
     *     &history,
     *     Matrix::from_fn((2, 2), |(r, c)| (r + c) as f32)
     * );
     * let iter = X.iter_row_major_as_records(); // shorthand helper method
     * let also_iter = AsRecords::from_matrix_row_major(&X);
     * ```
     */
    pub fn from_matrix_row_major(matrix: &'b RecordMatrix<'a, T, S>) -> Self {
        AsRecords::from(matrix.history, RowMajorIterator::from(matrix))
    }
}

impl<'a, 'b, T, S> AsRecords<'a, ColumnMajorIterator<'b, (T, Index), RecordMatrix<'a, T, S>>, T>
where
    T: Numeric + Primitive,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    /**
     * Given a record matrix returns a column major iterator of Records
     */
    pub fn from_matrix_column_major(matrix: &'b RecordMatrix<'a, T, S>) -> Self {
        AsRecords::from(matrix.history, ColumnMajorIterator::from(matrix))
    }
}

impl<'a, I, T> AsRecords<'a, I, T>
where
    T: Numeric + Primitive,
    I: Iterator<Item = (T, Index)>,
{
    /**
     * Given the WengertList an iterator of record numbers are for, returns an iterator of Records
     *
     * **The inputs are not checked for validity**. It is possible to pass in the wrong Wengert
     * list here or even numbers with indexes that aren't tracked on the WengertList.
     *
     * Where possible, consider using [from_tensor](AsRecords::from_tensor),
     * [from_matrix_row_major](AsRecords::from_matrix_row_major) or
     * [from_matrix_column_major](AsRecords::from_matrix_row_major) instead.
     */
    pub fn from(history: Option<&'a WengertList<T>>, numbers: I) -> Self {
        AsRecords { numbers, history }
    }
}

impl<'a, I, T> AsRecords<'a, I, T>
where
    T: Numeric + Primitive,
    I: Iterator<Item = (T, Index)> + Into<WithIndex<I>>,
{
    /**
     * An iterator of Records that is created from an iterator which can provide the index for
     * each element can also be coverted to a [WithIndex](WithIndex) iterator and provide the
     * index for each record.
     *
     * WithIndex appears twice in the return type because the original iterator itself is wrapped
     * in WithIndex to create an iterator that provides indexes, and AsRecords must also be
     * wrapped in WithIndex to implement the iterator trait with indexes from the original
     * iterator's implementation.
     *
     * ```
     * use easy_ml::differentiation::{WengertList, Record, RecordTensor};
     * use easy_ml::tensors::Tensor;
     *
     * let history = WengertList::new();
     * let X = RecordTensor::variables(
     *     &history,
     *     Tensor::from([("r", 2), ("c", 2)], vec![ 0.5, 1.5, 2.5, 3.5 ])
     * );
     * let Y = RecordTensor::from_iter(
     *     [("r", 2), ("c", 2)],
     *     // Most Easy ML matrix and tensor iterators implement Into<WithIndex<Self>>, so we can
     *     // call with_index after creating the iterator
     *     X.iter_as_records().with_index().map(|([r, c], x)| x + ((r + (2 * c)) as f32))
     * ).unwrap(); // we can unwrap here because we know the iterator is still 4 elements
     * // so matches the shape and we added constants to each Record element so the history
     * // is still consistent
     * assert_eq!(
     *     Tensor::from([("r", 2), ("c", 2)], vec![ 0.5, 3.5, 3.5, 6.5 ]),
     *     Y.view().map(|(x, _)| x)
     * );
     * ```
     */
    pub fn with_index(self) -> WithIndex<AsRecords<'a, WithIndex<I>, T>> {
        WithIndex {
            iterator: AsRecords {
                numbers: self.numbers.into(),
                history: self.history,
            },
        }
    }
}

impl<'a, I, O, T> AsRecords<'a, I, T>
where
    T: Numeric + Primitive,
    I: Iterator<Item = (O, (T, Index))>,
{
    /**
     * Given the WengertList an iterator of indexes and record numbers are for, returns an
     * iterator of indexes and Records
     *
     * **The inputs are not checked for validity**. It is possible to pass in the wrong Wengert
     * list here or even numbers with indexes that aren't tracked on the WengertList.
     *
     * Where possible, consider using [with_index](AsRecords::with_index) on an existing iterator
     * instead.
     */
    pub fn from_with_index(history: Option<&'a WengertList<T>>, numbers: I) -> Self {
        AsRecords { numbers, history }
    }
}

impl<'a, I, T> From<AsRecords<'a, I, T>> for WithIndex<AsRecords<'a, WithIndex<I>, T>>
where
    T: Numeric + Primitive,
    I: Iterator<Item = (T, Index)> + Into<WithIndex<I>>,
{
    fn from(iterator: AsRecords<'a, I, T>) -> Self {
        iterator.with_index()
    }
}

/**
 * AsRecords is an iterator of [Record](Record)s, merging the history together with each iterator
 * element.
 */
impl<'a, I, T> Iterator for AsRecords<'a, I, T>
where
    T: Numeric + Primitive,
    I: Iterator<Item = (T, Index)>,
{
    type Item = Record<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.numbers
            .next()
            .map(|number| Record::from_existing(number, self.history))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.numbers.size_hint()
    }
}

impl<'a, I, T> FusedIterator for AsRecords<'a, I, T>
where
    T: Numeric + Primitive,
    I: Iterator<Item = (T, Index)> + FusedIterator,
{
}

impl<'a, I, T> ExactSizeIterator for AsRecords<'a, I, T>
where
    T: Numeric + Primitive,
    I: Iterator<Item = (T, Index)> + ExactSizeIterator,
{
}

/**
 * When AsRecords contains an iterator `I` with the index `O` for each element, it is an iterator
 * of `O` and [Record]s, merging the history together with each iterator element.
 *
 * Depending on what iterator and `with_index` implementation was used, `O` might be the tuple
 * indexes for a matrix or the `const D: usize` length array of indexes for a tensor. In either
 * case the iterator implementation for `WithIndex<AsRecords<..>>` just forwards the `O` values
 * unchanged so it works with both.
 */
impl<'a, I, O, T> Iterator for WithIndex<AsRecords<'a, I, T>>
where
    T: Numeric + Primitive,
    I: Iterator<Item = (O, (T, Index))>,
{
    type Item = (O, Record<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.iterator
            .numbers
            .next()
            .map(|(i, number)| (i, Record::from_existing(number, self.iterator.history)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iterator.numbers.size_hint()
    }
}

impl<'a, I, O, T> FusedIterator for WithIndex<AsRecords<'a, I, T>>
where
    T: Numeric + Primitive,
    I: Iterator<Item = (O, (T, Index))> + FusedIterator,
{
}

impl<'a, I, O, T> ExactSizeIterator for WithIndex<AsRecords<'a, I, T>>
where
    T: Numeric + Primitive,
    I: Iterator<Item = (O, (T, Index))> + ExactSizeIterator,
{
}

/**
 * An error due to an invalid record iterator. One of three cases
 *
 * - `Shape`: the iterator data didn't match the number of elements needed for a given shape to
 * convert back into a record container
 * - `Empty`: the iterator was empty, which is always an invalid length for any shape
 * - `InconsistentHistory`: the iterator contains inconsistent histories in its data and so cannot
 * be converted into a record container because a record container can only have one history for
 * all its data
 */
#[derive(Clone, Debug)]
pub enum InvalidRecordIteratorError<'a, T, const D: usize> {
    Shape {
        requested: InvalidShapeError<D>,
        length: usize,
    },
    Empty,
    InconsistentHistory(InconsistentHistory<'a, T>),
}

/**
 * An error due to trying to create a RecordContainer with record data that has more than one
 * history. Since RecordContainer stores the history once for all records it contains, it cannot
 * support constants + variables or variables from multiple WengertLists.
 */
#[derive(Clone, Debug)]
pub struct InconsistentHistory<'a, T> {
    pub first: Option<&'a WengertList<T>>,
    pub later: Option<&'a WengertList<T>>,
}

impl<'a, T, const D: usize> fmt::Display for InvalidRecordIteratorError<'a, T, D>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { requested, length } => write!(
                f,
                "Shape {:?} does not match size of data {}",
                requested.shape(),
                length
            ),
            Self::Empty => write!(
                f,
                "Iterator was empty but all tensors and matrices must contain at least one element"
            ),
            Self::InconsistentHistory(h) => write!(
                f,
                "First history in iterator of records was {:?} but a later history in iterator was {:?}, record container cannot support different histories for a single tensor or matrix.",
                h.first,
                h.later,
            )
        }
    }
}

impl<'a, T> fmt::Display for InconsistentHistory<'a, T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
           f,
           "First history was {:?} but a later history in iterator was {:?}, record container cannot support different histories for a single tensor or matrix.",
           self.first,
           self.later,
       )
    }
}

impl<'a, T, const D: usize> Error for InvalidRecordIteratorError<'a, T, D> where T: Debug {}

impl<'a, T> Error for InconsistentHistory<'a, T> where T: Debug {}

struct RecordContainerComponents<'a, T> {
    history: Option<&'a WengertList<T>>,
    numbers: Vec<(T, Index)>,
}

/// Converts an iterator of Records into the shared, consistent, history and a vec of (T, Index)
/// or fails if the history is not consistent or the iterator is empty.
fn collect_into_components<'a, T, I, const D: usize>(
    iter: I,
) -> Result<RecordContainerComponents<'a, T>, InvalidRecordIteratorError<'a, T, D>>
where
    T: Numeric + Primitive,
    I: IntoIterator<Item = Record<'a, T>>,
{
    use crate::differentiation::record_operations::are_exact_same_list;

    let mut history: Option<Option<&WengertList<T>>> = None;
    let mut error: Option<InvalidRecordIteratorError<'a, T, D>> = None;

    let numbers: Vec<(T, Index)> = iter
        .into_iter()
        .map(|record| {
            match history {
                None => history = Some(record.history),
                Some(h) => {
                    if !are_exact_same_list(h, record.history) {
                        error = Some(InvalidRecordIteratorError::InconsistentHistory(
                            InconsistentHistory {
                                first: h,
                                later: record.history,
                            },
                        ));
                    }
                }
            }
            (record.number, record.index)
        })
        .collect();

    if let Some(error) = error {
        return Err(error);
    }

    let data_length = numbers.len();
    if data_length == 0 {
        Err(InvalidRecordIteratorError::Empty)
    } else {
        // We already checked if the iterator was empty so `history` is always `Some` here
        Ok(RecordContainerComponents {
            history: history.unwrap(),
            numbers,
        })
    }
}

/// Converts an iterator of an array of Records into n shared, consistent, histories and vecs of
/// (T, Index) or fails individually if a history is not consistent or for all N if the iterator
/// is empty.
fn collect_into_n_components<'a, T, I, const D: usize, const N: usize>(
    iter: I,
) -> [Result<RecordContainerComponents<'a, T>, InvalidRecordIteratorError<'a, T, D>>; N]
where
    T: Numeric + Primitive,
    I: IntoIterator<Item = [Record<'a, T>; N]>,
{
    use crate::differentiation::record_operations::are_exact_same_list;

    let iter = iter.into_iter();

    // We have N unique histories, potential errors, and vecs of numbers

    let mut histories: [Option<Option<&WengertList<T>>>; N] = [None; N];

    let mut errors: [Option<InvalidRecordIteratorError<'a, T, D>>; N] =
        std::array::from_fn(|_| None);

    let mut numbers: [Vec<(T, usize)>; N] =
        std::array::from_fn(|_| Vec::with_capacity(iter.size_hint().0));

    // The entire benefit to this method is still streaming all the record iterators without ever
    // collecting more than one element's worth of records at a time, so we build up each vec
    // of numbers together as we pass through the iterator and discard the duplicate histories
    for records in iter {
        for (n, record) in records.into_iter().enumerate() {
            let history = &mut histories[n];
            let error = &mut errors[n];
            match history {
                None => *history = Some(record.history),
                Some(h) => {
                    if !are_exact_same_list(*h, record.history) {
                        *error = Some(InvalidRecordIteratorError::InconsistentHistory(
                            InconsistentHistory {
                                first: *h,
                                later: record.history,
                            },
                        ));
                    }
                }
            }
            numbers[n].push((record.number, record.index));
        }
    }

    let mut histories = histories.into_iter();
    let mut errors = errors.into_iter();
    let mut numbers = numbers.into_iter();
    std::array::from_fn(|_| {
        // unwrap always succeeds because we're consuming 3 iterators each of length N a total of N
        // times
        let history = histories.next().unwrap();
        let error = errors.next().unwrap();
        let numbers = numbers.next().unwrap();
        let data_length = numbers.len();
        match error {
            Some(error) => Err(error),
            None => {
                if data_length == 0 {
                    Err(InvalidRecordIteratorError::Empty)
                } else {
                    // We already checked if the iterator was empty so `history` is always
                    // `Some` here
                    Ok(RecordContainerComponents {
                        history: history.unwrap(),
                        numbers,
                    })
                }
            }
        }
    })
}

impl<'a, T, const D: usize> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
{
    /**
     * Given an iterator of records and a matching shape, collects them back into a
     * [RecordTensor](RecordTensor).
     *
     * This should generally be preferred over converting the iterator to a [Vec] of Records, since
     * a Vec of Records has to store the [WengertList](WengertList) reference for each individual
     * record whereas a RecordTensor only stores it once.
     *
     * However, since a RecordTensor only stores the WengertList once, this conversion will fail
     * if there are different histories in the iterator. It also fails if the iterator is empty
     * or doesn't match the number of elements for the shape.
     *
     * See also: [elements](crate::tensors::dimensions::elements)
     */
    pub fn from_iter<I>(
        shape: [(Dimension, usize); D],
        iter: I,
    ) -> Result<Self, InvalidRecordIteratorError<'a, T, D>>
    where
        I: IntoIterator<Item = Record<'a, T>>,
    {
        let RecordContainerComponents { history, numbers } = collect_into_components(iter)?;
        let data_length = numbers.len();
        match Tensor::try_from(shape, numbers) {
            Ok(numbers) => Ok(RecordTensor::from_existing(
                history,
                TensorView::from(numbers),
            )),
            Err(invalid_shape) => Err(InvalidRecordIteratorError::Shape {
                requested: invalid_shape,
                length: data_length,
            }),
        }
    }

    /**
     * Given an iterator of N record pairs and a matching shape, collects them back into N
     * [RecordTensor](RecordTensor)s.
     *
     * This should generally be preferred over converting the iterator to N [Vec]s of Records,
     * since a Vec of Records has to store the [WengertList](WengertList) reference for each
     * individual record whereas a RecordTensor only stores it once.
     *
     * However, since a RecordTensor only stores the WengertList once, this conversion will fail
     * if there are different histories in the iterator. It also fails if the iterator is empty
     * or doesn't match the number of elements for the shape. Each failure due to different
     * histories is seperate, if the ith elements in the records of the iterator have a
     * consistent history but the jth elements do not then the ith result will be Ok but the
     * jth will be Err.
     *
     * See also: [elements](crate::tensors::dimensions::elements)
     */
    pub fn from_iters<I, const N: usize>(
        shape: [(Dimension, usize); D],
        iter: I,
    ) -> [Result<Self, InvalidRecordIteratorError<'a, T, D>>; N]
    where
        I: IntoIterator<Item = [Record<'a, T>; N]>,
    {
        let mut components = collect_into_n_components(iter).into_iter();
        std::array::from_fn(|_| match components.next().unwrap() {
            Err(error) => Err(error),
            Ok(RecordContainerComponents { history, numbers }) => {
                let data_length = numbers.len();
                match Tensor::try_from(shape, numbers) {
                    Ok(numbers) => Ok(RecordTensor::from_existing(
                        history,
                        TensorView::from(numbers),
                    )),
                    Err(invalid_shape) => Err(InvalidRecordIteratorError::Shape {
                        requested: invalid_shape,
                        length: data_length,
                    }),
                }
            }
        })
    }
}

impl<'a, T> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
{
    /**
     * Given an iterator of records and a matching size, collects them back into a
     * [RecordMatrix](RecordMatrix).
     *
     * This should generally be preferred over converting the iterator to a [Vec] of Records, since
     * a Vec of Records has to store the [WengertList](WengertList) reference for each individual
     * record whereas a RecordMatrix only stores it once.
     *
     * However, since a RecordMatrix only stores the WengertList once, this conversion will fail
     * if there are different histories in the iterator. It also fails if the iterator is empty
     * or doesn't match the R x C number of elements expected.
     */
    pub fn from_iter<I>(
        size: (Row, Column),
        iter: I,
    ) -> Result<Self, InvalidRecordIteratorError<'a, T, 2>>
    where
        I: IntoIterator<Item = Record<'a, T>>,
    {
        let RecordContainerComponents { history, numbers } = collect_into_components(iter)?;
        let data_length = numbers.len();
        if data_length == size.0 * size.1 {
            Ok(RecordMatrix::from_existing(
                history,
                MatrixView::from(Matrix::from_flat_row_major(size, numbers)),
            ))
        } else {
            Err(InvalidRecordIteratorError::Shape {
                requested: InvalidShapeError::new([("rows", size.0), ("columns", size.1)]),
                length: data_length,
            })
        }
    }

    /**
     * Given an iterator of N record pairs and a matching shape, collects them back into N
     * [RecordMatrix](RecordMatrix)s.
     *
     * This should generally be preferred over converting the iterator to N [Vec]s of Records,
     * since a Vec of Records has to store the [WengertList](WengertList) reference for each
     * individual record whereas a RecordMatrix only stores it once.
     *
     * However, since a RecordMatrix only stores the WengertList once, this conversion will fail
     * if there are different histories in the iterator. It also fails if the iterator is empty
     * or doesn't match the R x C number of elements expected. Each failure due to different
     * histories is seperate, if the ith elements in the records of the iterator have a
     * consistent history but the jth elements do not then the ith result will be Ok but the
     * jth will be Err.
     *
     * See also: [elements](crate::tensors::dimensions::elements)
     */
    pub fn from_iters<I, const N: usize>(
        size: (Row, Column),
        iter: I,
    ) -> [Result<Self, InvalidRecordIteratorError<'a, T, 2>>; N]
    where
        I: IntoIterator<Item = [Record<'a, T>; N]>,
    {
        let mut components = collect_into_n_components(iter).into_iter();
        std::array::from_fn(|_| match components.next().unwrap() {
            Err(error) => Err(error),
            Ok(RecordContainerComponents { history, numbers }) => {
                let data_length = numbers.len();
                if data_length == size.0 * size.1 {
                    Ok(RecordMatrix::from_existing(
                        history,
                        MatrixView::from(Matrix::from_flat_row_major(size, numbers)),
                    ))
                } else {
                    Err(InvalidRecordIteratorError::Shape {
                        requested: InvalidShapeError::new([("rows", size.0), ("columns", size.1)]),
                        length: data_length,
                    })
                }
            }
        })
    }
}
