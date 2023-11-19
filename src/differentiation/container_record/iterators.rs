/*!
 * Record container iterators, for manipulating iterators of Records and converting back to
 * Record containers.
 */

use crate::differentiation::{Index, Primitive, Record, WengertList};
use crate::differentiation::{RecordContainer, RecordMatrix, RecordTensor};
use crate::matrices::iterators::WithIndex;
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
 */
pub struct AsRecords<'a, I, T> {
    numbers: I,
    history: Option<&'a WengertList<T>>,
}

// TODO: Helper methods on RecordContainer to convert to iterator of records
// Doc example using zip to perform binary operation then collecting back?
// From impls for RecordContainer?

// doc example of Record variable elementwise with RecordContainer of constants to create
// RecordContainer of variables

impl<'a, 'b, T, S, const D: usize>
    AsRecords<'a, TensorIterator<'b, (T, Index), RecordTensor<'a, T, S, D>, D>, T>
where
    T: Numeric + Primitive,
    S: TensorRef<(T, Index), D>,
{
    /**
     * Given a record tensor returns an iterator of Records
     */
    pub fn from_tensor(tensor: &'b RecordTensor<'a, T, S, D>) -> Self {
        AsRecords::from(tensor.history, TensorIterator::from(tensor))
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
     * Where possible, consider using [from_tensor](AsRecords::from_tensor) instead.
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
     * Where possible, consider using [with_index](AsRecords::with_index) instead.
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
 * - the iterator data didn't match the number of elements needed for a given shape to convert
 * back into a record container
 * - the iterator was empty, which is always an invalid length for any shape
 * - the iterator contains inconsistent histories in its data and so cannot be converted into
 * a record container because a record container can only have one history for all its data
 */
#[derive(Clone, Debug)]
enum InvalidRecordIteratorError<'a, T, const D: usize> {
    Shape {
        requested: InvalidShapeError<D>,
        length: usize,
    },
    Empty,
    InconsistentHistory {
        first: Option<&'a WengertList<T>>,
        later: Option<&'a WengertList<T>>,
    },
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
            Self::InconsistentHistory { first, later } => write!(
                f,
                "First history in iterator of records was {:?} but a later history in iterator was {:?}, record container cannot support different histories for a single tensor or matrix.",
                first,
                later,
            )
        }
    }
}

impl<'a, T, const D: usize> Error for InvalidRecordIteratorError<'a, T, D> where T: Debug {}

impl<'a, T, const D: usize> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
{
    /**
     * Given an iterator of records and a matching shape, collects them back into a RecordTensor.
     *
     * This should generally be preferred over converting the iterator to a vec of Records, since
     * a vec of Records has to store the WengertList for each individual record whereas a
     * RecordTensor only stores it once.
     *
     * However, since a RecordTensor only stores the WengertList once, this conversion will fail
     * if there are different histories in the iterator. It also fails if the iterator is empty
     * or doesn't match the number of elements for the shape.
     *
     * See also: [elements](crate::tensors::dimensions::elements)
     */
    fn from_iter<I>(
        iter: I,
        shape: [(Dimension, usize); D],
    ) -> Result<Self, InvalidRecordIteratorError<'a, T, D>>
    where
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
                            error = Some(InvalidRecordIteratorError::InconsistentHistory {
                                first: h,
                                later: record.history,
                            });
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
            return Err(InvalidRecordIteratorError::Empty);
        }

        let numbers = TensorView::from(match Tensor::try_from(shape, numbers) {
            Ok(numbers) => numbers,
            Err(invalid_shape) => {
                return Err(InvalidRecordIteratorError::Shape {
                    requested: invalid_shape,
                    length: data_length,
                })
            }
        });

        // We already checked if the iterator was empty so `history` is always `Some` here
        Ok(RecordTensor::from_existing(history.unwrap(), numbers))
    }
}
