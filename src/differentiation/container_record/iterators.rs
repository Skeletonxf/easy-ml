/**
 * Record container iterators, for manipulating iterators of Records and converting back to
 * Record containers.
 */

use crate::numeric::Numeric;
use crate::differentiation::{WengertList, Index, Primitive, Record};
use crate::differentiation::{RecordContainer, RecordTensor, RecordMatrix};
use crate::tensors::views::{TensorRef, TensorView};
use crate::tensors::{Dimension, Tensor, InvalidShapeError};
use crate::tensors::indexing::TensorIterator;

// TODO: Make this play nice with WithIndex

pub struct AsRecords<'a, I, T> {
    numbers: I,
    history: Option<&'a WengertList<T>>,
}

// TODO: Helper methods on RecordContainer to convert to iterator of records
// Doc example using zip to perform binary operation then collecting back?
// From impls for RecordContainer?

// doc example of Record variable elementwise with RecordContainer of constants to create
// RecordContainer of variables

impl<'a, 'b, T, S, const D: usize> AsRecords<'a, TensorIterator<'b, (T, Index), RecordTensor<'a, T, S, D>, D>, T>
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
        AsRecords {
            numbers,
            history,
        }
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
        self.numbers.next().map(|number| Record::from_existing(number, self.history))
    }
}

// TODO: Implement traits, Error, Debug, Clone?
enum InvalidRecordIteratorError<'a, T, const D: usize> {
    Shape(InvalidShapeError<D>),
    Empty,
    InconsistentHistory {
        first: Option<&'a WengertList<T>>,
        later: Option<&'a WengertList<T>>,
    },
}

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
     */
    fn from_iter<I>(
        iter: I,
        shape: [(Dimension, usize); D]
    ) -> Result<Self, InvalidRecordIteratorError<'a, T, D>>
    where
        I: IntoIterator<Item = Record<'a, T>>
    {
        use crate::differentiation::record_operations::are_exact_same_list;

        let mut history: Option<Option<&WengertList<T>>> = None;
        let mut error: Option<InvalidRecordIteratorError<'a, T, D>> = None;

        let numbers: Vec<(T, Index)> = iter.into_iter().map(|record| {
            match history {
                None => history = Some(record.history),
                Some(h) => {
                    if !are_exact_same_list(h, record.history) {
                        error = Some(InvalidRecordIteratorError::InconsistentHistory {
                            first: h,
                            later: record.history
                        });
                    }
                }
            }
            (record.number, record.index)
        }).collect();

        if let Some(error) = error {
            return Err(error);
        }

        if numbers.is_empty() {
            return Err(InvalidRecordIteratorError::Empty);
        }

        let numbers = TensorView::from(
            match Tensor::try_from(
                shape,
                numbers,
            ) {
                Ok(numbers) => numbers,
                Err(invalid_shape) => return Err(InvalidRecordIteratorError::Shape(invalid_shape))
            }
        );

        // We already checked if the iterator was empty so `history` is always `Some` here
        Ok(RecordTensor::from_existing(history.unwrap(), numbers))
    }
}
