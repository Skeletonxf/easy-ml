//use crate::differentiation::Record;
use crate::numeric::Numeric;
use crate::differentiation::{Primitive, Index, WengertList};
use crate::tensors::views::{TensorView, TensorRef};
use crate::matrices::views::{MatrixView, MatrixRef, NoInteriorMutability};

/**
 * A pluralisation of [Record](crate::differentiation::Record) that groups together a
 * **s**ource of numbers instead of storing one number of type T individually.
 *
 * Typically you would refer to one of the type aliases to disambiguate the type of `S` and
 * use more succinct generics: [RecordMatrix](RecordMatrix), [RecordTensor](RecordTensor).
 */
#[derive(Debug)]
pub struct RecordContainer<'a, T: Primitive, S, const D: usize> {
    numbers: S,
    history: Option<&'a WengertList<T>>,
    // it should be possible to store only the first index, most operations will create all the
    // entries on the Wengert list contiguously. Only matrix multiplication is problematic.
    // We can contiguise arbitrary entries by adding 0 to all of them, however this will
    // complicate the implementation somewhat and require additional additions/multiplications in
    // favor of less memory usage so I'm not going to do this for the first iteration as I'm
    // not confident it would improve performance.
    indexes: Vec<Index>,
}

/**
 * Alias for succinctly refering to RecordContainers backed by a matrix.
 */
pub type RecordMatrix<'a, T, S> = RecordContainer<'a, T, MatrixView<T, S>, 2>;

/**
 * Alias for succinctly refering to RecordContainers backed by a tensor.
 */
pub type RecordTensor<'a, T, S, const D: usize> = RecordContainer<'a, T, TensorView<T, S, D>, D>;

fn calculate_incrementing_indexes(starting_index: usize, total: usize) -> Vec<Index> {
    let mut indexes = vec![0; total];
    for i in 0..total {
        indexes[i] = starting_index + i;
    }
    indexes
}

impl<'a, T, S, const D: usize> RecordTensor<'a, T, S, D>
where
    T: Numeric + Primitive,
    S: TensorRef<T, D>,
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
    pub fn constants(c: S) -> Self {
        RecordContainer {
            indexes: vec![0; RecordContainer::total(&c)],
            numbers: TensorView::from(c),
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
     *         Tensor::from([("r", 2), ("c", 2)], vec![ 1.0, 2.0, 3.0, 4.0 ]),
     *         &list
     *     )
     * }; // list no longer in scope
     * ```
     */
    pub fn variables(x: S, history: &'a WengertList<T>) -> Self {
        let total = RecordContainer::total(&x);
        let starting_index = history.append_nullary_repeating(total);
        RecordContainer {
            numbers: TensorView::from(x),
            history: Some(history),
            indexes: calculate_incrementing_indexes(starting_index, total),
        }
    }

    /**
     * Returns the number of elements stored by this container's source.
     *
     * For a 2 x 3 Tensor, this would return 6, and for a 2 x 3 x 4 Tensor this would return 24
     * and so on.
     *
     * see also [dimensions::elements](crate::tensors::dimensions::elements)
     */
    pub fn elements(&self) -> usize {
        RecordContainer::total(self.numbers.source_ref())
    }

    fn total(numbers: &S) -> usize {
        crate::tensors::dimensions::elements(&numbers.view_shape())
    }

    /**
     * Resets all of the records to place them back on the WengertList, for use
     * in performing another derivation after clearing the WengertList.
     */
    pub fn reset(&mut self) {
        match self.history {
            None => (), // noop
            Some(history) => self.indexes = {
                let total = self.elements();
                let starting_index = history.append_nullary_repeating(total);
                calculate_incrementing_indexes(starting_index, total)
            },
        };
    }

    /**
     * A convenience helper function which takes a RecordContainer by value and
     * calls [reset](RecordContainer::reset()) on it.
     */
    pub fn do_reset(mut x: Self) -> Self {
        x.reset();
        x
    }
}


impl<'a, T, S> RecordMatrix<'a, T, S>
where
    T: Numeric + Primitive,
    S: MatrixRef<T> + NoInteriorMutability,
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
    pub fn constants(c: S) -> Self {
        RecordContainer {
            indexes: vec![0; RecordContainer::size(&c)],
            numbers: MatrixView::from(c),
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
     *         Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
     *         &list
     *     )
     * }; // list no longer in scope
     * ```
     */
    pub fn variables(x: S, history: &'a WengertList<T>) -> Self {
        let total = RecordContainer::size(&x);
        let starting_index = history.append_nullary_repeating(total);
        RecordContainer {
            numbers: MatrixView::from(x),
            history: Some(history),
            indexes: calculate_incrementing_indexes(starting_index, total),
        }
    }

    /**
     * Returns the number of elements stored by this container's source.
     *
     * For a 2 x 3 Matrix, this would return 6, and for a 3 x 4 Matrix this would return 12
     * and so on.
     */
    pub fn elements(&self) -> usize {
        RecordContainer::size(self.numbers.source_ref())
    }

    fn size(numbers: &S) -> usize {
        numbers.view_rows() * numbers.view_columns()
    }

    /**
     * Resets all of the records to place them back on the WengertList, for use
     * in performing another derivation after clearing the WengertList.
     */
    pub fn reset(&mut self) {
        match self.history {
            None => (), // noop
            Some(history) => self.indexes = {
                let total = self.elements();
                let starting_index = history.append_nullary_repeating(total);
                calculate_incrementing_indexes(starting_index, total)
            },
        };
    }

    /**
     * A convenience helper function which takes a RecordContainer by value and
     * calls [reset](RecordContainer::reset()) on it.
     */
    pub fn do_reset(mut x: Self) -> Self {
        x.reset();
        x
    }
}
