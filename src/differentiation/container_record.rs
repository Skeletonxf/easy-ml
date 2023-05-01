//use crate::differentiation::Record;
use crate::numeric::{Numeric, NumericRef};
use crate::differentiation::{Primitive, Index, WengertList};
use crate::tensors::Tensor;
use crate::tensors::views::TensorMut;
use crate::tensors::indexing::TensorReferenceMutIterator;
use crate::matrices::Matrix;
use crate::matrices::iterators::RowMajorReferenceMutIterator;
use crate::matrices::views::{MatrixMut, NoInteriorMutability};

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
pub type RecordMatrix<'a, T> = RecordContainer<'a, T, Matrix<T>, 2>;

/**
 * Alias for succinctly refering to RecordContainers backed by a tensor.
 */
pub type RecordTensor<'a, T, const D: usize> = RecordContainer<'a, T, Tensor<T, D>, D>;

fn calculate_incrementing_indexes(starting_index: usize, total: usize) -> Vec<Index> {
    let mut indexes = vec![0; total];
    for i in 0..total {
        indexes[i] = starting_index + i;
    }
    indexes
}

impl<'a, T, const D: usize> RecordTensor<'a, T, D>
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
    pub fn constants<S>(mut c: S) -> Self
    where
        S: TensorMut<T, D>,
    {
        let total = crate::tensors::dimensions::elements(&c.view_shape());
        RecordContainer {
            indexes: vec![0; total],
            numbers: Tensor::from(
                c.view_shape(),
                // FIXME: Should generalise this as an owned TensorOwnedIterator
                TensorReferenceMutIterator::from(&mut c)
                    .map(|x: &mut T| std::mem::replace(x, T::zero()))
                    .collect()
            ),
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
    // TODO: Maybe swap parameter order here?
    pub fn variables<S>(mut x: S, history: &'a WengertList<T>) -> Self
    where
        S: TensorMut<T, D>,
    {
        let total = crate::tensors::dimensions::elements(&x.view_shape());
        let starting_index = history.append_nullary_repeating(total);
        RecordContainer {
            numbers: Tensor::from(
                x.view_shape(),
                // FIXME: Should generalise this as an owned TensorOwnedIterator
                TensorReferenceMutIterator::from(&mut x)
                    .map(|x: &mut T| std::mem::replace(x, T::zero()))
                    .collect()
            ),
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
        crate::tensors::dimensions::elements(&self.numbers.shape())
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


impl<'a, T> RecordMatrix<'a, T>
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
    pub fn constants<S>(mut c: S) -> Self
    where
        S: MatrixMut<T> + NoInteriorMutability,
    {
        RecordContainer {
            indexes: vec![0; c.view_rows() * c.view_columns()],
            numbers: Matrix::from_flat_row_major(
                (c.view_rows(), c.view_columns()),
                // FIXME: Should generalise this as an owned RowMajorOwnedIterator
                RowMajorReferenceMutIterator::from(&mut c)
                    .map(|x: &mut T| std::mem::replace(x, T::zero()))
                    .collect()
            ),
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
    pub fn variables<S>(mut x: S, history: &'a WengertList<T>) -> Self
    where
        S: MatrixMut<T> + NoInteriorMutability,
    {
        let total = x.view_rows() * x.view_columns();
        let starting_index = history.append_nullary_repeating(total);
        RecordContainer {
            numbers: Matrix::from_flat_row_major(
                (x.view_rows(), x.view_columns()),
                // FIXME: Should generalise this as an owned RowMajorOwnedIterator
                RowMajorReferenceMutIterator::from(&mut x)
                    .map(|x: &mut T| std::mem::replace(x, T::zero()))
                    .collect()
            ),
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
        self.numbers.rows() * self.numbers.columns()
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

impl<'a, T, const D: usize> RecordTensor<'a, T, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
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
     *     Tensor::from_fn(
     *         [("rows", 2), ("columns", 2)],
     *         |[r, c]| 0.15 * ((1 + r + c) as f32)
     *     ),
     *     &list
     * );
     * // the derivative of tanh(x) is sech(x) * sech(x) which is equivalent to
     * // 1 / (cosh(x) * cosh(x))
     * let Y = X.unary(|x| x.tanh(), |x| 1.0 / (x.cosh() * x.cosh()));
     * // TODO Inspecting derivatives
     * ```
     */
    pub fn unary(
        &self,
        fx: impl Fn(T) -> T,
        dfx_dx: impl Fn(T) -> T
    ) -> RecordTensor<'a, T, D> {
        match self.history {
            None => RecordTensor::constants(self.numbers.map(fx)),
            Some(history) => {
                let total = self.elements();
                assert_eq!(
                    total,
                    self.indexes.len(),
                    "Unexpected illegal state, number of elements should always match number of indexes"
                );
                let mut indexes = vec![0; total];
                let mut ys = vec![T::zero(); total];
                history.borrow(|history| {
                    // shadow the name so we can't accidentally try to use history while holding
                    // the borrow
                    // use enumerate not with_index because we need the 1D index for indexing
                    // self.indexes
                    for (i, (x, &parent)) in (self.numbers.iter().zip(&self.indexes)).enumerate() {
                        ys[i] = fx(x.clone());
                        let derivative = dfx_dx(x);
                        indexes[i] = history.append_unary(parent, derivative);
                    }
                }); // drop borrow on history
                RecordContainer {
                    // TODO: Consider using direct_from here to avoid recalculating the strides/shape
                    numbers: Tensor::from(self.numbers.shape(), ys),
                    history: Some(history),
                    indexes,
                }
            },
        }
    }
}