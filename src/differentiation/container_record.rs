//use crate::differentiation::Record;
use crate::numeric::Numeric;
use crate::differentiation::{Primitive, Index, WengertList};
use crate::interop::{TensorRefMatrix, DimensionNames, RowAndColumn};
use crate::tensors::InvalidShapeError;
use crate::tensors::views::TensorRef;
use crate::matrices::views::{MatrixRef, NoInteriorMutability};

/**
 * A pluralisation of [Record](crate::differentiation::Record) that groups together a
 * **s**ource of numbers instead of storing one number of type T individually.
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

impl<'a, T, S, const D: usize> RecordContainer<'a, T, S, D>
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
            numbers: c,
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
     * use easy_ml::differentiation::RecordContainer;
     * use easy_ml::differentiation::WengertList;
     * use easy_ml::tensors::Tensor;
     * let record = {
     *     let list = WengertList::new();
     *     RecordContainer::variables(
     *         Tensor::from([("r", 2), ("c", 2)], vec![ 1.0, 2.0, 3.0, 4.0 ]),
     *         &list
     *     )
     * }; // list no longer in scope
     * ```
     */
    pub fn variables(x: S, history: &'a WengertList<T>) -> Self {
        let total = RecordContainer::total(&x);
        let starting_index = history.append_nullary_repeating(total);
        let mut indexes = vec![0; total];
        for i in 0..total {
            indexes[i] = starting_index + i;
        }
        RecordContainer {
            numbers: x,
            history: Some(history),
            indexes,
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
        RecordContainer::total(&self.numbers)
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
                let mut indexes = vec![0; total];
                for i in 0..total {
                    indexes[i] = starting_index + i;
                }
                indexes
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


/**
 * Alias for succinctly refering to RecordContainers backed by a matrix.
 */
pub type RecordMatrixContainer<'a, T, S> = RecordContainer<'a, T, TensorRefMatrix<T, S, RowAndColumn>, 2>;

/**
 * Convenience helper functions for creating a RecordContainer from a matrix with defaulted
 * shape names.
 */
impl<'a, T, S> RecordMatrixContainer<'a, T, S>
where
    T: Numeric + Primitive,
    S: MatrixRef<T> + NoInteriorMutability,
{
    /**
     * `Err` variant is returned as documented on [TensorRefMatrix](TensorRefMatrix::from)
     */
    pub fn matrix_constants(c: S) -> Result<Self, InvalidShapeError<2>> {
        RecordContainer::matrix_with_names_constants(c, RowAndColumn)
    }

    /**
     * `Err` variant is returned as documented on [TensorRefMatrix](TensorRefMatrix::from)
     */
    pub fn matrix_variables(
        x: S, history: &'a WengertList<T>
    ) -> Result<Self, InvalidShapeError<2>> {
        RecordContainer::matrix_with_names_variables(x, RowAndColumn, history)
    }
}

/**
 * Convenience helper functions for creating a RecordContainer from a matrix.
 */
impl<'a, T, S, N> RecordContainer<'a, T, TensorRefMatrix<T, S, N>, 2>
where
    T: Numeric + Primitive,
    S: MatrixRef<T> + NoInteriorMutability,
    N: DimensionNames,
{
    /**
     * `Err` variant is returned as documented on [TensorRefMatrix](TensorRefMatrix::with_names)
     */
    pub fn matrix_with_names_constants(c: S, names: N) -> Result<Self, InvalidShapeError<2>> {
        TensorRefMatrix::with_names(c, names).map(|tensor| RecordContainer::constants(tensor))
    }

    /**
     * `Err` variant is returned as documented on [TensorRefMatrix](TensorRefMatrix::with_names)
     */
    pub fn matrix_with_names_variables(
        x: S,
        names: N,
        history: &'a WengertList<T>
    ) -> Result<Self, InvalidShapeError<2>> {
        TensorRefMatrix::with_names(x, names)
            .map(|tensor| RecordContainer::variables(tensor, history))
    }
}

// TODO: Need helper conversion methods for going from RecordContainer Tensor back to Matrix
// otherwise being able to start with matrices is a bit pointless because you don't end up with
// them after doing any operations.
