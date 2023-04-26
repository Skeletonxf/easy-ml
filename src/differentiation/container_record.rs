//use crate::differentiation::Record;
use crate::numeric::Numeric;
use crate::differentiation::{Primitive, Index, WengertList};
use crate::tensors::views::TensorRef;

// TODO: Doc everthing once sure this approach will actually work

/**
 * A pluralisation of [Record](Record) that groups together a **s**ource of numbers instead
 * of storing one number of type T individually.
 */
#[derive(Debug)]
pub struct RecordContainer<'a, T: Primitive, S, const D: usize> {
    numbers: S,
    history: Option<&'a WengertList<T>>,
    indexes: Vec<Index>,
}

impl<'a, T, S, const D: usize> RecordContainer<'a, T, S, D>
where
    T: Numeric + Primitive,
    S: TensorRef<T, D>,
{
    pub fn constants(c: S) -> Self {
        RecordContainer {
            indexes: vec![0; RecordContainer::total(&c)],
            numbers: c,
            history: None,
        }
    }

    /**
     * Returns the number of elements stored by this container's source.
     *
     * For a 2 x 3 Tensor, this would return 6, and for a 2 x 3 x 4 Tensor this would return 24
     * and so on.
     *
     * see [dimensions::elements](crate::tensors::dimensions::elements)
     */
    pub fn elements(&self) -> usize {
        RecordContainer::total(&self.numbers)
    }

    fn total(numbers: &S) -> usize {
        crate::tensors::dimensions::elements(&numbers.view_shape())
    }

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

    pub fn do_reset(mut x: Self) -> Self {
        x.reset();
        x
    }
}
