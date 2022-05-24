use crate::tensors::views::TensorRef;
use crate::tensors::{Dimension, InvalidShapeError};
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct TensorMask<T, S, const D: usize, const M: usize> {
    _source: S,
    _mask: [(Dimension, usize); M],
    _type: PhantomData<T>,
}

impl<T, S, const D: usize, const M: usize> TensorMask<T, S, D, M>
where
    S: TensorRef<T, D>,
{
    fn from(
        _source: S,
        _masks: [(Dimension, usize); M],
    ) -> Result<TensorMask<T, S, D, M>, InvalidShapeError<D>> {
        // TODO: How do we mask out individual indexes into a shape in a way that we can check
        // the shape is still valid with InvalidShapeError and still efficiently implement
        // TensorRef?
        // We'll need to know when to offset an index like MatrixRange but way more generalised.
        unimplemented!()
    }
}
