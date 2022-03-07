use crate::tensors::{Dimension, Tensor};
use crate::tensors::indexing::TensorAccess;
use crate::tensors::views::TensorRef;

pub(crate) fn transpose<T, S, const D: usize>(
    tensor: S,
    dimensions: [Dimension; D]
) -> Tensor<T, D>
where
    T: Clone,
    S: TensorRef<T, D>,
{
    let source_shape = tensor.view_shape();
    // TODO: Handle error case, propagate as Dimension names to transpose to must be the same set of dimension names in the tensor
    let transposed_order = TensorAccess::from(tensor, dimensions);
    let transposed_shape = transposed_order.shape();
    let dummy = transposed_order.get_reference([0; D]).clone();

    let mut transposed = Tensor::from(
        transposed_shape,
        vec![dummy; crate::tensors::dimensions::elements(&source_shape)]
    );

    let mut transposed_elements = transposed_order.index_order_reference_iter();
    for elem in transposed.data.iter_mut() {
        *elem = transposed_elements.next().unwrap().clone();
    }

    transposed
}
