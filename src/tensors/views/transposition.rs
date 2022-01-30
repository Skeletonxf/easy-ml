use crate::tensors::{Dimension, Tensor, elements};
use crate::tensors::indexing::TensorAccess;
use crate::tensors::views::{TensorRef, TensorMut};

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
        vec![dummy; elements(&source_shape)]
    );

    let mut transposed_elements = transposed_order.index_reference_iter();
    for elem in transposed.data.iter_mut() {
        *elem = transposed_elements.next().unwrap().clone();
    }

    transposed
}

/**
 * Returns true if the dimensions are all the same length. For 0 or 1 dimensions trivially returns
 * true. For 2 dimensions, this corresponds to a square matrix, and for 3 dimensions, a cube shaped
 * tensor.
 */
fn is_square<const D: usize>(dimensions: &[(Dimension, usize); D]) -> bool {
    if D > 1 {
        let first = dimensions[0].1;
        for d in 1..D {
            if dimensions[d].1 != first {
                return false;
            }
        }
        true
    } else {
        true
    }
}

pub(crate) fn transpose_mut<T, S, const D: usize>(
    mut tensor: S,
    dimensions: [Dimension; D]
)
where
    T: Clone,
    S: TensorMut<T, D>,
{
    use crate::tensors::indexing::{dimension_mapping, dimension_mapping_shape, map_dimensions};
    let source_shape = tensor.view_shape();
    if D == 2 && is_square(&source_shape) {
        // TODO: Handle error case, propagate as Dimension names to transpose to must be the same set of dimension names in the tensor
        let dimension_mapping = dimension_mapping(&source_shape, &dimensions).unwrap();

        // Don't actually create an iterator because we need to retain ownership of the tensor
        // data so we can transpose it while iterating.
        let mut indexes = [0; D];
        let mut finished = tensor.get_reference(map_dimensions(&dimension_mapping, &indexes)).is_some();
        let shape = dimension_mapping_shape(&source_shape, &dimension_mapping);

        while !finished {
            let index = indexes;
            let i = index[0];
            let j = index[1];
            if j >= i {
                let mapped_index = map_dimensions(&dimension_mapping, &index);
                // Swap elements from the upper triangle (using index order of the actual tensor's
                // shape)
                let temp = tensor.get_reference(index).unwrap().clone();
                // tensor[i,j] becomes tensor[mapping(i,j)]
                *tensor.get_reference_mut(index).unwrap() = tensor
                    .get_reference(mapped_index).unwrap()
                    .clone();
                // tensor[mapping(i,j)] becomes tensor[i,j]
                *tensor.get_reference_mut(mapped_index).unwrap() = temp;
                // If the mapping is a noop we've assigned i,j to i,j
                // If the mapping is i,j -> j,i we've assigned i,j to j,i and j,i to i,j
            }
            crate::tensors::indexing::index_order_iter(&mut finished, &mut indexes, &shape);
        }
    } else {
        unimplemented!()
    }
}
