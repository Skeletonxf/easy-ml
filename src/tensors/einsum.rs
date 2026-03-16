/*!
 * Einstein summation notation
 *
 * TODO docs
 */

 use crate::numeric::{Numeric, NumericRef};
 use crate::tensors::views::{TensorRename, TensorRef, TensorView};
 use crate::tensors::{Dimension, Tensor};

struct Einsum {
    // TODO
    // Maybe we make this not a zero size type and have it created with
    // the optimise implementation then convert all the of_X_with and of_X
    // methods to method calls?
    // Then something like Einsum::default().of_1((["a", "b"],), ["b", "a"], (&x,)) or
    // Einsum::default().of_2((["a", "b"], ["c", "b"]), ["a", "c"], (&x, &y)) for caller
    // Any kind of builder pattern here could massively help with not getting
    // lost in all the arrays and tuples. Final output dimensions could be easy
    // enough to have as as `.to([...])` style method call, just potentially need
    // 1 Einsum type per arity for prior call to take input arguments.
    // Einsum::default().of_2(["a", "b"], ["c", "b"], &x, &y).to(["a", "c"])
    // could be quite nice to use? Could easily add simplified helpers that
    // only take char for dimension names too then.
}

struct InconsistentDimensionLength {
    // TODO
}

// Return length of matching dimension in inputs, and error if the length of
// this output dimension is inconsistent in the input.
fn length_of(
    output_dimension: Dimension,
    inputs: &[&[(Dimension, usize)]]
) -> Result<usize, InconsistentDimensionLength> {
    unimplemented!()
}

fn tensor_with_name<T, I, S, const D: usize>(
    dimensions: [Dimension; D],
    tensor: I
) -> TensorView<T, TensorRename<T, S, D>, D>
where
    I: Into<TensorView<T, S, D>>,
    S: TensorRef<T, D>,
{
    let source: S = tensor.into().source();
    let with_names = TensorRename::from(source, dimensions);
    TensorView::from(with_names)
}

impl Einsum {
    fn of_1_with<T, S, I, const D: usize, const O: usize>(
        dimensions: ([Dimension; D],),
        output: [Dimension; O],
        tensors: (I,),
    ) -> Result<Tensor<T, O>, InconsistentDimensionLength>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        I: Into<TensorView<T, S, D>>,
        S: TensorRef<T, D>,
    {
        let input = (tensor_with_name(dimensions.0, tensors.0),);
        Einsum::of_1::<T, _, _, D, O>(input, output)
    }

    fn of_1<T, S, I, const D: usize, const O: usize>(
        input: (I,),
        output: [Dimension; O],
    ) -> Result<Tensor<T, O>, InconsistentDimensionLength>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        I: Into<TensorView<T, S, D>>,
        S: TensorRef<T, D>,
    {
        let first = input.0.into();
        let first_shape: &[(Dimension, usize)] = &first.shape();
        let input_shapes = &[first_shape];
        let mut output_shape = std::array::from_fn(|d| (output[d], 0));
        for d in 0..O {
            output_shape[d].1 = length_of(output_shape[d].0, input_shapes)?;
        }

        Ok(Tensor::empty(output_shape, T::zero()))
    }

    // TODO: Will need to generalise into macro somehow because realistically need to go to
    // at least 6 tensor inputs
    // Also ideally want a way to go higher with type erasure but not sure we
    // actually can type erase the const generics for the dimensionality?
    fn of_2_with<T, S1, S2, I1, I2, const D1: usize, const D2: usize, const O: usize>(
        dimensions: ([Dimension; D1], [Dimension; D2]),
        output: [Dimension; O],
        tensors: (I1, I2),
    ) -> Result<Tensor<T, O>, InconsistentDimensionLength>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        I1: Into<TensorView<T, S1, D1>>,
        I2: Into<TensorView<T, S2, D2>>,
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
    {
        let input = (tensor_with_name(dimensions.0, tensors.0), tensor_with_name(dimensions.1, tensors.1));
        Einsum::of_2::<T, _, _, _, _, D1, D2, O>(input, output)
    }

    fn of_2<T, S1, S2, I1, I2, const D1: usize, const D2: usize, const O: usize>(
        input: (I1, I2),
        output: [Dimension; O],
    ) -> Result<Tensor<T, O>, InconsistentDimensionLength>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        I1: Into<TensorView<T, S1, D1>>,
        I2: Into<TensorView<T, S2, D2>>,
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
    {
        let first = input.0.into();
        let second = input.1.into();
        let first_shape: &[(Dimension, usize)] = &first.shape();
        let second_shape: &[(Dimension, usize)] = &second.shape();
        let input_shapes = &[first_shape, second_shape];
        let mut output_shape = std::array::from_fn(|d| (output[d], 0));
        for d in 0..O {
            output_shape[d].1 = length_of(output_shape[d].0, input_shapes)?;
        }

        Ok(Tensor::empty(output_shape, T::zero()))
    }
}
