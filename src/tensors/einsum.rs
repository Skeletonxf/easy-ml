/*!
 * Einstein summation notation
 *
 * TODO docs
 */

use crate::numeric::{Numeric, NumericRef};
use crate::tensors::views::{TensorRename, TensorRef, TensorView};
use crate::tensors::{Dimension, Tensor};

use std::error::Error;
use std::fmt;

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

impl Einsum {
    /**
     * Returns an Einsum that will naively calculate the notation
     * without introducing any substeps.
     */
    fn naive() -> Self {
        Einsum {}
    }

    /**
     * Returns the default Einsum optimisation (currently naive).
     */
    fn default() -> Self {
        Einsum {}
    }

    fn with_1<T, S, I, const D: usize>(
        input_1: I,
    ) -> Einsum1<T, S, D>
    where
        S: TensorRef<T, D>,
        I: Into<TensorView<T, S, D>>,
    {
        Einsum1 { tensor_1: input_1.into() }
    }

    fn with_2<T, S1, S2, I1, I2, const D1: usize, const D2: usize>(
        input_1: I1,
        input_2: I2,
    ) -> Einsum2<T, S1, S2, D1, D2>
    where
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
        I1: Into<TensorView<T, S1, D1>>,
        I2: Into<TensorView<T, S2, D2>>,
    {
        Einsum2 { tensor_1: input_1.into(), tensor_2: input_2.into() }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InconsistentDimensionLengthError<const I: usize> {
    pub lengths: [Option<usize>; I],
    pub dimension: Dimension,
}

impl<const I: usize> fmt::Display for InconsistentDimensionLengthError<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "inconsistent dimension lengths for dimension '{}': {:?}, lengths must match when repeated in different shapes as the same dimension name",
            self.dimension,
            self.lengths,
        )
    }
}

impl<const I: usize> Error for InconsistentDimensionLengthError<I> {}

#[test]
fn test_inconsistent_dimension_length_error() {
    let error = InconsistentDimensionLengthError {
        lengths: [Some(3), None, Some(2)],
        dimension: "a",
    };
    assert_eq!(
        error.to_string(),
        "inconsistent dimension lengths for dimension 'a': [Some(3), None, Some(2)], lengths must match when repeated in different shapes as the same dimension name",
    )
}

/// Return length of matching dimension in inputs, and error if the length of
/// this output dimension is inconsistent in the input.
fn length_of<const I: usize>(
    output_dimension: Dimension,
    input: &[&[(Dimension, usize)]; I],
) -> Result<usize, InconsistentDimensionLengthError<I>> {
    let lengths = input.map(|shapes| {
        shapes
            .iter()
            .find(|(dimension, _length)| *dimension == output_dimension)
            .map(|(_dimension, length)| *length)
    });

    let first_length = lengths.iter().filter_map(|l| *l).next();
    if let Some(length) = first_length {
        // Check other lengths agree
        if lengths.iter().any(|l| l.is_some() && *l != Some(length)) {
            // Different length matches
            return Err(InconsistentDimensionLengthError {
                lengths: lengths,
                dimension: output_dimension,
            });
        } else {
            return Ok(length);
        }
    } else {
        // No matching lengths, we needed 1 match
        return Err(InconsistentDimensionLengthError {
            lengths,
            dimension: output_dimension,
        });
    }
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

/// Return required output shape given input shape sizes and output shape
/// dimension names. This can fail if a dimension in the requested output
/// shape isn't present in the input, or if the input has contradictory sizes
/// for it.
// We could validate some parts of the input earlier than when we have
// the output dimensions, but validating tensor lengths are consistent for
// each common input dimension name would happen multiple times in the scenario
// of a user using the `named` helper method, so it's a lot easier to use the API
// if we defer validation till the final method call.
fn output_shape_for<const I: usize, const O: usize>(
    input: &[&[(Dimension, usize)]; I],
    output: &[Dimension; O],
) -> Result<[(Dimension, usize); O], InconsistentDimensionLengthError<I>> {
    let mut output_shape = std::array::from_fn(|d| (output[d], 0));
    for d in 0..O {
        output_shape[d].1 = length_of(output_shape[d].0, input)?;
    }
    Ok(output_shape)
}

/// We sum over every dimension included in the input and not the output
///
/// Returns a vec of the summation dimensions along with their validated
/// lengths, and errors if the lengths of any summation dimensions in
/// the input are inconsistent.
fn summation_dimensions<const I: usize, const O: usize>(
    input: &[&[(Dimension, usize)]; I],
    output: &[Dimension; O],
) -> Result<Vec<(Dimension, usize)>, InconsistentDimensionLengthError<I>> {
    let mut total_dimensions = 0;
    for shape in input {
        total_dimensions += shape.len();
    }

    // Worst case is every dimension in each input tensor
    // has unique dimensions
    let mut unique_dimensions = Vec::with_capacity(total_dimensions);

    for shape in input {
        for (dimension, length) in shape.iter() {
            if output.contains(dimension) {
                // If this dimension is requested in the output we will be checking
                // for consistent lengths in `length_of` and this dimension won't be
                // a summation dimension so we can ignore it here.
                continue;
            }
            let existing = unique_dimensions.iter().find(|(d, _)| d == dimension);
            match existing {
                None => unique_dimensions.push((*dimension, *length)),
                Some((_, l)) => {
                    if length != l {
                        // Inconsistent lengths
                        return Err(InconsistentDimensionLengthError {
                            lengths: std::array::from_fn(|i| {
                                input[i].iter().find(|(d, _)| d == dimension).map(|(_, l)| *l)
                            }),
                            dimension,
                        })
                    }
                }
            }
        }
    }

    return Ok(unique_dimensions);
}

struct Einsum1<T, S1, const D1: usize> {
    tensor_1: TensorView<T, S1, D1>,
}

struct Einsum2<T, S1, S2, const D1: usize, const D2: usize> {
    tensor_1: TensorView<T, S1, D1>,
    tensor_2: TensorView<T, S2, D2>,
}

impl<T, S1, const D1: usize> Einsum1<T, S1, D1> {
    fn named(self, input_1: [Dimension; D1]) -> Einsum1<T, TensorRename<T, S1, D1>, D1>
    where
        S1: TensorRef<T, D1>,
    {
        Einsum1 {
            tensor_1: tensor_with_name(input_1, self.tensor_1)
        }
    }

    fn to<const O: usize>(
        self,
        output: [Dimension; O],
    ) -> Result<Tensor<T, O>, InconsistentDimensionLengthError<1>>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        S1: TensorRef<T, D1>,
    {
        let input_1_shape: &[(Dimension, usize)] = &self.tensor_1.shape();
        let input = &[input_1_shape];
        let output_shape = output_shape_for(input, &output)?;
        let mut output_tensor = Tensor::empty(output_shape, T::zero());
        let _summation_dimensions = summation_dimensions(input, &output)?;
        for (_indexes, _element) in output_tensor.index_mut().iter_reference_mut().with_index() {
            let mut _sum  = T::zero();
            // TODO We should be summing the products of each fully indexed input here
            // There will be as many inner loops as dimensions in the input not
            // specified in the output, but even if that number is zero, we would
            // do something here like *element += input1[i,j] * input2[j,k] once
        }

        Ok(output_tensor)
    }
}

impl<T, S1, S2, const D1: usize, const D2: usize> Einsum2<T, S1, S2, D1, D2> {
    fn named(
        self,
        input_1: [Dimension; D1],
        input_2: [Dimension; D2],
    ) -> Einsum2<T, TensorRename<T, S1, D1>, TensorRename<T, S2, D2>, D1, D2>
    where
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
    {
        Einsum2 {
            tensor_1: tensor_with_name(input_1, self.tensor_1),
            tensor_2: tensor_with_name(input_2, self.tensor_2),
        }
    }

    fn to<const O: usize>(
        self,
        output: [Dimension; O],
    ) -> Result<Tensor<T, O>, InconsistentDimensionLengthError<2>>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
    {
        let input_1_shape: &[(Dimension, usize)] = &self.tensor_1.shape();
        let input_2_shape: &[(Dimension, usize)] = &self.tensor_2.shape();
        let input = &[input_1_shape, input_2_shape];
        let output_shape = output_shape_for(input, &output)?;
        let mut output_tensor = Tensor::empty(output_shape, T::zero());
        let _summation_dimensions = summation_dimensions(input, &output)?;
        for (_indexes, _element) in output_tensor.index_mut().iter_reference_mut().with_index() {
            let mut _sum  = T::zero();
            // TODO We should be summing the products of each fully indexed input here
            // There will be as many inner loops as dimensions in the input not
            // specified in the output, but even if that number is zero, we would
            // do something here like *element += input1[i,j] * input2[j,k] once
        }

        Ok(output_tensor)
    }
}

// TODO: Once Tensor implementation is working, should be able to actually generalise
// to work on RecordTensor inputs too, they can be passed in up to the .to() step already.
// Final step needs to be aware of some kind of NumericLike type that knows how to lift
// and lower from additional context to a Numeric type for addition and multiplication.
// In some future work can introduce a generic associated type for TensorRef that
// 'knows' what the desired container output is for tensor operations like these to collect
// the results back into, so result type becomes Result<S1::Output<T, O>, InconsistentDimensionLength>
// and somehow we enforce S1::Output == S2::Output????
