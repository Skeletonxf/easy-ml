/*!
 * Einstein summation notation
 *
 * TODO docs
 */

use crate::numeric::{Numeric, NumericRef};
use crate::tensors::views::{TensorRename, TensorRef, TensorView};
use crate::tensors::indexing::DynamicShapeIterator;
use crate::tensors::{Dimension, Tensor};

use std::error::Error;
use std::fmt;

pub struct Einsum {
    // TODO: Include optimiser in here
}

/**
 * Einstein summation notation
 *
 * A very general purpose sum of products that can represent many
 * different tensor operations with a single notation. In Easy-ML,
 * as tensors are already named, their dimension names are used instead
 * of arbitary characters to refer to dimensions across inputs and
 * the output. Whereas the typical notation used in python libraries
 * is of the form `ab,bc->ac` or `ab->` these would be
 * `Einsum::default().with_2(&i, &j).to(["a", "c"])` or
 * `Einsum::default().with_1(&i).to([])` respectively. In scenarios
 * where the existing dimension names in a tensor aren't what you
 * need for the summation notation, there are `named` helper methods
 * to provide an override, so you can perform `ab,bc->ac` with
 * input tensors of different dimension names if you write
 * `Einsum::default().with_2(&i, &j).named(["a", "b"], ["b", "c"]).to(["a", "c"])`.
 *
 * See also
 * - [Einsum is All you Need - Einstein Summation in Deep Learning](https://rockt.ai/2018/04/30/einsum)
 * - [Einsum Is All You Need (Video)](https://www.youtube.com/watch?v=pkVwUVEHmfI)
 */
impl Einsum {
    /**
     * Returns an Einsum that will naively calculate the notation
     * in a single pass without introducing any substeps.
     */
    pub fn naive() -> Self {
        Einsum {}
    }

    /**
     * Returns the default Einsum optimisation (currently naive).
     */
    fn default() -> Self {
        Einsum {}
    }

    // TODO: Allow passing in a type that picks the desired contraction order,
    // only need to return a vec of differently sized arrays/vecs of usize
    // Consuming code puts inputs onto a list, runs Einsum::naive on selected
    // inputs from first entry in list, puts output onto end of inputs list
    // and loops till finished. Maybe a vec of enums so we can iterate over
    // each possible size of a contraction since we need to know at compile
    // time how many inputs we're using?
    // How do we know what the intermediate output names are though?
    // https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html

    /**
     * An operation with a single input tensor.
     */
    pub fn with_1<T, S, I, const D: usize>(
        self,
        input_1: I,
    ) -> Einsum1<T, S, D>
    where
        S: TensorRef<T, D>,
        I: Into<TensorView<T, S, D>>,
    {
        Einsum1 { tensor_1: input_1.into() }
    }

    /**
     * An operation with two input tensors.
     */
    pub fn with_2<T, S1, S2, I1, I2, const D1: usize, const D2: usize>(
        self,
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

    /**
     * An operation with three input tensors.
     */
    pub fn with_3<T, S1, S2, S3, I1, I2, I3, const D1: usize, const D2: usize, const D3: usize>(
        self,
        input_1: I1,
        input_2: I2,
        input_3: I3,
    ) -> Einsum3<T, S1, S2, S3, D1, D2, D3>
    where
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
        S3: TensorRef<T, D3>,
        I1: Into<TensorView<T, S1, D1>>,
        I2: Into<TensorView<T, S2, D2>>,
        I3: Into<TensorView<T, S3, D3>>,
    {
        Einsum3 {
            tensor_1: input_1.into(),
            tensor_2: input_2.into(),
            tensor_3: input_3.into(),
        }
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

#[track_caller]
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

/// Filters outer indexes to only the matching dimensions
/// for the input shape. Panics if any dimensions in the
/// input shape are missing from the outer slices, but accepts
/// more indexes and dimensions in the outer slices than
/// actually needed for the input shape without any errors.
fn filter_outer_indexes<const D: usize, const O: usize>(
    outer_indexes: &[usize; O],
    outer_shape: &[(Dimension, usize); O],
    input_shape: &[(Dimension, usize); D],
) -> [usize; D] {
    let mut input_indexes = [0; D];
    for d in 0..D {
        let mut found = false;
        let dimension = input_shape[d].0;
        for o in 0..O {
            let possible_dimension = outer_shape[o].0;
            if possible_dimension == dimension {
                input_indexes[d] = outer_indexes[o];
                found = true;
                break;
            }
        }
        if !found {
            panic!(
                "Expected to find an index for dimension {:?} but was not present in {:?} for {:?} while trying to index tensor of shape {:?}",
                dimension,
                outer_indexes,
                outer_shape,
                input_shape,
            );
        }
    }
    input_indexes
}

/// Filters outer indexes and summation indexes to only the
/// matching dimensions for the input shape. Panics if any dimensions
/// in the input shape are missing from the outer and summation slices,
/// but accepts more indexes and dimensions in the outer and summation
/// slices than actually needed for the input shape without any errors.
/// Summation slices must be the same length, we just don't know their
/// length at compile time so can't enforce it in the type system.
fn filter_outer_and_summation_indexes<const D: usize, const O: usize>(
    outer_indexes: &[usize; O],
    outer_shape: &[(Dimension, usize); O],
    summation_indexes: &[usize],
    summation_shape: &[(Dimension, usize)],
    input_shape: &[(Dimension, usize); D],
) -> [usize; D] {
    let mut input_indexes = [0; D];
    for d in 0..D {
        let mut found = false;
        let dimension = input_shape[d].0;
        for o in 0..O {
            let possible_dimension = outer_shape[o].0;
            if possible_dimension == dimension {
                input_indexes[d] = outer_indexes[o];
                found = true;
                break;
            }
        }
        let summation_iter = summation_indexes.iter().zip(summation_shape.iter());
        for (index, (possible_dimension, _length)) in summation_iter {
            if *possible_dimension == dimension {
                input_indexes[d] = *index;
                found = true;
                break;
            }
        }
        if !found {
            panic!(
                "Expected to find an index for dimension {:?} but was not present in {:?} for {:?} or {:?} for {:?} while trying to index tensor of shape {:?}",
                dimension,
                outer_indexes,
                outer_shape,
                summation_indexes,
                summation_shape,
                input_shape,
            );
        }
    }
    input_indexes
}

/**
 * Einstein summation notation operation with a single input tensor.
 */
pub struct Einsum1<T, S1, const D1: usize> {
    tensor_1: TensorView<T, S1, D1>,
}

/**
 * Einstein summation notation operation with two input tensors.
 */
pub struct Einsum2<T, S1, S2, const D1: usize, const D2: usize> {
    tensor_1: TensorView<T, S1, D1>,
    tensor_2: TensorView<T, S2, D2>,
}

/**
 * Einstein summation notation operation with three input tensors
 */
pub struct Einsum3<T, S1, S2, S3, const D1: usize, const D2: usize, const D3: usize> {
    tensor_1: TensorView<T, S1, D1>,
    tensor_2: TensorView<T, S2, D2>,
    tensor_3: TensorView<T, S3, D3>,
}

impl<T, S1, const D1: usize> Einsum1<T, S1, D1> {
    /**
     * Renames all input tensors to the new names. Their shapes will
     * still be in the same order with the same lengths of data, as
     * per [TensorRename]. As per TensorRename, dimension names for
     * each individual tensor must be unique.
     */
    #[track_caller]
    pub fn named(self, input_1: [Dimension; D1]) -> Einsum1<T, TensorRename<T, S1, D1>, D1>
    where
        S1: TensorRef<T, D1>,
    {
        Einsum1 {
            tensor_1: tensor_with_name(input_1, self.tensor_1)
        }
    }

    pub fn to<const O: usize>(
        self,
        output: [Dimension; O],
    ) -> Result<Tensor<T, O>, InconsistentDimensionLengthError<1>>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        S1: TensorRef<T, D1>,
    {
        let input_1_shape_const = &self.tensor_1.shape();
        let input_1_shape: &[(Dimension, usize)] = input_1_shape_const;
        let input = &[input_1_shape];

        let output_shape = output_shape_for(input, &output)?;
        let mut output_tensor = Tensor::empty(output_shape, T::zero());

        let summation_dimensions = summation_dimensions(input, &output)?;
        let tensor_1_indexing = self.tensor_1.index();

        for (indexes, element) in output_tensor.index_mut().iter_reference_mut().with_index() {
            let mut sum = T::zero();

            if summation_dimensions.is_empty() {
                let product_1 = tensor_1_indexing
                    .get_ref(
                        filter_outer_indexes(
                            &indexes,
                            &output_shape,
                            &input_1_shape_const
                        )
                    );
                sum = sum + product_1;
            } else {
                let mut summation_iterator = DynamicShapeIterator::from(&summation_dimensions);
                loop {
                    let next = summation_iterator.next();
                    match next {
                        Some(summation_indexes) => {
                            let product_1 = tensor_1_indexing
                                .get_ref(
                                    filter_outer_and_summation_indexes(
                                        &indexes,
                                        &output_shape,
                                        &summation_indexes,
                                        &summation_dimensions,
                                        &input_1_shape_const
                                    )
                                );
                            sum = sum + product_1;
                        }
                        None => break
                    }
                }
            }
            *element = sum;
        }

        Ok(output_tensor)
    }
}

impl<T, S1, S2, const D1: usize, const D2: usize> Einsum2<T, S1, S2, D1, D2> {
    /**
     * Renames all input tensors to the new names. Their shapes will
     * still be in the same order with the same lengths of data, as
     * per [TensorRename]. As per TensorRename, dimension names for
     * each individual tensor must be unique.
     */
    #[track_caller]
    pub fn named(
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

    pub fn to<const O: usize>(
        self,
        output: [Dimension; O],
    ) -> Result<Tensor<T, O>, InconsistentDimensionLengthError<2>>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
    {
        let input_1_shape_const = &self.tensor_1.shape();
        let input_1_shape: &[(Dimension, usize)] = input_1_shape_const;
        let input_2_shape_const = &self.tensor_2.shape();
        let input_2_shape: &[(Dimension, usize)] = input_2_shape_const;
        let input = &[input_1_shape, input_2_shape];

        let output_shape = output_shape_for(input, &output)?;
        let mut output_tensor = Tensor::empty(output_shape, T::zero());

        let summation_dimensions = summation_dimensions(input, &output)?;
        let tensor_1_indexing = self.tensor_1.index();
        let tensor_2_indexing = self.tensor_2.index();

        for (indexes, element) in output_tensor.index_mut().iter_reference_mut().with_index() {
            let mut sum = T::zero();

            if summation_dimensions.is_empty() {
                let product_1 = tensor_1_indexing
                    .get_ref(
                        filter_outer_indexes(
                            &indexes,
                            &output_shape,
                            &input_1_shape_const
                        )
                    );
                let product_2 = tensor_2_indexing
                    .get_ref(
                        filter_outer_indexes(
                            &indexes,
                            &output_shape,
                            &input_2_shape_const
                        )
                    );
                sum = sum + (product_1 * product_2);
            } else {
                let mut summation_iterator = DynamicShapeIterator::from(&summation_dimensions);
                loop {
                    let next = summation_iterator.next();
                    match next {
                        Some(summation_indexes) => {
                            let product_1 = tensor_1_indexing
                                .get_ref(
                                    filter_outer_and_summation_indexes(
                                        &indexes,
                                        &output_shape,
                                        &summation_indexes,
                                        &summation_dimensions,
                                        &input_1_shape_const
                                    )
                                );
                            let product_2 = tensor_2_indexing
                                .get_ref(
                                    filter_outer_and_summation_indexes(
                                        &indexes,
                                        &output_shape,
                                        &summation_indexes,
                                        &summation_dimensions,
                                        &input_2_shape_const
                                    )
                                );
                            sum = sum + (product_1 * product_2);
                        }
                        None => break
                    }
                }
            }

            *element = sum;
        }

        Ok(output_tensor)
    }
}

impl<T, S1, S2, S3, const D1: usize, const D2: usize, const D3: usize> Einsum3<T, S1, S2, S3, D1, D2, D3> {
    /**
     * Renames all input tensors to the new names. Their shapes will
     * still be in the same order with the same lengths of data, as
     * per [TensorRename]. As per TensorRename, dimension names for
     * each individual tensor must be unique.
     */
    #[track_caller]
    pub fn named(
        self,
        input_1: [Dimension; D1],
        input_2: [Dimension; D2],
        input_3: [Dimension; D3],
    ) -> Einsum3<T, TensorRename<T, S1, D1>, TensorRename<T, S2, D2>, TensorRename<T, S3, D3>, D1, D2, D3>
    where
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
        S3: TensorRef<T, D3>,
    {
        Einsum3 {
            tensor_1: tensor_with_name(input_1, self.tensor_1),
            tensor_2: tensor_with_name(input_2, self.tensor_2),
            tensor_3: tensor_with_name(input_3, self.tensor_3),
        }
    }

    pub fn to<const O: usize>(
        self,
        output: [Dimension; O],
    ) -> Result<Tensor<T, O>, InconsistentDimensionLengthError<3>>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
        S3: TensorRef<T, D3>,
    {
        let input_1_shape_const = &self.tensor_1.shape();
        let input_1_shape: &[(Dimension, usize)] = input_1_shape_const;
        let input_2_shape_const = &self.tensor_2.shape();
        let input_2_shape: &[(Dimension, usize)] = input_2_shape_const;
        let input_3_shape_const = &self.tensor_3.shape();
        let input_3_shape: &[(Dimension, usize)] = input_3_shape_const;
        let input = &[input_1_shape, input_2_shape, input_3_shape];

        let output_shape = output_shape_for(input, &output)?;
        let mut output_tensor = Tensor::empty(output_shape, T::zero());

        let summation_dimensions = summation_dimensions(input, &output)?;
        let tensor_1_indexing = self.tensor_1.index();
        let tensor_2_indexing = self.tensor_2.index();
        let tensor_3_indexing = self.tensor_3.index();

        for (indexes, element) in output_tensor.index_mut().iter_reference_mut().with_index() {
            let mut sum = T::zero();

            if summation_dimensions.is_empty() {
                let product_1 = tensor_1_indexing
                    .get_ref(
                        filter_outer_indexes(
                            &indexes,
                            &output_shape,
                            &input_1_shape_const
                        )
                    );
                let product_2 = tensor_2_indexing
                    .get_ref(
                        filter_outer_indexes(
                            &indexes,
                            &output_shape,
                            &input_2_shape_const
                        )
                    );
                let product_3 = tensor_3_indexing
                    .get_ref(
                        filter_outer_indexes(
                            &indexes,
                            &output_shape,
                            &input_3_shape_const
                        )
                    );
                sum = sum + (product_1 * product_2 * product_3);
            } else {
                let mut summation_iterator = DynamicShapeIterator::from(&summation_dimensions);
                loop {
                    let next = summation_iterator.next();
                    match next {
                        Some(summation_indexes) => {
                            let product_1 = tensor_1_indexing
                                .get_ref(
                                    filter_outer_and_summation_indexes(
                                        &indexes,
                                        &output_shape,
                                        &summation_indexes,
                                        &summation_dimensions,
                                        &input_1_shape_const
                                    )
                                );
                            let product_2 = tensor_2_indexing
                                .get_ref(
                                    filter_outer_and_summation_indexes(
                                        &indexes,
                                        &output_shape,
                                        &summation_indexes,
                                        &summation_dimensions,
                                        &input_2_shape_const
                                    )
                                );
                            let product_3 = tensor_3_indexing
                                .get_ref(
                                    filter_outer_and_summation_indexes(
                                        &indexes,
                                        &output_shape,
                                        &summation_indexes,
                                        &summation_dimensions,
                                        &input_3_shape_const
                                    )
                                );
                            sum = sum + (product_1 * product_2 * product_3);
                        }
                        None => break
                    }
                }
            }

            *element = sum;
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
