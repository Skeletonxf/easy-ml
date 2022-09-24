use crate::tensors::views::TensorRef;
use crate::tensors::Dimension;
use std::marker::PhantomData;
use crate::tensors::dimensions;

/**
 * Combines two or more tensors with the same shape along a new dimension to create a Tensor
 * with one additional dimension which stacks the sources together along that dimension.
 */
pub struct TensorStack<T, S, const D: usize> {
    sources: S,
    _type: PhantomData<T>,
    along: (usize, Dimension),
}

fn validate_shapes_equal<const D: usize, I>(mut shapes: I)
where
    I: Iterator<Item = [(Dimension, usize); D]>
{
    // We'll reject fewer than one tensors in the constructors before getting here, so first unwrap
    // is always going to succeed.
    let first_shape = shapes.next().unwrap();
    for (i, shape) in shapes.enumerate() {
        if shape != first_shape {
            panic!(
                "The shapes of each tensor in the sources to stack along must be the same. Shape {:?} {:?} did not match the first shape {:?}",
                i + 1, shape, first_shape
            );
        }
    }
}

// TODO: Can dyn TensorRef<T, D> be generalised here to any S: TensorRef<T, D>?
impl<T, const D: usize, const N: usize> TensorStack<T, [Box<dyn TensorRef<T, D>>; N], D> {
    #[track_caller]
    pub fn from(sources: [Box<dyn TensorRef<T, D>>; N], along: (usize, Dimension)) -> Self {
        if N == 0 {
            panic!("No sources provided");
        }
        if along.0 > D {
            panic!(
                "The extra dimension the sources are stacked along {:?} must be inserted in the range 0 <= d <= D of the source shapes",
                along
            );
        }
        let shape = sources[0].view_shape();
        if dimensions::contains(&shape, along.1) {
            panic!(
                "The extra dimension the sources are stacked along {:?} must not be one of the dimensions already in the source shapes: {:?}",
                along,
                shape
            );
        }
        validate_shapes_equal(sources.iter().map(|tensor| tensor.view_shape()));
        Self {
            sources,
            along,
            _type: PhantomData,
        }
    }
}

impl<T, S1, S2, const D: usize> TensorStack<T, (S1, S2), D>
where
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    #[track_caller]
    pub fn from(sources: (S1, S2), along: (usize, Dimension)) -> Self {
        if along.0 > D {
            panic!(
                "The extra dimension the sources are stacked along {:?} must be inserted in the range 0 <= d <= D of the source shapes",
                along
            );
        }
        let shape = sources.0.view_shape();
        if dimensions::contains(&shape, along.1) {
            panic!(
                "The extra dimension the sources are stacked along {:?} must not be one of the dimensions already in the source shapes: {:?}",
                along,
                shape
            );
        }
        validate_shapes_equal([sources.0.view_shape(), sources.1.view_shape()].into_iter());
        Self {
            sources,
            along,
            _type: PhantomData,
        }
    }
}

impl<T, S1, S2, S3, const D: usize> TensorStack<T, (S1, S2, S3), D>
where
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
    S3: TensorRef<T, D>,
{
    #[track_caller]
    pub fn from(sources: (S1, S2, S3), along: (usize, Dimension)) -> Self {
        if along.0 > D {
            panic!(
                "The extra dimension the sources are stacked along {:?} must be inserted in the range 0 <= d <= D of the source shapes",
                along
            );
        }
        let shape = sources.0.view_shape();
        if dimensions::contains(&shape, along.1) {
            panic!(
                "The extra dimension the sources are stacked along {:?} must not be one of the dimensions already in the source shapes: {:?}",
                along,
                shape
            );
        }
        validate_shapes_equal(
            [
                sources.0.view_shape(), sources.1.view_shape(), sources.2.view_shape()
            ].into_iter()
        );
        Self {
            sources,
            along,
            _type: PhantomData,
        }
    }
}
