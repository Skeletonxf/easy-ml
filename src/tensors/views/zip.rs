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

    fn source_view_shape(&self) -> [(Dimension, usize); D] {
        self.sources[0].view_shape()
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

    fn source_view_shape(&self) -> [(Dimension, usize); D] {
        self.sources.0.view_shape()
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

    fn source_view_shape(&self) -> [(Dimension, usize); D] {
        self.sources.0.view_shape()
    }
}


impl<T, S1, S2, S3, S4, const D: usize> TensorStack<T, (S1, S2, S3, S4), D>
where
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
    S3: TensorRef<T, D>,
    S4: TensorRef<T, D>,
{
    #[track_caller]
    pub fn from(sources: (S1, S2, S3, S4), along: (usize, Dimension)) -> Self {
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
                sources.0.view_shape(), sources.1.view_shape(),
                sources.2.view_shape(), sources.3.view_shape()
            ].into_iter()
        );
        Self {
            sources,
            along,
            _type: PhantomData,
        }
    }

    fn source_view_shape(&self) -> [(Dimension, usize); D] {
        self.sources.0.view_shape()
    }
}

macro_rules! tensor_stack_ref_impl {
    (unsafe impl TensorRef for TensorStack $d:literal $mod:ident) => {
        // To avoid helper name clashes we use a different module per macro invocation
        mod $mod {
            use crate::tensors::views::{TensorRef, DataLayout, TensorStack};
            use crate::tensors::Dimension;

            fn view_shape_impl(
                shape: [(Dimension, usize); $d],
                along: (usize, Dimension)
            ) -> [(Dimension, usize); $d + 1] {
                let mut extra_shape = [("", 0); $d + 1];
                let mut i = 0;
                for dimension in extra_shape.iter_mut() {
                    match i == along.0 {
                        true => {
                            *dimension = (along.1, along.0);
                            // Do not increment i, this is the added dimension
                        }
                        false => {
                            *dimension = shape[i];
                            i += 1;
                        }
                    }
                }
                extra_shape
            }

            unsafe impl<T, const N: usize> TensorRef<T, { $d + 1 }> for TensorStack<T, [Box<dyn TensorRef<T, $d>>; N], $d> {
                fn get_reference(&self, _indexes: [usize; $d + 1]) -> Option<&T> {
                    unimplemented!()
                }

                fn view_shape(&self) -> [(Dimension, usize); $d + 1] {
                    view_shape_impl(self.source_view_shape(), self.along)
                }

                unsafe fn get_reference_unchecked(&self, _indexes: [usize; $d + 1]) -> &T {
                    unimplemented!()
                }

                fn data_layout(&self) -> DataLayout<{ $d + 1 }> {
                    // Our stacked shapes means the view shape no longer matches up to a single
                    // line of data in memory.
                    DataLayout::NonLinear
                }
            }

            unsafe impl<T, S1, S2> TensorRef<T, { $d + 1 }> for TensorStack<T, (S1, S2), $d>
            where
                S1: TensorRef<T, $d>,
                S2: TensorRef<T, $d>,
            {
                fn get_reference(&self, _indexes: [usize; $d + 1]) -> Option<&T> {
                    unimplemented!()
                }

                fn view_shape(&self) -> [(Dimension, usize); $d + 1] {
                    view_shape_impl(self.source_view_shape(), self.along)
                }

                unsafe fn get_reference_unchecked(&self, _indexes: [usize; $d + 1]) -> &T {
                    unimplemented!()
                }

                fn data_layout(&self) -> DataLayout<{ $d + 1 }> {
                    // Our stacked shapes means the view shape no longer matches up to a single
                    // line of data in memory.
                    DataLayout::NonLinear
                }
            }

            unsafe impl<T, S1, S2, S3> TensorRef<T, { $d + 1 }> for TensorStack<T, (S1, S2, S3), $d>
            where
                S1: TensorRef<T, $d>,
                S2: TensorRef<T, $d>,
                S3: TensorRef<T, $d>,
            {
                fn get_reference(&self, _indexes: [usize; $d + 1]) -> Option<&T> {
                    unimplemented!()
                }

                fn view_shape(&self) -> [(Dimension, usize); $d + 1] {
                    view_shape_impl(self.source_view_shape(), self.along)
                }

                unsafe fn get_reference_unchecked(&self, _indexes: [usize; $d + 1]) -> &T {
                    unimplemented!()
                }

                fn data_layout(&self) -> DataLayout<{ $d + 1 }> {
                    // Our stacked shapes means the view shape no longer matches up to a single
                    // line of data in memory.
                    DataLayout::NonLinear
                }
            }

            unsafe impl<T, S1, S2, S3, S4> TensorRef<T, { $d + 1 }> for TensorStack<T, (S1, S2, S3, S4), $d>
            where
                S1: TensorRef<T, $d>,
                S2: TensorRef<T, $d>,
                S3: TensorRef<T, $d>,
                S4: TensorRef<T, $d>,
            {
                fn get_reference(&self, _indexes: [usize; $d + 1]) -> Option<&T> {
                    unimplemented!()
                }

                fn view_shape(&self) -> [(Dimension, usize); $d + 1] {
                    view_shape_impl(self.source_view_shape(), self.along)
                }

                unsafe fn get_reference_unchecked(&self, _indexes: [usize; $d + 1]) -> &T {
                    unimplemented!()
                }

                fn data_layout(&self) -> DataLayout<{ $d + 1 }> {
                    // Our stacked shapes means the view shape no longer matches up to a single
                    // line of data in memory.
                    DataLayout::NonLinear
                }
            }
        }
    }
}

tensor_stack_ref_impl!(unsafe impl TensorRef for TensorStack 0 zero);
tensor_stack_ref_impl!(unsafe impl TensorRef for TensorStack 1 one);
tensor_stack_ref_impl!(unsafe impl TensorRef for TensorStack 2 two);
tensor_stack_ref_impl!(unsafe impl TensorRef for TensorStack 3 three);
tensor_stack_ref_impl!(unsafe impl TensorRef for TensorStack 4 four);
tensor_stack_ref_impl!(unsafe impl TensorRef for TensorStack 5 five);
