use crate::tensors::views::{TensorRef, DataLayout};
use crate::tensors::Dimension;
use std::marker::PhantomData;
use crate::tensors::dimensions;

/**
 * Combines two or more tensors with the same shape along a new dimension to create a Tensor
 * with one additional dimension which stacks the sources together along that dimension.
 *
 * Note: due to limitations in Rust's const generics support, TensorStack only implements
 * TensorRef for D from `1` to `6` (from sources of `0` to `5` dimensions respectively), and
 * only supports tuple combinations for `2` to `4`. If you need to stack more than four tensors
 * together, you can stack any number with the `[S; N]` implementation, though note this requires
 * that all the tensors are the same type so you may need to box and erase the types to
 * `Box<dyn TensorRef<T, D>>`.
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * use easy_ml::tensors::views::{TensorView, TensorStack, TensorRef};
 * let vector1 = Tensor::from([("data", 5)], vec![0, 1, 2, 3, 4]);
 * let vector2 = Tensor::from([("data", 5)], vec![2, 4, 8, 16, 32]);
 * // Because there are 4 variants of `TensorStack::from` you may need to use the turbofish
 * // to tell the Rust compiler which variant you're using, but the actual type of `S` can be
 * // left unspecified by using an underscore.
 * let matrix = TensorStack::<i32, [_; 2], 1>::from([&vector1, &vector2], (0, "sample"));
 * let equal_matrix = Tensor::from([("sample", 2), ("data", 5)], vec![
 *   0, 1, 2, 3, 4,
 *   2, 4, 8, 16, 32
 * ]);
 * assert_eq!(equal_matrix, TensorView::from(matrix));
 *
 * let also_matrix = TensorStack::<i32, (_, _), 1>::from((vector1, vector2), (0, "sample"));
 * assert_eq!(equal_matrix, TensorView::from(&also_matrix));
 *
 * // To stack `equal_matrix` and `also_matrix` using the `[S; N]` implementation we have to first
 * // make them the same type, which we can do by boxing and erasing.
 * let matrix_erased: Box<dyn TensorRef<i32, 2>> = Box::new(also_matrix);
 * let equal_matrix_erased: Box<dyn TensorRef<i32, 2>> = Box::new(equal_matrix);
 * let tensor = TensorStack::<i32, [_; 2], 2>::from(
 *     [matrix_erased, equal_matrix_erased], (0, "experiment")
 * );
 * assert!(
 *     TensorView::from(tensor).eq(
 *         &Tensor::from([("experiment", 2), ("sample", 2), ("data", 5)], vec![
 *             0, 1, 2, 3, 4,
 *             2, 4, 8, 16, 32,
 *
 *             0, 1, 2, 3, 4,
 *             2, 4, 8, 16, 32
 *         ])
 *     ),
 * );
 * ```
 */
#[derive(Clone, Debug)]
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

impl<T, S, const D: usize, const N: usize> TensorStack<T, [S; N], D>
where
    S: TensorRef<T, D>
{
    /**
     * Creates a TensorStack from an array of sources of the same type and a tuple of which
     * dimension and name to stack the sources along in the range of 0 <= `d` <= D. The sources
     * must all have an identical shape, and the dimension name to add must not be in the sources'
     * shape already.
     *
     * # Panics
     *
     * If N == 0, the shapes of the sources are not identical, the dimension for stacking is out
     * of bounds, or the name is already in the sources' shape.
     *
     * While N == 1 arguments may be valid [TensorExpansion](crate::tensors::views::TensorExpansion)
     * is a more general way to add dimensions with no additional data.
     */
    #[track_caller]
    pub fn from(sources: [S; N], along: (usize, Dimension)) -> Self {
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

    pub fn sources(self) -> [S; N] {
        self.sources
    }

    // # Safety
    //
    // Giving out a mutable reference to our sources could allow then to be changed out from under
    // us and make our shape invalid. However, since the sources implement TensorRef interior
    // mutability is not allowed, so we can give out shared references without breaking our own
    // integrity.
    pub fn sources_ref(&self) -> &[S; N] {
        &self.sources
    }

    fn source_view_shape(&self) -> [(Dimension, usize); D] {
        self.sources[0].view_shape()
    }

    fn number_of_sources() -> usize {
        N
    }
}

impl<T, S1, S2, const D: usize> TensorStack<T, (S1, S2), D>
where
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    /**
     * Creates a TensorStack from two sources and a tuple of which dimension and name to stack
     * the sources along in the range of 0 <= `d` <= D. The sources must all have an identical
     * shape, and the dimension name to add must not be in the sources' shape already.
     *
     * # Panics
     *
     * If the shapes of the sources are not identical, the dimension for stacking is out
     * of bounds, or the name is already in the sources' shape.
     */
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

    pub fn sources(self) -> (S1, S2) {
        self.sources
    }

    // # Safety
    //
    // Giving out a mutable reference to our sources could allow then to be changed out from under
    // us and make our shape invalid. However, since the sources implement TensorRef interior
    // mutability is not allowed, so we can give out shared references without breaking our own
    // integrity.
    pub fn sources_ref(&self) -> &(S1, S2) {
        &self.sources
    }

    fn source_view_shape(&self) -> [(Dimension, usize); D] {
        self.sources.0.view_shape()
    }

    fn number_of_sources() -> usize {
        2
    }
}

impl<T, S1, S2, S3, const D: usize> TensorStack<T, (S1, S2, S3), D>
where
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
    S3: TensorRef<T, D>,
{
    /**
     * Creates a TensorStack from three sources and a tuple of which dimension and name to stack
     * the sources along in the range of 0 <= `d` <= D. The sources must all have an identical
     * shape, and the dimension name to add must not be in the sources' shape already.
     *
     * # Panics
     *
     * If the shapes of the sources are not identical, the dimension for stacking is out
     * of bounds, or the name is already in the sources' shape.
     */
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

    pub fn sources(self) -> (S1, S2, S3) {
        self.sources
    }

    // # Safety
    //
    // Giving out a mutable reference to our sources could allow then to be changed out from under
    // us and make our shape invalid. However, since the sources implement TensorRef interior
    // mutability is not allowed, so we can give out shared references without breaking our own
    // integrity.
    pub fn sources_ref(&self) -> &(S1, S2, S3) {
        &self.sources
    }

    fn source_view_shape(&self) -> [(Dimension, usize); D] {
        self.sources.0.view_shape()
    }

    fn number_of_sources() -> usize {
        3
    }
}

impl<T, S1, S2, S3, S4, const D: usize> TensorStack<T, (S1, S2, S3, S4), D>
where
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
    S3: TensorRef<T, D>,
    S4: TensorRef<T, D>,
{
    /**
     * Creates a TensorStack from four sources and a tuple of which dimension and name to stack
     * the sources along in the range of 0 <= `d` <= D. The sources must all have an identical
     * shape, and the dimension name to add must not be in the sources' shape already.
     *
     * # Panics
     *
     * If the shapes of the sources are not identical, the dimension for stacking is out
     * of bounds, or the name is already in the sources' shape.
     */
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

    pub fn sources(self) -> (S1, S2, S3, S4) {
        self.sources
    }

    // # Safety
    //
    // Giving out a mutable reference to our sources could allow then to be changed out from under
    // us and make our shape invalid. However, since the sources implement TensorRef interior
    // mutability is not allowed, so we can give out shared references without breaking our own
    // integrity.
    pub fn sources_ref(&self) -> &(S1, S2, S3, S4) {
        &self.sources
    }

    fn source_view_shape(&self) -> [(Dimension, usize); D] {
        self.sources.0.view_shape()
    }

    fn number_of_sources() -> usize {
        4
    }
}

macro_rules! tensor_stack_ref_impl {
    (unsafe impl TensorRef for TensorStack $d:literal $mod:ident) => {
        // To avoid helper name clashes we use a different module per macro invocation
        mod $mod {
            use crate::tensors::views::{TensorRef, TensorMut, DataLayout, TensorStack};
            use crate::tensors::Dimension;

            fn view_shape_impl(
                shape: [(Dimension, usize); $d],
                along: (usize, Dimension),
                sources: usize,
            ) -> [(Dimension, usize); $d + 1] {
                let mut extra_shape = [("", 0); $d + 1];
                let mut i = 0;
                for (d, dimension) in extra_shape.iter_mut().enumerate() {
                    match d == along.0 {
                        true => {
                            *dimension = (along.1, sources);
                            // Do not increment i, this is the added dimension
                        },
                        false => {
                            *dimension = shape[i];
                            i += 1;
                        }
                    }
                }
                extra_shape
            }

            fn indexing(
                indexes: [usize; $d + 1],
                along: (usize, Dimension)
            ) -> (usize, [usize; $d]) {
                let mut indexes_into_source = [0; $d];
                let mut i = 0;
                for (d, &index) in indexes.iter().enumerate() {
                    if d != along.0 {
                        indexes_into_source[i] = index;
                        i += 1;
                    }
                }
                (indexes[along.0], indexes_into_source)
            }

            unsafe impl<T, S, const N: usize> TensorRef<T, { $d + 1 }> for TensorStack<T, [S; N], $d>
            where
                S: TensorRef<T, $d>
            {
                fn get_reference(&self, indexes: [usize; $d + 1]) -> Option<&T> {
                    let (source, indexes) = indexing(indexes, self.along);
                    self.sources.get(source)?.get_reference(indexes)
                }

                fn view_shape(&self) -> [(Dimension, usize); $d + 1] {
                    view_shape_impl(self.source_view_shape(), self.along, Self::number_of_sources())
                }

                unsafe fn get_reference_unchecked(&self, indexes: [usize; $d + 1]) -> &T {
                    let (source, indexes) = indexing(indexes, self.along);
                    // TODO: Can we use get_unchecked here?
                    self.sources.get(source).unwrap().get_reference_unchecked(indexes)
                }

                fn data_layout(&self) -> DataLayout<{ $d + 1 }> {
                    // Our stacked shapes means the view shape no longer matches up to a single
                    // line of data in memory.
                    DataLayout::NonLinear
                }
            }

            unsafe impl<T, S, const N: usize> TensorMut<T, { $d + 1 }> for TensorStack<T, [S; N], $d>
            where
                S: TensorMut<T, $d>
            {
                fn get_reference_mut(&mut self, indexes: [usize; $d + 1]) -> Option<&mut T> {
                    let (source, indexes) = indexing(indexes, self.along);
                    self.sources.get_mut(source)?.get_reference_mut(indexes)
                }

                unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; $d + 1]) -> &mut T {
                    let (source, indexes) = indexing(indexes, self.along);
                    // TODO: Can we use get_unchecked here?
                    self.sources.get_mut(source).unwrap().get_reference_unchecked_mut(indexes)
                }
            }

            unsafe impl<T, S1, S2> TensorRef<T, { $d + 1 }> for TensorStack<T, (S1, S2), $d>
            where
                S1: TensorRef<T, $d>,
                S2: TensorRef<T, $d>,
            {
                fn get_reference(&self, indexes: [usize; $d + 1]) -> Option<&T> {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference(indexes),
                        1 => self.sources.1.get_reference(indexes),
                        _ => None
                    }
                }

                fn view_shape(&self) -> [(Dimension, usize); $d + 1] {
                    view_shape_impl(self.source_view_shape(), self.along, Self::number_of_sources())
                }

                unsafe fn get_reference_unchecked(&self, indexes: [usize; $d + 1]) -> &T {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference_unchecked(indexes),
                        1 => self.sources.1.get_reference_unchecked(indexes),
                        // TODO: Can we use unreachable_unchecked here?
                        _ => panic!(
                            "Invalid index should never be given to get_reference_unchecked"
                        )
                    }
                }

                fn data_layout(&self) -> DataLayout<{ $d + 1 }> {
                    // Our stacked shapes means the view shape no longer matches up to a single
                    // line of data in memory.
                    DataLayout::NonLinear
                }
            }

            unsafe impl<T, S1, S2> TensorMut<T, { $d + 1 }> for TensorStack<T, (S1, S2), $d>
            where
                S1: TensorMut<T, $d>,
                S2: TensorMut<T, $d>,
            {
                fn get_reference_mut(&mut self, indexes: [usize; $d + 1]) -> Option<&mut T> {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference_mut(indexes),
                        1 => self.sources.1.get_reference_mut(indexes),
                        _ => None
                    }
                }

                unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; $d + 1]) -> &mut T {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference_unchecked_mut(indexes),
                        1 => self.sources.1.get_reference_unchecked_mut(indexes),
                        // TODO: Can we use unreachable_unchecked here?
                        _ => panic!(
                            "Invalid index should never be given to get_reference_unchecked"
                        )
                    }
                }
            }

            unsafe impl<T, S1, S2, S3> TensorRef<T, { $d + 1 }> for TensorStack<T, (S1, S2, S3), $d>
            where
                S1: TensorRef<T, $d>,
                S2: TensorRef<T, $d>,
                S3: TensorRef<T, $d>,
            {
                fn get_reference(&self, indexes: [usize; $d + 1]) -> Option<&T> {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference(indexes),
                        1 => self.sources.1.get_reference(indexes),
                        2 => self.sources.2.get_reference(indexes),
                        _ => None
                    }
                }

                fn view_shape(&self) -> [(Dimension, usize); $d + 1] {
                    view_shape_impl(self.source_view_shape(), self.along, Self::number_of_sources())
                }

                unsafe fn get_reference_unchecked(&self, indexes: [usize; $d + 1]) -> &T {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference_unchecked(indexes),
                        1 => self.sources.1.get_reference_unchecked(indexes),
                        2 => self.sources.2.get_reference_unchecked(indexes),
                        // TODO: Can we use unreachable_unchecked here?
                        _ => panic!(
                            "Invalid index should never be given to get_reference_unchecked"
                        )
                    }
                }

                fn data_layout(&self) -> DataLayout<{ $d + 1 }> {
                    // Our stacked shapes means the view shape no longer matches up to a single
                    // line of data in memory.
                    DataLayout::NonLinear
                }
            }

            unsafe impl<T, S1, S2, S3> TensorMut<T, { $d + 1 }> for TensorStack<T, (S1, S2, S3), $d>
            where
                S1: TensorMut<T, $d>,
                S2: TensorMut<T, $d>,
                S3: TensorMut<T, $d>,
            {
                fn get_reference_mut(&mut self, indexes: [usize; $d + 1]) -> Option<&mut T> {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference_mut(indexes),
                        1 => self.sources.1.get_reference_mut(indexes),
                        2 => self.sources.2.get_reference_mut(indexes),
                        _ => None
                    }
                }

                unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; $d + 1]) -> &mut T {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference_unchecked_mut(indexes),
                        1 => self.sources.1.get_reference_unchecked_mut(indexes),
                        2 => self.sources.2.get_reference_unchecked_mut(indexes),
                        // TODO: Can we use unreachable_unchecked here?
                        _ => panic!(
                            "Invalid index should never be given to get_reference_unchecked"
                        )
                    }
                }
            }

            unsafe impl<T, S1, S2, S3, S4> TensorRef<T, { $d + 1 }> for TensorStack<T, (S1, S2, S3, S4), $d>
            where
                S1: TensorRef<T, $d>,
                S2: TensorRef<T, $d>,
                S3: TensorRef<T, $d>,
                S4: TensorRef<T, $d>,
            {
                fn get_reference(&self, indexes: [usize; $d + 1]) -> Option<&T> {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference(indexes),
                        1 => self.sources.1.get_reference(indexes),
                        2 => self.sources.2.get_reference(indexes),
                        3 => self.sources.3.get_reference(indexes),
                        _ => None
                    }
                }

                fn view_shape(&self) -> [(Dimension, usize); $d + 1] {
                    view_shape_impl(self.source_view_shape(), self.along, Self::number_of_sources())
                }

                unsafe fn get_reference_unchecked(&self, indexes: [usize; $d + 1]) -> &T {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference_unchecked(indexes),
                        1 => self.sources.1.get_reference_unchecked(indexes),
                        2 => self.sources.2.get_reference_unchecked(indexes),
                        3 => self.sources.3.get_reference_unchecked(indexes),
                        // TODO: Can we use unreachable_unchecked here?
                        _ => panic!(
                            "Invalid index should never be given to get_reference_unchecked"
                        )
                    }
                }

                fn data_layout(&self) -> DataLayout<{ $d + 1 }> {
                    // Our stacked shapes means the view shape no longer matches up to a single
                    // line of data in memory.
                    DataLayout::NonLinear
                }
            }

            unsafe impl<T, S1, S2, S3, S4> TensorMut<T, { $d + 1 }> for TensorStack<T, (S1, S2, S3, S4), $d>
            where
                S1: TensorMut<T, $d>,
                S2: TensorMut<T, $d>,
                S3: TensorMut<T, $d>,
                S4: TensorMut<T, $d>,
            {
                fn get_reference_mut(&mut self, indexes: [usize; $d + 1]) -> Option<&mut T> {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference_mut(indexes),
                        1 => self.sources.1.get_reference_mut(indexes),
                        2 => self.sources.2.get_reference_mut(indexes),
                        3 => self.sources.3.get_reference_mut(indexes),
                        _ => None
                    }
                }

                unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; $d + 1]) -> &mut T {
                    let (source, indexes) = indexing(indexes, self.along);
                    match source {
                        0 => self.sources.0.get_reference_unchecked_mut(indexes),
                        1 => self.sources.1.get_reference_unchecked_mut(indexes),
                        2 => self.sources.2.get_reference_unchecked_mut(indexes),
                        3 => self.sources.3.get_reference_unchecked_mut(indexes),
                        // TODO: Can we use unreachable_unchecked here?
                        _ => panic!(
                            "Invalid index should never be given to get_reference_unchecked"
                        )
                    }
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

#[test]
fn test_stacking() {
    use crate::tensors::Tensor;
    use crate::tensors::views::{TensorView, TensorMut};
    let vector1 = Tensor::from([("a", 3)], vec![9, 5, 2]);
    let vector2 = Tensor::from([("a", 3)], vec![3, 6, 0]);
    let vector3 = Tensor::from([("a", 3)], vec![8, 7, 1]);
    let matrix = TensorView::from(
        TensorStack::<_, (_, _, _), 1>::from((&vector1, &vector2, &vector3), (1, "b"))
    );
    assert_eq!(
        matrix,
        Tensor::from([("a", 3), ("b", 3)], vec![
            9, 3, 8,
            5, 6, 7,
            2, 0, 1,
        ])
    );
    let different_matrix = TensorView::from(
        TensorStack::<_, (_, _, _), 1>::from((&vector1, &vector2, &vector3), (0, "b"))
    );
    assert_eq!(
        different_matrix,
        Tensor::from([("b", 3), ("a", 3)], vec![
            9, 5, 2,
            3, 6, 0,
            8, 7, 1,
        ])
    );
    let matrix_erased: Box<dyn TensorMut<_, 2>> = Box::new(matrix.map(|x| x));
    let different_matrix_erased: Box<dyn TensorMut<_, 2>> = Box::new(
        different_matrix.rename_view(["a", "b"]).map(|x| x)
    );
    let tensor = TensorView::from(
        TensorStack::<_, [_; 2], 2>::from([matrix_erased, different_matrix_erased], (2, "c"))
    );
    assert!(
        tensor.eq(
            &Tensor::from([("a", 3), ("b", 3), ("c", 2)], vec![
                9, 9,
                3, 5,
                8, 2,

                5, 3,
                6, 6,
                7, 0,

                2, 8,
                0, 7,
                1, 1
            ])
        ),
    );
    let matrix_erased: Box<dyn TensorMut<_, 2>> = Box::new(matrix.map(|x| x));
    let different_matrix_erased: Box<dyn TensorMut<_, 2>> = Box::new(
        different_matrix.rename_view(["a", "b"]).map(|x| x)
    );
    let different_tensor = TensorView::from(
        TensorStack::<_, [_; 2], 2>::from([matrix_erased, different_matrix_erased], (1, "c"))
    );
    assert!(
        different_tensor.eq(
            &Tensor::from([("a", 3), ("c", 2), ("b", 3)], vec![
                9, 3, 8,
                9, 5, 2,

                5, 6, 7,
                3, 6, 0,

                2, 0, 1,
                8, 7, 1
            ])
        ),
    );
    let matrix_erased: Box<dyn TensorRef<_, 2>> = Box::new(matrix.map(|x| x));
    let different_matrix_erased: Box<dyn TensorRef<_, 2>> = Box::new(
        different_matrix.rename_view(["a", "b"]).map(|x| x)
    );
    let another_tensor = TensorView::from(
        TensorStack::<_, [_; 2], 2>::from([matrix_erased, different_matrix_erased], (0, "c"))
    );
    assert!(
        another_tensor.eq(
            &Tensor::from([("c", 2), ("a", 3), ("b", 3)], vec![
                9, 3, 8,
                5, 6, 7,
                2, 0, 1,

                9, 5, 2,
                3, 6, 0,
                8, 7, 1,
            ])
        ),
    );
}

#[derive(Clone, Debug)]
pub struct TensorZip<T, S, const D: usize> {
    sources: S,
    _type: PhantomData<T>,
    along: usize,
}

fn validate_shapes_similar<const D: usize, I>(mut shapes: I, along: usize)
where
    I: Iterator<Item = [(Dimension, usize); D]>
{
    // We'll reject fewer than one tensors in the constructors before getting here, so first unwrap
    // is always going to succeed.
    let first_shape = shapes.next().unwrap();
    for (i, shape) in shapes.enumerate() {
        for d in 0..D {
            let similar = if d == along {
                // don't need match for dimension lengths in the `along` dimension
                shape[d].0 == first_shape[d].0
            } else {
                shape[d] == first_shape[d]
            };
            if !similar {
                panic!(
                    "The shapes of each tensor in the sources to zip along must be the same. Shape {:?} {:?} did not match the first shape {:?}",
                    i + 1, shape, first_shape
                );
            }
        }
    }
}

impl<T, S, const D: usize, const N: usize> TensorZip<T, [S; N], D>
where
    S: TensorRef<T, D>
{
    /**
     * Creates a TensorZip from an array of sources of the same type and the dimension name to
     * zip the sources along. The sources must all have an identical shape, including the provided
     * dimension, except for the dimension lengths of the provided dimension name which may be
     * different.
     *
     * # Panics
     *
     * If N == 0, D == 0, the shapes of the sources are not identical*, or the dimension for
     * zipping is not in sources' shape.
     *
     * *except for the lengths along the provided dimension.
     */
    #[track_caller]
    pub fn from(sources: [S; N], along: Dimension) -> Self {
        if N == 0 {
            panic!("No sources provided");
        }
        if D == 0 {
            panic!("Can't zip along 0 dimensional tensors");
        }
        let shape = sources[0].view_shape();
        let along = match dimensions::position_of(&shape, along) {
            Some(d) => d,
            None => panic!("The dimension {:?} is not in the source's shapes: {:?}", along, shape),
        };
        validate_shapes_similar(sources.iter().map(|tensor| tensor.view_shape()), along);
        Self {
            sources,
            along,
            _type: PhantomData,
        }
    }

    pub fn sources(self) -> [S; N] {
        self.sources
    }

    // # Safety
    //
    // Giving out a mutable reference to our sources could allow then to be changed out from under
    // us and make our shape invalid. However, since the sources implement TensorRef interior
    // mutability is not allowed, so we can give out shared references without breaking our own
    // integrity.
    pub fn sources_ref(&self) -> &[S; N] {
        &self.sources
    }
}

impl<T, S1, S2, const D: usize> TensorZip<T, (S1, S2), D>
where
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
{
    /**
     * Creates a TensorZip from two sources and the dimension name to zip the sources along. The
     * sources must all have an identical shape, including the provided dimension, except for the
     * dimension lengths of the provided dimension name which may be different.
     *
     * # Panics
     *
     * If D == 0, the shapes of the sources are not identical*, or the dimension for
     * zipping is not in sources' shape.
     *
     * *except for the lengths along the provided dimension.
     */
    #[track_caller]
    pub fn from(sources: (S1, S2), along: Dimension) -> Self {
        if D == 0 {
            panic!("Can't zip along 0 dimensional tensors");
        }
        let shape = sources.0.view_shape();
        let along = match dimensions::position_of(&shape, along) {
            Some(d) => d,
            None => panic!("The dimension {:?} is not in the source's shapes: {:?}", along, shape),
        };
        validate_shapes_similar(
            [sources.0.view_shape(), sources.1.view_shape()].into_iter(),
            along
        );
        Self {
            sources,
            along,
            _type: PhantomData,
        }
    }

    pub fn sources(self) -> (S1, S2) {
        self.sources
    }

    // # Safety
    //
    // Giving out a mutable reference to our sources could allow then to be changed out from under
    // us and make our shape invalid. However, since the sources implement TensorRef interior
    // mutability is not allowed, so we can give out shared references without breaking our own
    // integrity.
    pub fn sources_ref(&self) -> &(S1, S2) {
        &self.sources
    }
}

impl<T, S1, S2, S3, const D: usize> TensorZip<T, (S1, S2, S3), D>
where
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
    S3: TensorRef<T, D>,
{
    /**
     * Creates a TensorZip from three sources and the dimension name to zip the sources along. The
     * sources must all have an identical shape, including the provided dimension, except for the
     * dimension lengths of the provided dimension name which may be different.
     *
     * # Panics
     *
     * If D == 0, the shapes of the sources are not identical*, or the dimension for
     * zipping is not in sources' shape.
     *
     * *except for the lengths along the provided dimension.
     */
    #[track_caller]
    pub fn from(sources: (S1, S2, S3), along: Dimension) -> Self {
        if D == 0 {
            panic!("Can't zip along 0 dimensional tensors");
        }
        let shape = sources.0.view_shape();
        let along = match dimensions::position_of(&shape, along) {
            Some(d) => d,
            None => panic!("The dimension {:?} is not in the source's shapes: {:?}", along, shape),
        };
        validate_shapes_similar(
            [
                sources.0.view_shape(), sources.1.view_shape(), sources.2.view_shape()
            ].into_iter(),
            along
        );
        Self {
            sources,
            along,
            _type: PhantomData,
        }
    }

    pub fn sources(self) -> (S1, S2, S3) {
        self.sources
    }

    // # Safety
    //
    // Giving out a mutable reference to our sources could allow then to be changed out from under
    // us and make our shape invalid. However, since the sources implement TensorRef interior
    // mutability is not allowed, so we can give out shared references without breaking our own
    // integrity.
    pub fn sources_ref(&self) -> &(S1, S2, S3) {
        &self.sources
    }
}

impl<T, S1, S2, S3, S4, const D: usize> TensorZip<T, (S1, S2, S3, S4), D>
where
    S1: TensorRef<T, D>,
    S2: TensorRef<T, D>,
    S3: TensorRef<T, D>,
    S4: TensorRef<T, D>,
{
    /**
     * Creates a TensorZip from four sources and the dimension name to zip the sources along. The
     * sources must all have an identical shape, including the provided dimension, except for the
     * dimension lengths of the provided dimension name which may be different.
     *
     * # Panics
     *
     * If D == 0, the shapes of the sources are not identical*, or the dimension for
     * zipping is not in sources' shape.
     *
     * *except for the lengths along the provided dimension.
     */
    #[track_caller]
    pub fn from(sources: (S1, S2, S3, S4), along: Dimension) -> Self {
        if D == 0 {
            panic!("Can't zip along 0 dimensional tensors");
        }
        let shape = sources.0.view_shape();
        let along = match dimensions::position_of(&shape, along) {
            Some(d) => d,
            None => panic!("The dimension {:?} is not in the source's shapes: {:?}", along, shape),
        };
        validate_shapes_similar(
            [
                sources.0.view_shape(), sources.1.view_shape(),
                sources.2.view_shape(), sources.3.view_shape()
            ].into_iter(),
            along
        );
        Self {
            sources,
            along,
            _type: PhantomData,
        }
    }

    pub fn sources(self) -> (S1, S2, S3, S4) {
        self.sources
    }

    // # Safety
    //
    // Giving out a mutable reference to our sources could allow then to be changed out from under
    // us and make our shape invalid. However, since the sources implement TensorRef interior
    // mutability is not allowed, so we can give out shared references without breaking our own
    // integrity.
    pub fn sources_ref(&self) -> &(S1, S2, S3, S4) {
        &self.sources
    }
}

// For indexing, need to go from indexes into D dimensions, to indexes into D dimensions + which
// source the provided dimension index fell into, in general case, we can iterate till our index
// is less than the length of the nth tensor, and when it's not, subtract that length and increase
// n by one
// View shape will match sources in all dimensions expect provided, where it'll be the sum of the
// lengths

unsafe impl<T, S, const D: usize, const N: usize> TensorRef<T, D> for TensorZip<T, [S; N], D>
where
    S: TensorRef<T, D>
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        unimplemented!()
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        unimplemented!()
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        unimplemented!()
    }

    fn data_layout(&self) -> DataLayout<D> {
        // Our zipped shapes means the view shape no longer matches up to a single
        // line of data in memory in the general case.
        DataLayout::NonLinear
    }
}
