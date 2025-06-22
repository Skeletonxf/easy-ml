use crate::tensors::Dimension;
use crate::tensors::views::{DataLayout, TensorMut, TensorRef};
use std::marker::PhantomData;

/**
* A combination of pre provided indexes and a tensor. The provided indexes reduce the
* dimensionality of the TensorRef exposed to less than the dimensionality of the TensorRef
* this is created from.
*
* ```
* use easy_ml::tensors::Tensor;
* use easy_ml::tensors::views::{TensorView, TensorIndex};
* let vector = Tensor::from([("a", 2)], vec![ 16, 8 ]);
* let scalar = vector.select([("a", 0)]);
* let also_scalar = TensorView::from(TensorIndex::from(&vector, [("a", 0)]));
* assert_eq!(scalar.index_by([]).get([]), also_scalar.index_by([]).get([]));
* assert_eq!(scalar.index_by([]).get([]), 16);
* ```
*
* Note: due to limitations in Rust's const generics support, TensorIndex only implements TensorRef
* for D from `1` to `6`.

*/
#[derive(Clone, Debug)]
pub struct TensorIndex<T, S, const D: usize, const I: usize> {
    source: S,
    provided: [Option<usize>; D],
    _type: PhantomData<T>,
}

impl<T, S, const D: usize, const I: usize> TensorIndex<T, S, D, I>
where
    S: TensorRef<T, D>,
{
    /**
     * Creates a TensorIndex from a source and a list of provided dimension name/index pairs.
     *
     * The corresponding dimensions in the source will be masked to always return the provided
     * index. Henece, a matrix can be viewed as a vector if you provide one of the row/column
     * index to use. More generally, the tensor the TensorIndex exposes will have a dimensionality
     * of D - I, where D is the dimensionality of the source, and I is the dimensionality of the
     * provided indexes.
     *
     * # Panics
     *
     * - If any provided index is for a dimension that does not exist in the source's shape.
     * - If any provided index is not within range for the length of the dimension.
     * - If multiple indexes are provided for the same dimension.
     */
    #[track_caller]
    pub fn from(source: S, provided_indexes: [(Dimension, usize); I]) -> TensorIndex<T, S, D, I> {
        let shape = source.view_shape();
        if I > D {
            panic!("D - I must be >= 0, D: {:?}, I: {:?}", D, I);
        }
        if crate::tensors::dimensions::has_duplicates(&provided_indexes) {
            panic!(
                "Multiple indexes cannot be provided for the same dimension name, provided: {:?}",
                provided_indexes,
            );
        }
        let mut provided = [None; D];
        for (name, index) in &provided_indexes {
            // Every provided index must match a dimension name in the source and be a valid
            // index within the length
            match shape
                .iter()
                .enumerate()
                .find(|(_i, (n, length))| n == name && index < length)
            {
                None => panic!(
                    "Provided indexes must all correspond to valid indexes into the source shape, source shape: {:?}, provided: {:?}",
                    shape, provided_indexes,
                ),
                // Assign the provided index to the matching position of the source
                Some((i, (_n, _length))) => provided[i] = Some(*index),
            }
        }
        TensorIndex {
            source,
            provided,
            _type: PhantomData,
        }
    }

    /**
     * Consumes the TensorIndex, yielding the source it was created from.
     */
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Gives a reference to the TensorIndex's source (in which the data is not reduced in
     * dimensionality).
     */
    // # Safety
    //
    // Giving out a mutable reference to our source could allow it to be changed out from under us
    // and make our provided indexes invalid. However, since the source implements TensorRef
    // interior mutability is not allowed, so we can give out shared references without breaking
    // our own integrity.
    pub fn source_ref(&self) -> &S {
        &self.source
    }
}

macro_rules! tensor_index_ref_impl {
    (unsafe impl TensorRef for TensorIndex $d:literal $i:literal $helper_name:ident) => {
        impl<T, S> TensorIndex<T, S, $d, $i>
        where
            S: TensorRef<T, $d>,
        {
            fn $helper_name(&self, indexes: [usize; $d - $i]) -> Option<[usize; $d]> {
                let mut supplied = indexes.iter();
                // Indexes have to be in the order of our shape, so they must fill in the None
                // slots of our provided array since we created that in the same order as our
                // view_shape
                let mut combined = [0; $d];
                let mut d = 0;
                for provided in self.provided.iter() {
                    combined[d] = match provided {
                        // This error case should never happen but depending on if we're using
                        // this method in an unsafe function or not we may want to handle it
                        // differently
                        None => *supplied.next()?,
                        Some(i) => *i,
                    };
                    d += 1;
                }
                Some(combined)
            }
        }

        // # Safety
        // The source we index into implements TensorRef, and we do not give out any mutable
        // references to it. Since it may not implement interior mutability due to implementing
        // TensorRef, we know that it won't change under us. Since we know it won't change under
        // us, we can rely on the invariants when we created the provided array. The provided
        // array therefore will be in the same order as the source's view_shape. Hence we can
        // index correctly by filling in the None slots of provided with the supplied indexes,
        // which also have to be in order.
        unsafe impl<T, S> TensorRef<T, { $d - $i }> for TensorIndex<T, S, $d, $i>
        where
            S: TensorRef<T, $d>,
        {
            fn get_reference(&self, indexes: [usize; $d - $i]) -> Option<&T> {
                // unwrap because None returns from the helper method are not input error, they
                // should never happen for any input
                self.source
                    .get_reference(self.$helper_name(indexes).unwrap())
            }

            fn view_shape(&self) -> [(Dimension, usize); $d - $i] {
                let shape = self.source.view_shape();
                let mut unprovided = shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| self.provided[*i].is_none())
                    .map(|(_, (name, length))| (*name, *length));
                std::array::from_fn(|_| unprovided.next().unwrap())
            }

            unsafe fn get_reference_unchecked(&self, indexes: [usize; $d - $i]) -> &T {
                unsafe {
                    // TODO: Can we use unwrap_unchecked here?
                    self.source
                        .get_reference_unchecked(self.$helper_name(indexes).unwrap())
                }
            }

            fn data_layout(&self) -> DataLayout<{ $d - $i }> {
                // Our pre provided index means the view shape no longer matches up to a single
                // line of data in memory.
                DataLayout::NonLinear
            }
        }

        unsafe impl<T, S> TensorMut<T, { $d - $i }> for TensorIndex<T, S, $d, $i>
        where
            S: TensorMut<T, $d>,
        {
            fn get_reference_mut(&mut self, indexes: [usize; $d - $i]) -> Option<&mut T> {
                self.source
                    .get_reference_mut(self.$helper_name(indexes).unwrap())
            }

            unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; $d - $i]) -> &mut T {
                unsafe {
                    // TODO: Can we use unwrap_unchecked here?
                    self.source
                        .get_reference_unchecked_mut(self.$helper_name(indexes).unwrap())
                }
            }
        }
    };
}

tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 6 1 compute_select_indexes_6_1);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 6 2 compute_select_indexes_6_2);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 6 3 compute_select_indexes_6_3);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 6 4 compute_select_indexes_6_4);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 6 5 compute_select_indexes_6_5);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 6 6 compute_select_indexes_6_6);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 5 1 compute_select_indexes_5_1);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 5 2 compute_select_indexes_5_2);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 5 3 compute_select_indexes_5_3);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 5 4 compute_select_indexes_5_4);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 5 5 compute_select_indexes_5_5);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 4 1 compute_select_indexes_4_1);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 4 2 compute_select_indexes_4_2);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 4 3 compute_select_indexes_4_3);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 4 4 compute_select_indexes_4_4);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 3 1 compute_select_indexes_3_1);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 3 2 compute_select_indexes_3_2);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 3 3 compute_select_indexes_3_3);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 2 1 compute_select_indexes_2_1);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 2 2 compute_select_indexes_2_2);
tensor_index_ref_impl!(unsafe impl TensorRef for TensorIndex 1 1 compute_select_indexes_1_1);

#[test]
fn dimensionality_reduction() {
    use crate::tensors::Tensor;
    use crate::tensors::views::TensorView;
    #[rustfmt::skip]
    let tensor = Tensor::from([("batch", 2), ("row", 2), ("column", 2)], vec![
        0, 1,
        2, 3,

        4, 5,
        6, 7
    ]);
    // selects second 2x2
    let matrix = TensorView::from(TensorIndex::from(&tensor, [("batch", 1)]));
    assert_eq!(matrix.shape(), [("row", 2), ("column", 2)]);
    assert_eq!(
        matrix,
        Tensor::from([("row", 2), ("column", 2)], vec![4, 5, 6, 7])
    );
    // selects first column
    let vector = TensorView::from(TensorIndex::from(matrix.source(), [("column", 0)]));
    assert_eq!(vector.shape(), [("row", 2)]);
    assert_eq!(vector, Tensor::from([("row", 2)], vec![4, 6]));
    // equivalent to selecting both together
    let vector = TensorView::from(TensorIndex::from(&tensor, [("batch", 1), ("column", 0)]));
    assert_eq!(vector.shape(), [("row", 2)]);
    assert_eq!(vector, Tensor::from([("row", 2)], vec![4, 6]));

    // selects second row of data
    let matrix = TensorView::from(TensorIndex::from(&tensor, [("row", 1)]));
    assert_eq!(matrix.shape(), [("batch", 2), ("column", 2)]);
    assert_eq!(
        matrix,
        Tensor::from([("batch", 2), ("column", 2)], vec![2, 3, 6, 7])
    );

    // selects second column of data
    let matrix = TensorView::from(TensorIndex::from(&tensor, [("column", 1)]));
    assert_eq!(matrix.shape(), [("batch", 2), ("row", 2)]);
    assert_eq!(
        matrix,
        Tensor::from([("batch", 2), ("row", 2)], vec![1, 3, 5, 7])
    );
    // selects first batch
    let vector = TensorView::from(TensorIndex::from(matrix.source(), [("batch", 0)]));
    assert_eq!(vector.shape(), [("row", 2)]);
    assert_eq!(vector, Tensor::from([("row", 2)], vec![1, 3,]));
    // equivalent to selecting both together
    let vector = TensorView::from(TensorIndex::from(&tensor, [("batch", 0), ("column", 1)]));
    assert_eq!(vector.shape(), [("row", 2)]);
    assert_eq!(vector, Tensor::from([("row", 2)], vec![1, 3,]));
}

/**
 * A combination of dimension names and a tensor. The provided dimensions increase
 * the dimensionality of the TensorRef exposed to more than the dimensionality of the TensorRef
 * this is created from, by adding additional dimensions with a length of one.
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * use easy_ml::tensors::views::{TensorView, TensorExpansion};
 * let vector = Tensor::from([("a", 2)], vec![ 16, 8 ]);
 * let matrix = vector.expand([(1, "b")]);
 * let also_matrix = TensorView::from(TensorExpansion::from(&vector, [(1, "b")]));
 * assert_eq!(matrix, also_matrix);
 * assert_eq!(matrix, Tensor::from([("a", 2), ("b", 1)], vec![ 16, 8 ]));
 * ```
 *
 * Note: due to limitations in Rust's const generics support, TensorExpansion only implements
 * TensorRef for D from `1` to `6`.
 */
#[derive(Clone, Debug)]
pub struct TensorExpansion<T, S, const D: usize, const I: usize> {
    source: S,
    extra: [(usize, Dimension); I],
    _type: PhantomData<T>,
}

impl<T, S, const D: usize, const I: usize> TensorExpansion<T, S, D, I>
where
    S: TensorRef<T, D>,
{
    /**
     * Creates a TensorExpansion from a source and extra dimension names inserted into the shape
     * at the provided indexes.
     *
     * Each extra dimension name adds a dimension to the tensor with a length of 1 so they
     * do not change the total number of elements. Hence, a vector can be viewed as a matrix
     * if you provide an extra row/column dimension. More generally, the tensor the TensorExpansion
     * exposes will have a dimensionality of D + I, where D is the dimensionality of the source,
     * and I is the dimensionality of the extra dimensions.
     *
     * The extra dimension names can be added before any dimensions in the source's shape, in
     * the range 0 inclusive to D inclusive. It is possible to add multiple dimension names before
     * an existing dimension.
     *
     * # Panics
     *
     * - If any extra dimension name is already in use
     * - If any dimension number `d` to insert an extra dimension name into is not 0 <= `d` <= D
     * - If the extra dimension names are not unique
     */
    #[track_caller]
    pub fn from(
        source: S,
        extra_dimension_names: [(usize, Dimension); I],
    ) -> TensorExpansion<T, S, D, I> {
        let mut dimensions = extra_dimension_names;
        if crate::tensors::dimensions::has_duplicates_extra_names(&extra_dimension_names) {
            panic!("All extra dimension names {:?} must be unique", dimensions,);
        }
        let shape = source.view_shape();
        for &(d, name) in &dimensions {
            if d > D {
                panic!(
                    "All extra dimensions {:?} must be inserted in the range 0 <= d <= D of the source shape {:?}",
                    dimensions, shape
                );
            }
            for &(n, _) in &shape {
                if name == n {
                    panic!(
                        "All extra dimension names {:?} must not be already present in the source shape {:?}",
                        dimensions, shape
                    );
                }
            }
        }

        // Sort by ascending insertion positions
        dimensions.sort_by(|a, b| a.0.cmp(&b.0));

        TensorExpansion {
            source,
            extra: dimensions,
            _type: PhantomData,
        }
    }

    /**
     * Consumes the TensorExpansion, yielding the source it was created from.
     */
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Gives a reference to the TensorExpansion's source (in which the data is not increased in
     * dimensionality).
     */
    // # Safety
    //
    // Giving out a mutable reference to our source could allow it to be changed out from under us
    // and make our extra dimensions invalid. However, since the source implements TensorRef
    // interior mutability is not allowed, so we can give out shared references without breaking
    // our own integrity.
    pub fn source_ref(&self) -> &S {
        &self.source
    }
}

macro_rules! tensor_expansion_ref_impl {
    (unsafe impl TensorRef for TensorExpansion $d:literal $i:literal $helper_name:ident) => {
        impl<T, S> TensorExpansion<T, S, $d, $i>
        where
            S: TensorRef<T, $d>,
        {
            fn $helper_name(&self, indexes: [usize; $d + $i]) -> Option<[usize; $d]> {
                let mut used = [0; $d];
                let mut i = 0; // index from 0..D
                let mut extra = 0; // index from 0..I
                for &index in indexes.iter() {
                    match self.extra.get(extra) {
                        // Simple case is when we've already matched against each index for the
                        // extra dimensions, in which case the rest of the provided indexes will
                        // be used on the source.
                        None => {
                            used[i] = index;
                            i += 1;
                        }
                        Some((j, _name)) => {
                            if *j == i {
                                // The i'th actual dimension in our source is preceeded by this
                                // dimension in our extra dimensions.
                                if index != 0 {
                                    // Invalid index
                                    return None;
                                }
                                extra += 1;
                                // Do not increment i, and don't actually index our source with
                                // this index value as it indexes against the extra dimension
                            } else {
                                used[i] = index;
                                i += 1;
                            }
                        }
                    }
                }
                // Now we've filtered out the indexes that were for extra dimensions from the
                // input, we can index our source with the real indexes.
                Some(used)
            }
        }

        unsafe impl<T, S> TensorRef<T, { $d + $i }> for TensorExpansion<T, S, $d, $i>
        where
            S: TensorRef<T, $d>,
        {
            fn get_reference(&self, indexes: [usize; $d + $i]) -> Option<&T> {
                self.source.get_reference(self.$helper_name(indexes)?)
            }

            fn view_shape(&self) -> [(Dimension, usize); $d + $i] {
                let shape = self.source.view_shape();
                let mut extra_shape = [("", 0); $d + $i];
                let mut i = 0; // index from 0..D
                let mut extra = 0; // index from 0..I
                for dimension in extra_shape.iter_mut() {
                    match self.extra.get(extra) {
                        None => {
                            *dimension = shape[i];
                            i += 1;
                        }
                        Some((j, extra_name)) => {
                            if *j == i {
                                *dimension = (extra_name, 1);
                                extra += 1;
                                // Do not increment i, this was an extra dimension
                            } else {
                                *dimension = shape[i];
                                i += 1;
                            }
                        }
                    }
                }
                extra_shape
            }

            unsafe fn get_reference_unchecked(&self, indexes: [usize; $d + $i]) -> &T {
                unsafe {
                    // TODO: Can we use unwrap_unchecked here?
                    self.source
                        .get_reference_unchecked(self.$helper_name(indexes).unwrap())
                }
            }

            fn data_layout(&self) -> DataLayout<{ $d + $i }> {
                // Our extra dimensions means the view shape no longer matches up to a single
                // line of data in memory.
                DataLayout::NonLinear
            }
        }

        unsafe impl<T, S> TensorMut<T, { $d + $i }> for TensorExpansion<T, S, $d, $i>
        where
            S: TensorMut<T, $d>,
        {
            fn get_reference_mut(&mut self, indexes: [usize; $d + $i]) -> Option<&mut T> {
                self.source.get_reference_mut(self.$helper_name(indexes)?)
            }

            unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; $d + $i]) -> &mut T {
                unsafe {
                    // TODO: Can we use unwrap_unchecked here?
                    self.source
                        .get_reference_unchecked_mut(self.$helper_name(indexes).unwrap())
                }
            }
        }
    };
}

tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 0 1 compute_expansion_indexes_0_1);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 0 2 compute_expansion_indexes_0_2);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 0 3 compute_expansion_indexes_0_3);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 0 4 compute_expansion_indexes_0_4);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 0 5 compute_expansion_indexes_0_5);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 0 6 compute_expansion_indexes_0_6);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 1 1 compute_expansion_indexes_1_1);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 1 2 compute_expansion_indexes_1_2);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 1 3 compute_expansion_indexes_1_3);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 1 4 compute_expansion_indexes_1_4);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 1 5 compute_expansion_indexes_1_5);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 2 1 compute_expansion_indexes_2_1);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 2 2 compute_expansion_indexes_2_2);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 2 3 compute_expansion_indexes_2_3);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 2 4 compute_expansion_indexes_2_4);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 3 1 compute_expansion_indexes_3_1);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 3 2 compute_expansion_indexes_3_2);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 3 3 compute_expansion_indexes_3_3);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 4 1 compute_expansion_indexes_4_1);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 4 2 compute_expansion_indexes_4_2);
tensor_expansion_ref_impl!(unsafe impl TensorRef for TensorExpansion 5 1 compute_expansion_indexes_5_1);

#[test]
fn dimensionality_expansion() {
    use crate::tensors::Tensor;
    use crate::tensors::views::TensorView;
    let tensor = Tensor::from([("row", 2), ("column", 2)], (0..4).collect());
    let tensor_3 = TensorView::from(TensorExpansion::from(&tensor, [(0, "batch")]));
    assert_eq!(tensor_3.shape(), [("batch", 1), ("row", 2), ("column", 2)]);
    assert_eq!(
        tensor_3,
        Tensor::from([("batch", 1), ("row", 2), ("column", 2)], vec![0, 1, 2, 3,])
    );
    let vector = Tensor::from([("a", 5)], (0..5).collect());
    let tensor = TensorView::from(TensorExpansion::from(&vector, [(1, "b"), (1, "c")]));
    assert_eq!(
        tensor,
        Tensor::from([("a", 5), ("b", 1), ("c", 1)], (0..5).collect())
    );
    let matrix = Tensor::from([("row", 2), ("column", 2)], (0..4).collect());
    let dataset = TensorView::from(TensorExpansion::from(&matrix, [(2, "color"), (0, "batch")]));
    assert_eq!(
        dataset,
        Tensor::from(
            [("batch", 1), ("row", 2), ("column", 2), ("color", 1)],
            (0..4).collect()
        )
    );
}

#[test]
#[should_panic(
    expected = "Unable to index with [2, 2, 2, 2], Tensor dimensions are [(\"a\", 2), (\"b\", 2), (\"c\", 1), (\"d\", 2)]."
)]
fn dimensionality_reduction_invalid_extra_index() {
    use crate::tensors::Tensor;
    use crate::tensors::views::TensorView;
    let tensor = Tensor::from([("a", 2), ("b", 2), ("d", 2)], (0..8).collect());
    let tensor = TensorView::from(TensorExpansion::from(&tensor, [(2, "c")]));
    assert_eq!(tensor.shape(), [("a", 2), ("b", 2), ("c", 1), ("d", 2)]);
    tensor.index().get_ref([2, 2, 2, 2]);
}
