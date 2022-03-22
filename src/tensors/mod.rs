/**
 * Generic n dimensional [named tensors](http://nlp.seas.harvard.edu/NamedTensor).
 *
 * Tensors are generic over some type `T` and some usize `D`. If `T` is [Numeric](super::numeric)
 * then the tensor can be used in a mathematical way. `D` is the number of dimensions in the tensor
 * and a compile time constant.
 */
use crate::tensors::indexing::TensorAccess;
use crate::tensors::views::{TensorIndex, TensorMut, TensorRef, TensorView};

pub mod dimensions;
mod display;
pub mod indexing;
pub mod operations;
pub mod views;

/**
 * Dimension names are represented as static string references.
 *
 * This allows you to use string literals to refer to named dimensions, for example you might want
 * to construct a tensor with a shape of
 * `[("batch", 1000), ("height", 100), ("width", 100), ("rgba", 4)]`.
 *
 * Alternatively you can define the strings once as constants and refer to your dimension
 * names by the constant identifiers.
 *
 * ```
 * const BATCH: &'static str = "batch";
 * const HEIGHT: &'static str = "height";
 * const WIDTH: &'static str = "width";
 * const RGBA: &'static str = "rgba";
 * ```
 *
 * Although `Dimension` is interchangable with `&'static str` as it is just a type alias, Easy ML
 * uses `Dimension` whenever dimension names are expected to distinguish the types from just
 * strings.
 */
type Dimension = &'static str;

/**
 * A [named tensor](http://nlp.seas.harvard.edu/NamedTensor).
 *
 * TODO: Summary
 *
 * See also:
 * - [indexing](indexing)
 */
#[derive(Debug)]
pub struct Tensor<T, const D: usize> {
    data: Vec<T>,
    dimensions: [(Dimension, usize); D],
    strides: [usize; D],
}

impl<T, const D: usize> Tensor<T, D> {
    /**
     * Creates a Tensor with a particular number of dimensions and length in each dimension.
     *
     * The product of the dimension lengths corresponds to the number of elements the Tensor
     * will store. Elements are stored in what would be row major order for a Matrix.
     * Each step in memory through the N dimensions corresponds to incrementing the rightmost
     * index, hence a shape of `[("row", 5), ("column", 5)]` would mean the first 6 elements
     * passed in the Vec would be for (0,0), (0,1), (0,2), (0,3), (0,4), (1,0) and so on to (4,4)
     * for the 25th and final element.
     *
     * # Panics
     *
     * - If the number of provided elements does not match the product of the dimension lengths.
     * - If a dimension name is not unique
     * - If any dimension has 0 elements
     *
     * Note that an empty list for dimensions is valid, and constructs a 0 dimensional tensor with
     * a single element (since the product of an empty list is 1).
     */
    #[track_caller]
    pub fn from(dimensions: [(Dimension, usize); D], data: Vec<T>) -> Self {
        let elements = crate::tensors::dimensions::elements(&dimensions);
        if data.len() != elements {
            panic!(
                "Product of dimension lengths must match size of data. {} != {}",
                elements,
                data.len()
            );
        }
        if has_duplicates(&dimensions) {
            panic!("Dimension names must all be unique: {:?}", &dimensions);
        }
        if dimensions.iter().any(|d| d.1 == 0) {
            panic!("No dimension can have 0 elements: {:?}", &dimensions);
        }

        let strides = compute_strides(&dimensions);
        Tensor {
            data,
            dimensions,
            strides,
        }
    }

    pub fn shape(&self) -> [(Dimension, usize); D] {
        self.dimensions
    }
}

impl<T> Tensor<T, 0> {
    /**
     * Creates a 0 dimensional tensor from some scalar
     */
    pub fn from_scalar(value: T) -> Tensor<T, 0> {
        Tensor {
            data: vec![value],
            dimensions: [],
            strides: [],
        }
    }
}

// # Safety
//
// We promise to never implement interior mutability for Tensor.
/**
 * A Tatrix implements TensorRef.
 */
unsafe impl<T, const D: usize> TensorRef<T, D> for Tensor<T, D> {
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions)?;
        self.data.get(i)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        Tensor::shape(self)
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        // The point of get_reference_unchecked is no bounds checking, and therefore
        // it does not make any sense to just use `unwrap` here. The trait documents that
        // it's undefind behaviour to call this method with an out of bounds index, so we
        // can assume the None case will never happen.
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions).unwrap_unchecked();
        self.data.get_unchecked(i)
    }
}

// # Safety
//
// We promise to never implement interior mutability for Tensor.
unsafe impl<T, const D: usize> TensorMut<T, D> for Tensor<T, D> {
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions)?;
        self.data.get_mut(i)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        // The point of get_reference_unchecked_mut is no bounds checking, and therefore
        // it does not make any sense to just use `unwrap` here. The trait documents that
        // it's undefind behaviour to call this method with an out of bounds index, so we
        // can assume the None case will never happen.
        let i = get_index_direct(&indexes, &self.strides, &self.dimensions).unwrap_unchecked();
        self.data.get_unchecked_mut(i)
    }
}

/**
 * Any tensor of a Displayable type implements Display
 */
impl<T: std::fmt::Display> std::fmt::Display for Tensor<T, 0> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(self, f)
    }
}

/**
 * Any tensor of a Displayable type implements Display
 */
impl<T: std::fmt::Display> std::fmt::Display for Tensor<T, 1> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(self, f)
    }
}

/**
 * Any tensor of a Displayable type implements Display
 */
impl<T: std::fmt::Display> std::fmt::Display for Tensor<T, 2> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(self, f)
    }
}

/**
 * Any tensor of a Displayable type implements Display
 */
impl<T: std::fmt::Display> std::fmt::Display for Tensor<T, 3> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::tensors::display::format_view(self, f)
    }
}

fn compute_strides<const D: usize>(dimensions: &[(Dimension, usize); D]) -> [usize; D] {
    let mut strides = [0; D];
    for d in 0..D {
        strides[d] = dimensions.iter().skip(d + 1).map(|d| d.1).product();
    }
    strides
}

// returns the 1 dimensional index to use to get the requested index into some tensor
#[inline]
fn get_index_direct<const D: usize>(
    // indexes to use
    indexes: &[usize; D],
    // strides for indexing into the tensor
    strides: &[usize; D],
    // dimensions of the tensor to index into
    dimensions: &[(Dimension, usize); D],
) -> Option<usize> {
    let mut index = 0;
    for d in 0..D {
        let n = indexes[d];
        if n >= dimensions[d].1 {
            return None;
        }
        index += n * strides[d];
    }
    Some(index)
}

fn has_duplicates(dimensions: &[(Dimension, usize)]) -> bool {
    for i in 1..dimensions.len() {
        let name = dimensions[i - 1].0;
        if dimensions[i..].iter().any(|d| d.0 == name) {
            return true;
        }
    }
    false
}

fn has(dimensions: &[(Dimension, usize)], name: Dimension) -> bool {
    dimensions.iter().any(|d| d.0 == name)
}

impl<T, const D: usize> Tensor<T, D> {
    pub fn view(&self) -> TensorView<T, &Tensor<T, D>, D> {
        TensorView::from(self)
    }

    pub fn view_mut(&mut self) -> TensorView<T, &mut Tensor<T, D>, D> {
        TensorView::from(self)
    }

    pub fn view_owned(self) -> TensorView<T, Tensor<T, D>, D> {
        TensorView::from(self)
    }

    #[track_caller]
    pub fn get(&self, dimensions: [Dimension; D]) -> TensorAccess<T, &Tensor<T, D>, D> {
        TensorAccess::from(self, dimensions)
    }

    #[track_caller]
    pub fn get_mut(&mut self, dimensions: [Dimension; D]) -> TensorAccess<T, &mut Tensor<T, D>, D> {
        TensorAccess::from(self, dimensions)
    }

    #[track_caller]
    pub fn get_owned(self, dimensions: [Dimension; D]) -> TensorAccess<T, Tensor<T, D>, D> {
        TensorAccess::from(self, dimensions)
    }

    /**
     * Creates a TensorAccess which will index into the dimensions this Tensor was created with
     * in the same order as they were provided. See [TensorAccess::from_source_order].
     */
    pub fn source_order(&self) -> TensorAccess<T, &Tensor<T, D>, D> {
        TensorAccess::from_source_order(self)
    }

    /**
     * Creates a TensorAccess which will index into the dimensions this Tensor was
     * created with in the same order as they were provided. The TensorAccess mutably borrows
     * the Tensor, and can therefore mutate it. See [TensorAccess::from_source_order].
     */
    pub fn source_order_mut(&mut self) -> TensorAccess<T, &mut Tensor<T, D>, D> {
        TensorAccess::from_source_order(self)
    }

    /**
     * Creates a TensorAccess which will index into the dimensions this Tensor was
     * created with in the same order as they were provided. The TensorAccess takes ownership
     * of the Tensor, and can therefore mutate it. See [TensorAccess::from_source_order].
     */
    pub fn source_order_owned(self) -> TensorAccess<T, Tensor<T, D>, D> {
        TensorAccess::from_source_order(self)
    }

    // Non public index order reference iterator since we don't want to expose our implementation
    // details to public API since then we could never change them.
    pub(crate) fn direct_index_order_reference_iter(&self) -> std::slice::Iter<T> {
        self.data.iter()
    }

    /**
     * Renames the dimension names of the tensor without changing the lengths of the dimensions
     * in the tensor or moving any data around.
     *
     * ```
     * use easy_ml::tensors::Tensor;
     * let mut tensor = Tensor::from([("x", 2), ("y", 3)], vec![1, 2, 3, 4, 5, 6]);
     * tensor.rename(["y", "z"]);
     * assert_eq!([("y", 2), ("z", 3)], tensor.shape());
     * ```
     */
    // TODO: View version
    pub fn rename(&mut self, dimensions: [Dimension; D]) {
        for d in 0..D {
            self.dimensions[d].0 = dimensions[d];
        }
    }
}

macro_rules! tensor_select_impl {
    (impl Tensor $d:literal 1) => {
        impl<T> Tensor<T, $d> {
            /**
             * Selects the provided dimension name and index pairs in this Tensor, returning a
             * TensorView which has fewer dimensions than this Tensor, with the removed dimensions
             * always indexed as the provided values.
             *
             * This is a shorthand for manually constructing the TensorView and TensorIndex
             *
             * ```
             * use easy_ml::tensors::Tensor;
             * use easy_ml::tensors::views::{TensorView, TensorIndex};
             * let vector = Tensor::from([("a", 2)], vec![ 16, 8 ]);
             * let scalar = vector.select([("a", 0)]);
             * let also_scalar = TensorView::from(TensorIndex::from(&vector, [("a", 0)]));
             * assert_eq!(scalar.get([]).get([]), also_scalar.get([]).get([]));
             * assert_eq!(scalar.get([]).get([]), 16);
             * ```
             *
             * Note: due to limitations in Rust's const generics support, this method is only
             * implemented for `provided_indexes` of length 1 and `D` from 1 to 6. You can fall
             * back to manual construction to create `TensorIndex`es with multiple provided
             * indexes if you need to reduce dimensionality by more than 1 dimension at a time.
             */
            pub fn select(
                &self,
                provided_indexes: [(Dimension, usize); 1],
            ) -> TensorView<T, TensorIndex<T, &Tensor<T, $d>, $d, 1>, { $d - 1 }> {
                TensorView::from(TensorIndex::from(self, provided_indexes))
            }

            /**
             * Selects the provided dimension name and index pairs in this Tensor, returning a
             * TensorView which has fewer dimensions than this Tensor, with the removed dimensions
             * always indexed as the provided values. The TensorIndex mutably borrows this
             * Tensor, and can therefore mutate it
             *
             * See [select](Tensor::select)
             */
            pub fn select_mut(
                &mut self,
                provided_indexes: [(Dimension, usize); 1],
            ) -> TensorView<T, TensorIndex<T, &mut Tensor<T, $d>, $d, 1>, { $d - 1 }> {
                TensorView::from(TensorIndex::from(self, provided_indexes))
            }

            /**
             * Selects the provided dimension name and index pairs in this Tensor, returning a
             * TensorView which has fewer dimensions than this Tensor, with the removed dimensions
             * always indexed as the provided values. The TensorIndex takes ownership ofthis
             * Tensor, and can therefore mutate it
             *
             * See [select](Tensor::select)
             */
            pub fn select_owned(
                self,
                provided_indexes: [(Dimension, usize); 1],
            ) -> TensorView<T, TensorIndex<T, Tensor<T, $d>, $d, 1>, { $d - 1 }> {
                TensorView::from(TensorIndex::from(self, provided_indexes))
            }
        }
    };
}

tensor_select_impl!(impl Tensor 6 1);
tensor_select_impl!(impl Tensor 5 1);
tensor_select_impl!(impl Tensor 4 1);
tensor_select_impl!(impl Tensor 3 1);
tensor_select_impl!(impl Tensor 2 1);
tensor_select_impl!(impl Tensor 1 1);

impl<T, const D: usize> Tensor<T, D>
where
    T: Clone,
{
    /**
     * Creates a tensor with a particular number of dimensions and length in each dimension
     * with all elements initialised to the provided value.
     *
     * # Panics
     *
     * - If a dimension name is not unique
     * - If any dimension has 0 elements
     */
    #[track_caller]
    pub fn empty(dimensions: [(Dimension, usize); D], value: T) -> Self {
        let elements = crate::tensors::dimensions::elements(&dimensions);
        Tensor::from(dimensions, vec![value; elements])
    }

    // TODO: View version
    /**
     * Returns a new Tensor which has the same data as this tensor, but with the order of the
     * dimensions and corresponding order of data changed.
     *
     * For example, with a `[("row", x), ("column", y)]` tensor you could call
     * `transpose(["y", "x"])` which would return a new tensor where every (y,x) of its data
     * corresponds to (x,y) in the original.
     *
     * This method need not shift *all* the dimensions though, you could also swap the width
     * and height of images in a tensor with a shape of
     * `[("batch", b), ("h", h), ("w", w), ("c", c)]` via `transpose(["batch", "w", "h", "c"])`
     * which would return a new tensor where every (b,w,h,c) of its data corresponds to (b,h,w,c)
     * in the original.
     *
     * # Panics
     *
     * If the set of dimensions in the tensor does not match the set of dimensions provided. The
     * order need not match (and if the order does match, this function is just an expensive
     * clone).
     */
    #[track_caller]
    pub fn transpose(&self, dimensions: [Dimension; D]) -> Tensor<T, D> {
        // TODO: Handle error case, propagate as Dimension names to transpose to must be the same set of dimension names in the tensor
        let transposed_order = TensorAccess::from(&self, dimensions);
        let transposed_shape = transposed_order.shape();
        Tensor::from(
            transposed_shape,
            transposed_order.index_order_iter().collect(),
        )
    }

    /**
     * Edits this tensor to have the same data as before, but with the order of the
     * dimensions and corresponding order of data changed.
     *
     * For example, with a `[("row", x), ("column", y)]` tensor you could call
     * `transpose_mut(["y", "x"])` which would edit the tensor so every (y,x) of its data
     * corresponds to (x,y) before the transposition.
     *
     * The transposition will try to be in place, but this is currently only supported for
     * square tensors with 2 dimensions. Other types of tensors will not be transposed in place.
     *
     * # Panics
     *
     * If the set of dimensions in the tensor does not match the set of dimensions provided. The
     * order need not match (and if the order does match, this function is just an expensive
     * clone).
     */
    #[track_caller]
    pub fn transpose_mut(&mut self, dimensions: [Dimension; D]) {
        use crate::tensors::dimensions::{
            dimension_mapping, dimension_mapping_shape, map_dimensions,
        };
        if D == 2 && crate::tensors::dimensions::is_square(&self.dimensions) {
            // TODO: Handle error case, propagate as Dimension names to transpose to must be the same set of dimension names in the tensor
            let dimension_mapping = dimension_mapping(&self.dimensions, &dimensions).unwrap();

            // Don't actually create an iterator because we need to retain ownership of our
            // data so we can transpose it while iterating.
            let mut indexes = [0; D];
            let mut finished = self
                .get_reference(map_dimensions(&dimension_mapping, &indexes))
                .is_none();
            let shape = dimension_mapping_shape(&self.dimensions, &dimension_mapping);

            while !finished {
                let index = indexes;
                let i = index[0];
                let j = index[1];
                if j >= i {
                    let mapped_index = map_dimensions(&dimension_mapping, &index);
                    // Swap elements from the upper triangle (using index order of the actual tensor's
                    // shape)
                    let temp = self.get_reference(index).unwrap().clone();
                    // tensor[i,j] becomes tensor[mapping(i,j)]
                    *self.get_reference_mut(index).unwrap() =
                        self.get_reference(mapped_index).unwrap().clone();
                    // tensor[mapping(i,j)] becomes tensor[i,j]
                    *self.get_reference_mut(mapped_index).unwrap() = temp;
                    // If the mapping is a noop we've assigned i,j to i,j
                    // If the mapping is i,j -> j,i we've assigned i,j to j,i and j,i to i,j
                }
                crate::tensors::indexing::index_order_iter(&mut finished, &mut indexes, &shape);
            }

            // now update our shape and strides to match
            self.dimensions = shape;
            self.strides = compute_strides(&shape);
        } else {
            // fallback to allocating a new transposed tensor
            let transposed = self.transpose(dimensions);
            self.data = transposed.data;
            self.dimensions = transposed.dimensions;
            self.strides = transposed.strides;
        }
    }

    /**
     * Creates and returns a new tensor with all values from the original with the
     * function applied to each. This can be used to change the type of the tensor
     * such as creating a mask:
     * ```
     * use easy_ml::tensors::Tensor;
     * let x = Tensor::from([("a", 2), ("b", 2)], vec![
     *    0.0, 1.2,
     *    5.8, 6.9
     * ]);
     * let y = x.map(|element| element > 2.0);
     * let result = Tensor::from([("a", 2), ("b", 2)], vec![
     *    false, false,
     *    true, true
     * ]);
     * assert_eq!(&y, &result);
     * ```
     */
    pub fn map<U>(&self, mapping_function: impl Fn(T) -> U) -> Tensor<U, D> {
        let mapped = self
            .data
            .iter()
            .map(|x| mapping_function(x.clone()))
            .collect();
        // We're not changing the shape of the Tensor, so don't need to revalidate
        Tensor {
            data: mapped,
            dimensions: self.dimensions,
            strides: self.strides,
        }
    }

    /**
     * Creates and returns a new tensor with all values from the original and
     * the index of each value mapped by a function.
     */
    pub fn map_with_index<U>(&self, mapping_function: impl Fn([usize; D], T) -> U) -> Tensor<U, D> {
        self.source_order().map_with_index(mapping_function)
    }

    /**
     * Applies a function to all values in the tensor, modifying
     * the tensor in place.
     */
    pub fn map_mut(&mut self, mapping_function: impl Fn(T) -> T) {
        for value in self.data.iter_mut() {
            *value = mapping_function(value.clone());
        }
    }

    /**
     * Applies a function to all values and each value's index in the tensor, modifying
     * the tensor in place.
     */
    pub fn map_mut_with_index(&mut self, mapping_function: impl Fn([usize; D], T) -> T) {
        self.source_order_mut().map_mut_with_index(mapping_function);
    }
}

#[test]
fn indexing_test() {
    let tensor = Tensor::from([("x", 2), ("y", 2)], vec![1, 2, 3, 4]);
    let xy = tensor.get(["x", "y"]);
    let yx = tensor.get(["y", "x"]);
    assert_eq!(xy.get([0, 0]), 1);
    assert_eq!(xy.get([0, 1]), 2);
    assert_eq!(xy.get([1, 0]), 3);
    assert_eq!(xy.get([1, 1]), 4);
    assert_eq!(yx.get([0, 0]), 1);
    assert_eq!(yx.get([0, 1]), 3);
    assert_eq!(yx.get([1, 0]), 2);
    assert_eq!(yx.get([1, 1]), 4);
}

#[test]
#[should_panic]
fn repeated_name() {
    Tensor::from([("x", 2), ("x", 2)], vec![1, 2, 3, 4]);
}

#[test]
#[should_panic]
fn wrong_size() {
    Tensor::from([("x", 2), ("y", 3)], vec![1, 2, 3, 4]);
}

#[test]
#[should_panic]
fn bad_indexing() {
    let tensor = Tensor::from([("x", 2), ("y", 2)], vec![1, 2, 3, 4]);
    tensor.get(["x", "x"]);
}

#[test]
fn transpose() {
    #[rustfmt::skip]
    let tensor = Tensor::from(
        [("x", 3), ("y", 2)], vec![
        0, 1,
        2, 3,
        4, 5
    ]);
    let transposed = tensor.transpose(["y", "x"]);
    assert_eq!(transposed.data, vec![0, 2, 4, 1, 3, 5]);
}

#[test]
#[rustfmt::skip]
fn transpose_more_dimensions() {
    let tensor = Tensor::from(
        [("batch", 2), ("y", 10), ("x", 10), ("color", 1)], vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    let transposed = tensor.transpose(["batch", "x", "y", "color"]);
    assert_eq!(transposed.data, vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
}
