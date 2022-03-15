/*!
# Using trait objects with MatrixViews

It may be desirable to use trait objects for the source type of [`MatrixView`s](MatrixView) as this
can avoid a lot of monomorphisation, though this may trade compile time savings for runtime
performance.

When using a `MatrixView` with a trait object you will likely want to define your own traits
in order to include all the behavior you want such as `Debug` or `Clone` that are not required
by [`MatrixRef`](MatrixRef), [`MatrixMut`](MatrixMut) or
[`NoInteriorMutability`](NoInteriorMutability) but are still useful for the source type of a
MatrixView to implement.
```
use easy_ml::matrices::views::{DataLayout, MatrixView, MatrixRef, MatrixMut, NoInteriorMutability};
use easy_ml::matrices::{Matrix, Row, Column};

// Define a trait object with all the traits desired
// For this example we just use MatrixMut and NoInteriorMutability to keep the code as short as
// possible while still allowing the trait object to be used for mutable iterators by the
// MatrixView
trait MatrixMutNoInteriorMutability<T>: MatrixMut<T> + NoInteriorMutability {}

// We can implement this trait object for any type that implements its supertraits.
impl<T, S> MatrixMutNoInteriorMutability<T> for S where S: MatrixMut<T> + NoInteriorMutability {}

// Likewise, any box of our trait object can implement each of the supertraits
// # Safety
// We know the type we're boxing implements MatrixRef, and we've not changed any of the
// behaviour nor added interior mutability, so we can implement it too by delegating to `T`.
unsafe impl<T> MatrixRef<T> for Box<dyn MatrixMutNoInteriorMutability<T>> {
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        self.as_ref().try_get_reference(row, column)
    }

    fn view_rows(&self) -> Row { self.as_ref().view_rows() }

    fn view_columns(&self) -> Column { self.as_ref().view_columns() }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        self.as_ref().get_reference_unchecked(row, column)
    }

    fn data_layout(&self) -> DataLayout { self.as_ref().data_layout() }
}

// # Safety
// We know the type we're boxing implements MatrixMut, and we've not changed any of the
// behaviour nor added interior mutability, so we can implement it too by delegating to `T`.
unsafe impl<T> MatrixMut<T> for Box<dyn MatrixMutNoInteriorMutability<T>> {
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        self.as_mut().try_get_reference_mut(row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        self.as_mut().get_reference_unchecked_mut(row, column)
    }
}

// # Safety
// We know the type we're boxing implements NoInteriorMutability, and we've not added interior
// mutability, so we can implement it too by delegating to `T`.
unsafe impl<T> NoInteriorMutability for Box<dyn MatrixMutNoInteriorMutability<T>> {}

// Now any box we construct of a `dyn MatrixMutNoInteriorMutability<_>` type will implement all
// of the supertraits of MatrixMutNoInteriorMutability too.

let matrix = Matrix::from_flat_row_major((2, 2), vec![1, 2, 3, 4]);
// Box the concrete Matrix<u32> type and erase the type to a
// Box<dyn MatrixMutNoInteriorMutability<u32>> instead.
let boxed: Box<dyn MatrixMutNoInteriorMutability<u32>> = Box::new(matrix);
let mut view = MatrixView::from(boxed);

// We've thrown away the details of the implementing type, and we can no longer
// use `println!("{:?}", view)` because our trait object doesn't include Debug which MatrixView
// needs for its own Debug implementation, but we can still call methods on the MatrixView
// which require its source to be MatrixMut + NoInteriorMutability.
view.column_major_reference_mut_iter()
    .with_index()
    .for_each(|((_, _), x)| *x += 1);
assert!(Matrix::from_flat_row_major((2, 2), vec![2, 3, 4, 5]) == view);
```

*/

#[allow(unused_imports)] // used in doc links
use crate::matrices::views::{MatrixMut, MatrixRef, MatrixView, NoInteriorMutability};
