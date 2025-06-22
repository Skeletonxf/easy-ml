use crate::matrices::views::{DataLayout, MatrixRef, NoInteriorMutability};
use crate::matrices::{Column, Row};

use std::marker::PhantomData;

/**
 * A combination of a mapping function and a matrix.
 *
 * The provided function lazily transforms the data in the matrix for the MatrixRef
 * implementation.
 */
// TODO: Is there any way this can be extended to work for MatrixMut too without requiring user
// to provide two nearly identical tranformation functions that differ only on & vs &mut inputs
// and outputs?
#[derive(Clone, Debug)]
pub(crate) struct MatrixMap<T, U, S, F> {
    source: S,
    f: F,
    _from: PhantomData<T>,
    _to: PhantomData<U>,
}

impl<T, U, S, F> MatrixMap<T, U, S, F>
where
    S: MatrixRef<T>,
    F: Fn(&T) -> &U,
{
    /**
     * Creates a MatrixMap from a source and a function to lazily transform the data with.
     */
    #[track_caller]
    pub fn from(source: S, f: F) -> MatrixMap<T, U, S, F> {
        MatrixMap {
            source,
            f,
            _from: PhantomData,
            _to: PhantomData,
        }
    }

    /**
     * Consumes the MatrixMap, yielding the source it was created from.
     */
    #[allow(dead_code)]
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Gives a reference to the MatrixMap's source (in which the data is not transformed).
     */
    #[allow(dead_code)]
    pub fn source_ref(&self) -> &S {
        &self.source
    }

    /**
     * Gives a mutable reference to the MatrixMap's source (in which the data is not transformed).
     */
    #[allow(dead_code)]
    pub fn source_ref_mut(&mut self) -> &mut S {
        &mut self.source
    }
}

unsafe impl<T, U, S, F> NoInteriorMutability for MatrixMap<T, U, S, F> where S: NoInteriorMutability {}

// # Safety
//
// Since the MatrixRef we own must implement MatrixRef correctly, so do we by delegating to it,
// as we don't introduce any interior mutability.
unsafe impl<T, U, S, F> MatrixRef<U> for MatrixMap<T, U, S, F>
where
    S: MatrixRef<T>,
    F: Fn(&T) -> &U,
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&U> {
        Some((self.f)(self.source.try_get_reference(row, column)?))
    }

    fn view_rows(&self) -> Row {
        self.source.view_rows()
    }

    fn view_columns(&self) -> Column {
        self.source.view_columns()
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &U {
        unsafe { (self.f)(self.source.get_reference_unchecked(row, column)) }
    }

    fn data_layout(&self) -> DataLayout {
        self.source.data_layout()
    }
}
