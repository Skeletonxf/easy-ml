use crate::tensors::views::{Dimension, TensorRef, TensorView};
use std::marker::PhantomData;

/**
 * A wrapper around a tensor with the const generic parameter type erased.
 *
 * This is rarely useful as the API surface becomes much more difficult to use,
 * but when you need to store tensors of different ranks in the same structure
 * this can be necessary.
 */
pub struct DynamicTensor<T> {
    rank: usize,
    // Erased pointer to the box we allocated on construction
    pointer: *mut (),
    // Pointer to our VTable for all the methods using our pointer
    vtable: *const VTable<T>,
    _type: PhantomData<T>,
}

struct VTable<T> {
    get_dynamic_reference: fn(*const (), indexes: &[usize]) -> Option<&T>,
    get_dynamic_reference_mut: fn(*mut (), indexes: &[usize]) -> Option<&mut T>,
    drop: fn(*mut ()),
}

impl<T> DynamicTensor<T> {
    /**
     * Creates the DynamicTensor from any source type that can be converted to a
     * TensorView, which includes Tensor, &Tensor, &mut Tensor and references
     * to a TensorView.
     */
    pub fn from<S, I, const D: usize>(tensor: I) -> Self
    where
        I: Into<TensorView<T, S, D>>,
        S: TensorRef<T, D>,
    {
        let view = tensor.into();
        let rank = view.shape().len();
        let boxed = Box::new(view);

        // Convert box into raw pointer, we are now responsible for dropping it.
        let pointer = Box::into_raw(boxed) as *mut TensorView<T, S, D> as *mut ();

        let vtable = &VTable::<T> {
            get_dynamic_reference: |pointer, indexes| unsafe {
                let reference = &*(pointer as *const TensorView<T, S, D>);
                // TODO: Actual implementation
                None
            },
            get_dynamic_reference_mut: |pointer, indexes| unsafe {
                let reference_mut = &mut *(pointer as *mut TensorView<T, S, D>);
                // TODO: Actual implementation
                None
            },
            drop: |pointer| unsafe {
                drop(Box::from_raw(pointer as *mut TensorView<T, S, D>));
            },
        };

        DynamicTensor {
            rank,
            pointer,
            vtable,
            _type: PhantomData,
        }
        // WIP: We're going to need to use some unsafe here to create
        // our own VTable and pointers
        // Should be similar to: https://lucumr.pocoo.org/2024/5/16/macro-vtable-magic/
        // except we're not using Arc
    }

    // Returns the vtable as a reference.
    //
    // # Safety
    // This is safe because we always ensure the reference is constructed matching
    // this conversion.
    fn vtable(&self) -> &VTable<T> {
        unsafe { &*(self.vtable) }
    }

    /**
     * Returns the number of dimensions, also known as rank, of this tensor.
     */
    pub fn dimensions_len(&self) -> usize {
        self.rank
    }

    /**
     * Gets the nth dimension name and length of this tensor, if there is one.
     *
     * Dimensions are indexed from 0 inclusive to the number of dimensions exclusive.
     */
    fn get_dimension(&self, index: usize) -> Option<(Dimension, usize)> {
        unimplemented!()
    }

    /**
     * Gets a this tensor's shape if the const parameter matches the number of dimensions
     * in this tensor. Otherwise returns None.
     */
    fn view_shape_dynamic_const<const D: usize>(&self) -> Option<[(Dimension, usize); D]> {
        unimplemented!()
    }

    /**
     * Gets a reference to the value at the index if the index is in range and of a matching
     * length to the number of dimensions in this tensor. Otherwise returns None.
     */
    // TODO: Is this what we want for lifetimes?
    fn get_dynamic_reference<'a, 'b>(&'a self, indexes: &'b [usize]) -> Option<&'a T>
    where
        'b: 'a,
    {
        (self.vtable().get_dynamic_reference)(self.pointer, indexes)
    }

    /**
     * Gets a reference to the value at the index if the index is in range and of a matching
     * length to the number of dimensions in this tensor. Otherwise returns None.
     */
    fn get_dynamic_const_reference<const D: usize>(&self, indexes: [usize; D]) -> Option<&T> {
        unimplemented!()
    }
}

impl<T> Drop for DynamicTensor<T> {
    fn drop(&mut self) {
        (self.vtable().drop)(self.pointer)
    }
}
