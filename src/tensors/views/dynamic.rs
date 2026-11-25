use crate::tensors::views::{Dimension, TensorRef, TensorView};

/**
 * A wrapper around a tensor with the source and const generic parameter type erased.
 *
 * This is rarely useful as the API surface becomes much more difficult to use,
 * and a lot of information about the source type is lost, but when you need to
 * store read only tensors of different ranks in the same structure this can be
 * necessary.
 */
pub struct DynamicTensor<T> {
    rank: usize,
    // Erased pointer to the box we allocated on construction
    pointer: *mut (),
    // VTable doesn't also need type erasing so we can just store
    // a box here to put it on the heap and not deal with more
    // pointers.
    vtable: Box<VTable<T>>,
}

/// All the functions we need to store for a given pointer to a tensorview
/// NB: In theory we could support mutation too but it would require quite
/// a few hacks to safely restrict it to just source types that implemented
/// TensorMut and I don't think many of the use cases for this tensor actually
/// require mutation so for the initial version I'd just defining read access
/// methods in the VTable to provide dynamic access to the TensorRef impl.
struct VTable<T> {
    // NB: We can't really track lifetimes here, so enforcing lifetimes correctly
    // is the responsibility of the wrapping code. The requirement, as with TensorRef,
    // is that references to the source data borrow from self.
    get_dynamic_reference: fn(*const (), indexes: &[usize]) -> Option<*const T>,
    get_dimension: fn(*const (), index: usize) -> Option<(Dimension, usize)>,
    drop: fn(*mut ()),
}

impl<T> DynamicTensor<T> {
    // WIP: We're going to need to use some unsafe here to create
    // our own VTable and pointers
    // Should be similar to: https://lucumr.pocoo.org/2024/5/16/macro-vtable-magic/
    // except we're not using Arc

    /**
     * Creates the DynamicTensor from any **owned** source type that can be converted to a
     * TensorView. These APIs ordinarily include Tensor, &Tensor, &mut Tensor and references
     * to a TensorView. However, as DynamicTensor erases the source type, the
     * source type must be owned (ie static) or a caller could create a DynamicTensor
     * from a reference source type and drop it too early.
     *
     * In such a scenario, the type system couldn't enforce that the DynamicTensor must live
     * at least as long as the reference it holds, since the reference has been erased from
     * the DynamicTensor type but we still use it to give out references to data from the
     * source in the methods of DynamicTensor, which would then be invalid.
     */
    pub fn from_owned<S, I, const D: usize>(tensor: I) -> Self
    where
        I: Into<TensorView<T, S, D>>,
        S: TensorRef<T, D> + 'static,
    {
        let view = tensor.into();
        let rank = view.shape().len();
        let boxed = Box::new(view);

        // Convert box into raw pointer, we are now responsible for dropping it.
        let pointer = Box::into_raw(boxed) as *mut TensorView<T, S, D> as *mut ();

        let vtable = Box::new(VTable::<T> {
            get_dynamic_reference: |pointer, indexes| {
                let tensor = unsafe { &*(pointer as *const TensorView<T, S, D>) };
                match get_dynamic_reference(tensor, indexes) {
                    Some(reference) => Some(reference as *const T),
                    None => None
                }
            },
            get_dimension: |pointer, index| {
                let tensor = unsafe { &*(pointer as *const TensorView<T, S, D>) };
                tensor.shape().get(index).cloned()
            },
            drop: |pointer| unsafe {
                drop(Box::from_raw(pointer as *mut TensorView<T, S, D>));
            },
        });

        DynamicTensor {
            rank,
            pointer,
            vtable,
        }
    }

    // TODO: Consider adding an unsafe from_unowned that warns the caller they must ensure
    // they drop the DynamicTensor before dropping any borrows in the source type?
    // For inline manipulation where the DynamicTensors aren't getting stored in something
    // that outlives the stack this API should be relatively safe to use correctly?

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
    pub fn get_dimension(&self, index: usize) -> Option<(Dimension, usize)> {
        (self.vtable.get_dimension)(self.pointer, index)
    }

    /**
     * Gets this tensor's shape if the const parameter matches the number of dimensions
     * in this tensor. Otherwise returns None.
     */
    pub fn view_shape_dynamic_const<const D: usize>(&self) -> Option<[(Dimension, usize); D]> {
        if self.rank == D {
            let mut shape = [("", 0); D];
            for (d, x) in shape.iter_mut().enumerate() {
                *x = self.get_dimension(d)?;
            }
            Some(shape)
        } else {
            None
        }
    }

    /**
     * Gets a reference to the value at the index if the index is in range and of a matching
     * length to the number of dimensions in this tensor. Otherwise returns None.
     */
    pub fn get_dynamic_reference<'a>(&'a self, indexes: &[usize]) -> Option<&'a T>
    {
        match (self.vtable.get_dynamic_reference)(self.pointer, indexes) {
            // SAFETY: This is safe because the lifetime we coerce to here is
            // always exactly the same as the lifetime when we obtained the
            // reference in the non method get_dynamic_reference function implementation.
            // We are essentially undoing the pointer casts required to pass it
            // through the erased types of the VTable
            Some(reference) => Some(unsafe { &*reference }),
            None => None,
        }
    }

    /**
     * Gets a reference to the value at the index if the index is in range and of a matching
     * length to the number of dimensions in this tensor. Otherwise returns None.
     */
    pub fn get_dynamic_const_reference<const D: usize>(&self, indexes: [usize; D]) -> Option<&T> {
        self.get_dynamic_reference(&indexes)
    }
}

impl<T> Drop for DynamicTensor<T> {
    fn drop(&mut self) {
        (self.vtable.drop)(self.pointer)
    }
}

fn get_dynamic_reference<'a, T, S, const D: usize>(
    tensor: &'a TensorView<T, S, D>,
    indexes: &[usize],
) -> Option<&'a T>
where
    S: TensorRef<T, D>,
{
    let i: [usize; D] = indexes.try_into().ok()?;
    tensor.source_ref().get_reference(i)
}
