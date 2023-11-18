/*!
 * Other operator related implementations for the differentiation module
 *
 * These implementations are written here but Rust docs will display them on their implemented
 * types.
 */

use crate::differentiation::Primitive;
use std::num::{Saturating, Wrapping};

macro_rules! impl_primitive {
    ($T:tt) => {
        impl Primitive for $T {}
    };
}

impl_primitive!(u8);
impl_primitive!(i8);
impl_primitive!(u16);
impl_primitive!(i16);
impl_primitive!(u32);
impl_primitive!(i32);
impl_primitive!(u64);
impl_primitive!(i64);
impl_primitive!(u128);
impl_primitive!(i128);
impl_primitive!(f32);
impl_primitive!(f64);
impl_primitive!(usize);
impl_primitive!(isize);

impl<T: Primitive> Primitive for Wrapping<T> {}
impl<T: Primitive> Primitive for Saturating<T> {}
