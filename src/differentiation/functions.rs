use crate::numeric::{Numeric, NumericRef};
use crate::differentiation::Primitive;

use std::marker::PhantomData;

pub trait FunctionDerivative<T> {
    fn function(x: T, y: T) -> T;
    fn d_function_dx(x: T, y: T) -> T;
    fn d_function_dy(x: T, y: T) -> T;
}

pub struct Addition<T> {
    _type: PhantomData<T>,
}

impl<T> FunctionDerivative<T> for Addition<T>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    /// `x + y`
    fn function(x: T, y: T) -> T {
        x + y
    }

    /// `d(x + y) / dx = 1`
    fn d_function_dx(_x: T, _y: T) -> T {
        T::one()
    }

    /// `d(x + y) / dy = 1`
    fn d_function_dy(_x: T, _y: T) -> T {
        T::one()
    }
}

pub struct Subtraction<T> {
    _type: PhantomData<T>,
}

impl<T> FunctionDerivative<T> for Subtraction<T>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    /// `x - y`
    fn function(x: T, y: T) -> T {
        x - y
    }

    /// `d(x - y) / dx = 1`
    fn d_function_dx(_x: T, _y: T) -> T {
        T::one()
    }

    /// `d(x - y) / dy = -1`
    fn d_function_dy(_x: T, _y: T) -> T {
        -T::one()
    }
}

pub struct Multiplication<T> {
    _type: PhantomData<T>,
}

impl<T> FunctionDerivative<T> for Multiplication<T>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    /// `x * y`
    fn function(x: T, y: T) -> T {
        x * y
    }

    /// `d(x * y) / dx = y`
    fn d_function_dx(_x: T, y: T) -> T {
        y
    }

    /// `d(x * y) / dy = x`
    fn d_function_dy(x: T, _y: T) -> T {
        x
    }
}

pub struct Division<T> {
    _type: PhantomData<T>,
}

impl<T> FunctionDerivative<T> for Division<T>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    /// `x / y`
    fn function(x: T, y: T) -> T {
        x / y
    }

    /// `d(x / y) / dx = 1 / y`
    fn d_function_dx(_x: T, y: T) -> T {
        T::one() / y
    }

    /// `d(x / y) / dy = -x / (y^2)`
    fn d_function_dy(x: T, y: T) -> T {
        -x / (y.clone() * y)
    }
}
