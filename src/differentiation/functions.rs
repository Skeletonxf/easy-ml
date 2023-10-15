use crate::differentiation::Primitive;
use crate::numeric::extra::{Real, RealRef};
use crate::numeric::{Numeric, NumericRef};

use std::marker::PhantomData;

pub trait FunctionDerivative<T> {
    fn function(x: T, y: T) -> T;
    fn d_function_dx(x: T, y: T) -> T;
    fn d_function_dy(x: T, y: T) -> T;
}

pub trait UnaryFunctionDerivative<T> {
    fn function(x: T) -> T;
    fn d_function_dx(x: T) -> T;
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

pub struct Negation<T> {
    _type: PhantomData<T>,
}

impl<T> UnaryFunctionDerivative<T> for Negation<T>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    /// `x - y`
    fn function(x: T) -> T {
        -x
    }

    /// `d(-x) / dx = -1` (same as `d(x - y) / dy for x = 0`)
    fn d_function_dx(_x: T) -> T {
        -T::one()
    }
}

pub struct Sine<T> {
    _type: PhantomData<T>,
}

impl<T> UnaryFunctionDerivative<T> for Sine<T>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    /// `sin(x)`
    fn function(x: T) -> T {
        x.sin()
    }

    /// `d(sin(x)) / dx = cos(x)`
    fn d_function_dx(x: T) -> T {
        x.cos()
    }
}

pub struct Cosine<T> {
    _type: PhantomData<T>,
}

impl<T> UnaryFunctionDerivative<T> for Cosine<T>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    /// `cos(x)`
    fn function(x: T) -> T {
        x.cos()
    }

    /// `d(cos(x)) / dx = -sin(x)`
    fn d_function_dx(x: T) -> T {
        -x.sin()
    }
}

pub struct Exponential<T> {
    _type: PhantomData<T>,
}

impl<T> UnaryFunctionDerivative<T> for Exponential<T>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    /// `e^x`
    fn function(x: T) -> T {
        x.exp()
    }

    /// `d(e^x) / dx = e^x (itself)`
    fn d_function_dx(x: T) -> T {
        x.exp()
    }
}

pub struct NaturalLogarithm<T> {
    _type: PhantomData<T>,
}

impl<T> UnaryFunctionDerivative<T> for NaturalLogarithm<T>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    /// `ln(x)`
    fn function(x: T) -> T {
        x.ln()
    }

    /// `d(ln(x)) / dx = 1 / x`
    fn d_function_dx(x: T) -> T {
        T::one() / x
    }
}

pub struct SquareRoot<T> {
    _type: PhantomData<T>,
}

impl<T> UnaryFunctionDerivative<T> for SquareRoot<T>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    /// `sqrt(x)`
    fn function(x: T) -> T {
        x.sqrt()
    }

    /// `d(sqrt(x)) / dx = 1 / (2*sqrt(x))`
    fn d_function_dx(x: T) -> T {
        T::one() / ((T::one() + T::one()) * x.sqrt())
    }
}
