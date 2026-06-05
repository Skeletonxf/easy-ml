# Easy ML

A completely deterministic machine learning library providing named tensors, matrices, linear algebra and automatic differentiation over generic number types aimed at being easy to use.

## Usage

Add `easy-ml = "2.2"` to your `[dependencies]`<sup>1</sup>.

## Overview
This is a pure Rust library which makes heavy use of passing closures, iterators, generic types, and other rust idioms that machine learning libraries which wrap around another language backend could never provide easily. The implementations of everything are more or less textbook mathematical definitions, and do not feature extensive optimisation. As well as providing lots of APIs to manipulate N dimensional data with tensors and matrices, there are APIs covering linear algebra and gradient descent. Forward and reverse automatic differentiation are both supported. Easy ML supports compilation to Web Assembly.

This library tries to provide adequate documentation to explain what the functions compute, what the computations mean, and examples of tasks you might want to do with this library:

- Backprop with Automatic Differentiation
- Feedforward neural networks
- Linear Regression
- k-means Clustering
- Logistic Regression
- Naïve Bayes
- using a custom numeric type such as `num_bigint::BigInt`
- Handwritten digit recognition in the browser
- Einstein summation notation

## Level of abstraction

Where as other machine learning libraries often create objects/structs to represent algorithms like linear regression or k-means, Easy ML instead only represents the data in structs and the consuming code determines all of the control flow. While this may take more effort to write for consuming code initially, it means making changes to the algorithms are much easier.

The tensors in this library are named (at runtime), and specify their rank (number of dimensions) at compile time. Most APIs let you refer to dimensions by the string names you give when you create a tensor, so `Tensor<f64, 2>` is a different type (due to different number of dimensions) to `Tensor<f64, 3>`, and if one rank 2 tensor is "width" by "height" data it will not be conflated when you operate on it with another tensor that is "height" by "width" data. If you don't need the complexity of tensors there are also dedicated matrix APIs which are represented as two dimensional arrays without any dimension names.

## Features

- `serde` - Optional, enables [serde](https://crates.io/crates/serde) Serialize and Deserialize implementations for `Matrix` and `Tensor`.

*****

1 - If you need to freeze your rust compiler version you should specify a minor version with a [tilde requirement](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#tilde-requirements) such as `easy-ml = "~2.2"`. Easy ML will not introduce breaking API changes between minor versions, but does follow the latest stable version of rust, and thus may introduce dependencies on newer language features (eg const generics) in minor version updates.
