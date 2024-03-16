# Easy ML

A completely deterministic machine learning library providing matrices, named tensors, linear algebra and automatic differentiation over generic number types aimed at being easy to use.

## Usage

Add `easy-ml = "1.10"` to your `[dependencies]`<sup>1</sup>.

## Overview
This is a pure Rust library which makes heavy use of passing closures, iterators, generic types, and other rust idioms that machine learning libraries which wrap around another language backend could never provide easily. This library tries to provide adequate documentation to explain what the functions compute, what the computations mean, and examples of tasks you might want to do with this library:

- Linear Regression
- k-means Clustering
- Logistic Regression
- Naïve Bayes
- Feedforward neural networks
- Backprop with Automatic Differentiation
- using a custom numeric type such as `num_bigint::BigInt`
- Handwritten digit recognition in the browser

This library is not designed for deep learning. The implementations of everything are more or less textbook mathematical definitions, and do not feature extensive optimisation. You might find that you need to use a faster library once you've explored your problem, or that you need something that's not implemented here and have to switch. I hope that being able to at least start here may be of use.

## State of library

Easy ML is currently usable for simple linear algebra tasks like linear regression. Easy ML also has support for storing and manipulating N dimensional data, which can form the basis of many classical machine learning tasks. There is also support for forward and reverse automatic differentiation which can be used to train simple feedforward neural networks. Easy ML supports compilation to Web Assembly.

## Level of abstraction

Where as other machine learning libraries often create objects/structs to represent algorithms like linear regression or k-means, Easy ML instead only represents the data in structs and the consuming code determines all of the control flow. While this may take more effort to write for consuming code initially, it means making changes to the algorithms are much easier.

## Features

- `serde` - Optional, enables [serde](https://crates.io/crates/serde) Serialize and Deserialize implementations for `Matrix` and `Tensor`.

*****

1 - If you need to freeze your rust compiler version you should specify a minor version with a [tilde requirement](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#tilde-requirements) such as `easy-ml = "~1.10"`. Easy ML will not introduce breaking API changes between minor versions, but does follow the latest stable version of rust, and thus may introduce dependencies on newer language features (eg const generics) in minor version updates.
