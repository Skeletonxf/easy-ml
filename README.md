# Easy ML

A completely deterministic matrix and linear algebra library over generic number types aimed at being easy to use.

## Usage

Add `easy-ml = "1"` to your `[dependencies]`<sup>1</sup>.

## Overview
This is a pure Rust library which makes heavy use of passing closures, iterators, generic types, and other rust idioms that machine learning libraries which wrap around another language backend could never provide easily. This library tries to provide adequate documentation to explain what the functions compute, what the computations mean, and examples of tasks you might want to do with this library:

- Linear Regression
- k-means Clustering
- Logistic Regression
- Naïve Bayes
- Feedforward neural networks
- using a custom numeric type such as `num_bigint::BigInt`

This library is not designed for deep learning. I have not tried to optimise much of anything. The implementations of everything are more or less textbook mathematical definitions. You might find that you need to use a faster library once you've explored your problem, or that you need something that's not implemented here and have to switch. I hope that being able to at least start here may be of use.

## State of library

This library is currently usable for simple linear algebra tasks like linear regression and for storing 2 dimensional data. This library also has support for forward and reverse automatic differentiation. Currently there is no support for 3d or higher dimensional data, though matrices can be nested into each other.

## Level of abstraction

Where as other machine learning libraries often create objects/structs to represent algorithms like linear regression or k-means, Easy ML instead only represents the data in structs and the consuming code determines all of the control flow. While this may take more effort to write for consuming code initially making changes to the algorithm is much easier.

## Roadmap

- Implement more linear algebra, examples and supporting functions
  - computer vision with mnist dataset
  - bayesian regression WIP

Planned but unlikely to happen any time soon:

- Views and slicing on matrices and named tensors to cut down on copies and looping in consuming code
- Const generics for matrices, records, (and named tensors?) to move a lot of runtime errors into compile time
- Named tensors that will support arbitary numbers of dimensions
- Eigenvalue decomposition and PCA support

*****

1 - If you need to freeze your rust compiler version you should specify a minor version such as `easy-ml = "1.3"`. Easy ML will not introduce breaking API changes between minor versions, but does follow the latest stable version of rust, and thus may introduce dependencies on newer language features (eg const generics) in minor version updates.
