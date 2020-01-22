# Easy ML

A completely deterministic matrix and linear algebra library over generic number types aimed at being easy to use.

## Overview
This is a pure Rust library which makes heavy use of passing closures, iterators, generic types, and other rust idioms that machine learning libraries which wrap around another language could never provide easily. This library tries to provide adequate documentation to explain what the functions compute, what the computations mean, and examples of tasks you might want to do with this library:

- linear regression
- k-means clustering
- using a custom numeric type such as `num_bigint::BigInt`.

This library is not designed for deep learning. I have not tried to optimise much of anything. The implementations of everything are more or less textbook mathematical definitions. You might find that you need to use a faster library once you've explored your problem, or that you need something that's not implemented here and have to switch. I hope that being able to at least start here may be of use.

## State of library

This library is currently usable for simple linear algebra tasks like linear regression and for storing 2 dimensional data. There is no support for 3d or higher dimensional data, though matrices can be nested into each other.

## Roadmap

- Make runtime assert errors for things like mismatched matrices more user friendly
- Implement more linear algebra, examples and supporting functions
  - bayesian regression WIP
  - naive bayes
  - gradient descent
  - computer vision with mnist dataset

Planned but unlikely to happen any time soon:

- Eigenvalue decomposition and PCA support
- Implement named tensors that will support arbitary numbers of dimensions
- Implement views and slicing on matrices and named tensors to cut down on copies and looping in consuming code
- Introduce const generics into matrices (and named tensors?) to move a lot of runtime errors into compile time
