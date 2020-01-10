# Easy ML

A completely deterministic matrix and linear algebra library aimed at being easy to use.

## Overview
This library was born out of a frustration with doing data science in python. Not because the libraries commonly used with python are bad, but because:
- I use them rarely enough I never manage to memorize their abbreviated function names
- I often need to search stackoverflow for how to call their functions because the documentation pages don't provide example code for what I want to do or I can't find them easily by searching
- sometimes I find the huge amount of overloaded arguments confusing to read through on the documentation pages
- and some of the time I would really like to just pass a function over to numpy
- and of course you're reading this too so perhaps you would also prefer a statically typed ecosystem.

This is a pure Rust library which makes heavy use of passing closures, iterators, generic types, and other rust idioms that wrappers around another language could never provide easily. Hopefully you'll find the documentation adequate, not just for finding the right function to call but for understanding what you are doing. The documentation also covers examples of tasks you might want to do with this library: linear regression and using a custom numeric type such as `num_bigint::BigInt`.

This library is not designed for deep learning. I have not tried to optimise much of anything. The implementations of everything are more or less textbook mathematical definitions. You might find that you need to use a faster library once you've explored your problem, or that you need something that's not implemented here and have to switch. I hope that being able to at least start here may be of use.

## State of library

This library is currently usable for simple linear algebra tasks like linear regression and for storing 2 dimensional data. There is no support for 3d or higher dimensional data, though matrices can be nested into each other.

## Roadmap

- Make runtime assert errors for things like mismatched matrices more user friendly
- Implement more linear algebra, examples and supporting functions
  - bayesian regression WIP
  - naive bayes
  - gradient descent
  - k means
  - computer vision with mnist dataset

Planned but unlikely to happen any time soon:

- Eigenvalue decomposition and PCA support
- Implement named tensors that will support arbitary dimensions
- Implement views and slicing on matrices and named tensors to cut down on copies and looping in consuming code
- Introduce const generics into matrices (and named tensors?) to move a lot of runtime errors into compile time

