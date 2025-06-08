# OpenFHE-NumPy API Summary

## Core Matrix Operations

| Operation | Description | Syntax | Implemented | Tested |
|-----------|-------------|--------|------------|--------|
| `add` | Element-wise addition | `add(a, b)` | [ ] | [ ] |
| `subtract` | Element-wise subtraction | `subtract(a, b)` | [ ] | [ ] |
| `multiply` | Element-wise multiplication | `multiply(a, b)` | [ ] | [ ] |
| `dot` | Dot product for vectors, matrix multiplication for 2D arrays | `dot(a, b)` | [ ] | [ ] |
| `matmul` | Matrix multiplication | `matmul(a, b)` | [ ] | [ ] |
| `power` | Element-wise power (integer exponents only) | `power(a, exponent)` | [ ] | [ ] |
| `transpose` | Matrix transposition | `transpose(a)` | [ ] | [ ] |
| `cumsum` | Cumulative sum along an axis | `cumsum(a, axis=0, keepdims=False)` | [ ] | [ ] |
| `cumreduce` | Cumulative reduction along an axis | `cumreduce(a, axis=0, keepdims=False)` | [ ] | [ ] |

## Planned Array Creation Functions

| Operation | Description | Syntax | Implemented | Tested |
|-----------|-------------|--------|------------|--------|
| `zeros` | Create an encrypted array of zeros | `zeros(shape, crypto_context, key)` | [ ] | [ ] |
| `ones` | Create an encrypted array of ones | `ones(shape, crypto_context, key)` | [ ] | [ ] |
| `eye` | Create an encrypted identity matrix | `eye(n, crypto_context, key)` | [ ] | [ ] |

## Planned Array Manipulation Functions

| Operation | Description | Syntax | Implemented | Tested |
|-----------|-------------|--------|------------|--------|
| `reshape` | Reshape tensor to new dimensions | `reshape(tensor, new_shape)` | [ ] | [ ] |
| `concat` | Concatenate tensors along specified axis | `concat(tensors, axis=0)` | [ ] | [ ] |

## API Design

- **NumPy Compatibility**: Functions follow NumPy naming conventions and similar signatures
- **Type Annotations**: Uses Python type hints with `ArrayLike` for function parameters
- **Function Dispatching**: Operations use a `tensor_function_api` decorator to handle different tensor types
- **Binary vs. Unary Operations**: API distinguishes between binary operations (requiring two operands) and unary operations

## Implementation Progress

- Total operations: 14
- Implemented: 0/14 (0%)
- Tested: 0/14 (0%)

This checklist can be used to track implementation and testing progress as the library develops.