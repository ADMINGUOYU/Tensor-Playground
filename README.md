# Tensor-Playground

Experiencing Tensors and Mathematics -- **A purely C++ implementation of the most fundamental aspects of Tensors.**

> [!NOTE]
> We designed it to be a FULL HEADER library (for now).\
> The SIMD library is precompiled and linked to the project. (C language C99 standard)

<!-- How-tos -->
## How to use

> [!TIP]
> We use C++11 standard.\
> **AND** We have provided you the makefile.\
> Simply create a **`main.cpp`** under **project root** (next to Makefile) to get started.

1. Import the tensor library
```c
#include "Tensor/Tensor.hpp"
```

2. Create a Tensor
```cpp
ty::Tensor<> tensor { };  // default type: float
ty::Tensor<float> float_tensor { };
ty::Tensor<double> double_tensor { };
ty::Tensor<int> int_tensor { };
```

3. Set it with a shape and allocate the memory it needs
```cpp
{
  size_t shape[] = { 6, 4, 8 };
  tensor.allocate(shape, sizeof(shape)/sizeof(size_t));
}
// This creates a tensor shaped (6, 4, 8)
// NOTE: { } scoped the 'shape' array
```

4. Fancy indexing using Indexer object
```cpp
{
  // Get the Indexer object (default set to stride 1 and at step 0)
  // The full type of Indexer object is TENSOR_UTILITIES::Indexer
  auto indexer = tensor.get_shape().generate_indexer();
  // Set the Indexer to index along the first axis
  indexer.set_stride(tensor.get_shape().get_stride(0));
  // Iterate the Indexer
  do
  {
    // Set the indexed position with value 8.8
    tensor.set_as(indexer, 8.8);
  } while (indexer.next());
  // NOTE: the Indexer object DOES NOT have a proper 'start state', which means the state is valid at the beginning. Use with care when working with normal while() { } loop.
}
```

5. Permuting a tensor
```cpp
{
  size_t shape[] = { 0, 2, 1 };
  tensor.permute(shape);
  // This permutes the 2nd and 3rd dimension of the tensor.
}
```

6. Printing a tensor
```cpp
// Print the tensor
tensor.print();
// Print the memory buffer info
tensor.get_buffer().print();
```

<!-- Version and release date -->
## [ALPHA] Version 0.1.0 (June 21, 2026)
<!-- Checklist of features -->
### Features
- [x] Create Tensor using raw C++ types (int, float, double). Call member function `allocate()` to actually initialise a tensor with a certain shape
- [x] Copy Tensors (same and different types). Use member function `copy_to()`.
- [x] Basic indexing and data access
- [x] Shape management (reshape, permutate)
- [x] Contiguity operations
- [ ] Improve `contiguous()` function, now it checks equal element number 3 times (call + recursive call + `Shape::viewable_as()`)
- [ ] Slicing / sub-Tensor
- [ ] Parallel management of large tensors (memory operations / indexing)
- [ ] Basic tensor creation helpers (external functions or macros)
- [ ] Basic math libraries

<!-- Components' checklists -->
### Components
#### `namespace ty` ('**T**in**Y**' Tensor)

> [!NOTE]
> We only implement basic Tensor management functionalities here. Extra helpers / Math libraries ... are omitted.

- [ ] Tensor/Tensor.hpp
  - [x] API revised and ready for implementation
  - [x] Added proxies for protected internal functions (`data()` and `set_as()` - flattened index version). **This protection is intended.** Proxies should be used when calling from general (Purely abstract) base class `ty::_Tensor`
  - [ ] Revise and get ready for external libraries (i.e. Math libraries)
- [ ] Tensor/Tensor.tpp
  - [x] Finish sketch implementation of the header (all functions are presumably usable)
  - [x] `print()` function ready
  - [ ] Debug

#### `namespace TENSOR_UTILITIES` (Tensor/Tensor_Utilities)
- [ ] ./Memory
  - [x] Buffer setup/access/delete
  - [x] Buffer copy and move (cross-type)
  - [x] Buffer `append()` and `shrink()` (considered useless?)
  - [x] Buffer debug printout
  - [ ] Promote realloc and better memory allocation efficiency
  - [ ] Support SIMD
  - [ ] Parallelism should be revised
- [ ] ./TensorDescription
  - [x] General Shape class
    - [x] Re-implement using MemoryContainer(Buffer) as backend
    - [x] Simple Broad-casting
    - [x] View generator
    - [x] Interaction with Indexer (see below)
    - [ ] Consider change of backend? MemoryContainer is quite heavy
    - [ ] Contiguity checking is delegated here, efficiency should be improved?
    - [ ] Dimension with size **1** should be revised (design choice)? 
  - [x] Indexer (indexable iterator)
    - [x] Internal state management (getter and setter API)
    - [x] `next()` function to advance iterator
    - [x] `next()` able to reverse traversal

#### SIMD (single instruction, multiple data) - precompiled C library (Tensor/SIMD)
- [x] SIMD copying
- [x] SIMD memory filling