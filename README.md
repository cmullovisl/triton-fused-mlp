This repository implements a fused 2-layer MLP, as it is commonly used in Transformers.
The implemented compute kernel avoids writing the first-layer activations to
global GPU memory by computing one part of the second layer output and directly
adding it onto the correct block in the final output matrix using the [`atomic_add`](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html)
operator (also see the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd)).

In other words, the `atomic_add` operator allows us to accumulate result blocks in the
back-to-back matrix multiplication directly in global memory (see [triton-b2b-matmul.py](./triton-b2b-matmul.py#L105)).

This implementation reduces memory usage, at the cost of additional memory reads and writes.
The performance strongly depends on the implementation of the `atomic_add`
operator, which might strongly differ between different GPU architectures.

For an explanation of the basic block-based Triton matrix multiplication (which
this implementation is based on) see the [Triton tutorial page](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).

For a sequential, but more readable implementation of the fused back-to-back matrix multiplication in Python/Pytorch, see [ref.py](./ref.py).
