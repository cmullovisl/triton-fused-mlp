#!/usr/bin/env python3
import torch

def matmul(A, B):
    M, K = A.shape
    K, N = B.shape
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    C = torch.empty(M, N)

    # from https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#motivations
    # Do in parallel
    for m in range(0, M, BLOCK_SIZE_M):
      # Do in parallel
      for n in range(0, N, BLOCK_SIZE_N):
        acc = torch.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32)
        for k in range(0, K, BLOCK_SIZE_K):
          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
          #acc += dot(a, b)
          acc += torch.matmul(a, b)
        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc

    return C


if __name__ == "__main__":

        bsz = 32
        size = 1024
        # torch.manual_seed(12321)
        A = torch.randn((bsz, size), device='cpu', dtype=torch.float32)
        B = torch.randn((size, size * 4), device='cpu', dtype=torch.float32)

        triton_output = matmul(A, B)
        torch_output = torch.matmul(A, B)

        print(f"triton_output={triton_output}")
        print(f"torch_output={torch_output}")
        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")

        import numpy as np

        np.testing.assert_allclose(
            triton_output.detach().cpu().numpy(),
            torch_output.detach().cpu().numpy(),
            atol=1e-2, rtol=0)
