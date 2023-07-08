#!/usr/bin/env python3
import torch

def matmul(A, B, C):
    M, K = A.shape
    K, N = B.shape
    #N, K = C.shape
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    D = torch.zeros(M, K)
    #M, K = D.shape

    # Do in parallel
    for m in range(0, M, BLOCK_SIZE_M):
      # Do in parallel
      for n in range(0, N, BLOCK_SIZE_N):
        acc = torch.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32)
        for k in range(0, K, BLOCK_SIZE_K):
          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
          # BLOCK_SIZE_M x BLOCK_SIZE_N
          acc += torch.matmul(a, b)
          #acc += dot(a, b)
        for k2 in range(0, K, BLOCK_SIZE_K):
          c = C[n : n+BLOCK_SIZE_N, k2 : k2+BLOCK_SIZE_K]
          # BLOCK_SIZE_M x BLOCK_SIZE_N * BLOCK_SIZE_N x BLOCK_SIZE_K
          acc2 = torch.matmul(acc, c)
          D[m : m+BLOCK_SIZE_M, k2 : k2+BLOCK_SIZE_K] += acc2

    return D


if __name__ == "__main__":

        bsz = 32
        size = 1024
        # torch.manual_seed(12321)
        A = torch.randn((bsz, size), device='cpu', dtype=torch.float32)
        B = torch.randn((size, size * 4), device='cpu', dtype=torch.float32)
        C = torch.randn((size * 4, size), device='cpu', dtype=torch.float32)

        triton_output = matmul(A, B, C)
        torch_output = torch.matmul(torch.matmul(A, B), C)

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
