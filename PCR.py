import numpy as np
import math
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from tridiagonal_solving.coefs import generate_coefs

CUDA_SRC = r"""
extern "C" __global__
void pcr_kernel(const float* __restrict__ ga,
                const float* __restrict__ gb,
                const float* __restrict__ gc,
                const float* __restrict__ gd,
                float* __restrict__ gx,
                int n, int num_steps)
{{
    extern __shared__ unsigned char smem_raw[];
    float* a = (float*)smem_raw;
    float* b = a + n;
    float* c = b + n;
    float* d = c + n;
    float* x = d + n;

    int i = threadIdx.x;

    a[i] = ga[i];
    b[i] = gb[i];
    c[i] = gc[i];
    d[i] = gd[i];
    __syncthreads();

    int delta = 1;
    for (int step = 0; step < num_steps; ++step)
    {{
        int iLeft  = i - delta; if (iLeft  < 0)       iLeft  = 0;
        int iRight = i + delta; if (iRight >= n)      iRight = n - 1;

        float tmp1 = a[i] / b[iLeft];
        float tmp2 = c[i] / b[iRight];

        float bNew = b[i] - c[iLeft]  * tmp1 - a[iRight] * tmp2;
        float dNew = d[i] - d[iLeft]  * tmp1 - d[iRight] * tmp2;
        float aNew = -a[iLeft]  * tmp1;
        float cNew = -c[iRight] * tmp2;

        __syncthreads(); 

        b[i] = bNew;
        d[i] = dNew;
        a[i] = aNew;
        c[i] = cNew;

        __syncthreads();

        delta <<= 1; // delta *= 2
    }}

    int half = delta;
    if (i < half)
    {{
        int addr1 = i;
        int addr2 = i + half;

        float tmp = b[addr2] * b[addr1] - c[addr1] * a[addr2];
        x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp;
        x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp;
    }}
    __syncthreads();

    gx[i] = x[i];
}}
"""


def pcr_solve_pycuda(a, b, c, d, dtype=np.float32):
    assert a.dtype == b.dtype == c.dtype == d.dtype == np.float32
    assert a.shape == b.shape == c.shape == d.shape
    num_systems, n = a.shape
    assert n and (n & (n - 1) == 0), "n должно быть степенью 2"

    # Граничные условия
    a[0] = dtype.type(0) if hasattr(dtype, "type") else dtype(0)
    c[-1] = dtype.type(0) if hasattr(dtype, "type") else dtype(0)

    num_steps = int(math.log2(n)) - 1
    if num_steps < 0:
        num_steps = 0

    a_g = drv.mem_alloc(a.nbytes)
    drv.memcpy_htod(a_g, a)
    b_g = drv.mem_alloc(b.nbytes)
    drv.memcpy_htod(b_g, b)
    c_g = drv.mem_alloc(c.nbytes)
    drv.memcpy_htod(c_g, c)
    d_g = drv.mem_alloc(d.nbytes)
    drv.memcpy_htod(d_g, d)
    x_g = drv.mem_alloc(d.nbytes)

    mod = SourceModule(CUDA_SRC)
    kern = mod.get_function("pcr_kernel")

    elem_size = np.dtype(dtype).itemsize
    shared_bytes = 5 * n * elem_size

    block = (n, 1, 1)
    grid = (1, 1, 1)
    kern(a_g, b_g, c_g, d_g, x_g, np.int32(n), np.int32(num_steps),
         block=block, grid=grid, shared=shared_bytes)

    x = np.empty_like(d)
    drv.memcpy_dtoh(x, x_g)

    a_g.free()
    b_g.free()
    c_g.free()
    d_g.free()
    x_g.free()

    return x


if __name__ == "__main__":
    n = 2 << 8

    generate_coefs(n)

    a = np.load("coefs/A.npy")
    b = np.load("coefs/B.npy")
    c = np.load("coefs/C.npy")
    d = np.load("coefs/D.npy")

    x_gpu = pcr_solve_pycuda(a, b, c, d)

    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if i > 0:   A[i, i - 1] = a[0, i]
        A[i, i] = b[0, i]
        if i < n - 1: A[i, i + 1] = c[0, i]
    x_ref = np.linalg.solve(A, d[0])
    print("max abs err:", np.max(np.abs(x_gpu[0] - x_ref)))
