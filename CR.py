import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from tridiagonal_solving.coefs import generate_coefs

CUDA_SRC = r"""
extern "C" __global__
void cr_solve(const float *a_g, const float *b_g, const float *c_g,
              const float *d_g, float *x_g, int n)
{
    extern __shared__ float s[];
    float *a = s;
    float *b = s + n;
    float *c = s + 2*n;
    float *d = s + 3*n;
    float *x = s + 4*n;

    const int sys = blockIdx.x;               
    const int tid = threadIdx.x;              
    const int nthreads0 = blockDim.x;     //сдвиг по блокам     
    const int base = sys * n;                 

    for (int i = tid; i < n; i += nthreads0) {
        a[i] = a_g[base + i];
        b[i] = b_g[base + i];
        c[i] = c_g[base + i];
        d[i] = d_g[base + i];
        x[i] = 0.0f;
    }
    __syncthreads();


    int stride = 1;
    int numThreads = n/2;
    int steps = 0;
    for (int s = n; s > 2; s >>= 1) { steps++; }

    for (int j = 0; j < steps; ++j) {
        __syncthreads();
        if (tid < numThreads) {
            int delta = stride;
            int i  = (stride<<1) * tid + (stride<<1) - 1;
            int iL = i - delta;                          
            int iR = i + delta;                          
            if (iR >= n) iR = n - 1;                     

            float k1 = a[i] / b[iL];
            float k2 = c[i] / b[iR];

            float ai = -a[iL] * k1;
            float bi =  b[i] - c[iL]*k1 - a[iR]*k2;
            float ci = -c[iR] * k2;
            float di =  d[i] - d[iL]*k1 - d[iR]*k2;

            a[i] = ai; b[i] = bi; c[i] = ci; d[i] = di;
        }
        stride <<= 1;       
        numThreads >>= 1;    
    }

    __syncthreads();


    if (tid < 2) {
        int i1 = stride - 1;      
        int i2 = (stride<<1) - 1;

        float denom = b[i2]*b[i1] - c[i1]*a[i2];
        x[i1] = ( b[i2]*d[i1] - c[i1]*d[i2]) / denom;
        x[i2] = ( d[i2]*b[i1] - d[i1]*a[i2]) / denom;
    }

    __syncthreads();

    numThreads = 2;
    for (int j = 0; j < steps; ++j) {
        int delta = stride >> 1;
        __syncthreads();
        if (tid < numThreads) {
            int i = stride*tid + (stride>>1) - 1;
            if (i == delta - 1) {
                x[i] = (d[i] - c[i]*x[i + delta]) / b[i];
            } else {
                x[i] = (d[i] - a[i]*x[i - delta] - c[i]*x[i + delta]) / b[i];
            }
        }
        stride >>= 1;
        numThreads <<= 1;
    }

    __syncthreads();


    for (int i = tid; i < n; i += nthreads0) {
        x_g[base + i] = x[i];
    }
}
"""


def cr_solve_pycuda(a, b, c, d):
    assert a.dtype == b.dtype == c.dtype == d.dtype == np.float32
    assert a.shape == b.shape == c.shape == d.shape
    num_systems, n = a.shape
    # assert n and (n & (n - 1) == 0), "n должно быть степенью 2"  ?

    mod = SourceModule(CUDA_SRC)
    kern = mod.get_function("cr_solve")

    x = np.empty_like(d)

    def pack(mat): return np.ascontiguousarray(mat.reshape(-1))

    a_g = drv.mem_alloc(a.nbytes);
    drv.memcpy_htod(a_g, pack(a))
    b_g = drv.mem_alloc(b.nbytes);
    drv.memcpy_htod(b_g, pack(b))
    c_g = drv.mem_alloc(c.nbytes);
    drv.memcpy_htod(c_g, pack(c))
    d_g = drv.mem_alloc(d.nbytes);
    drv.memcpy_htod(d_g, pack(d))
    x_g = drv.mem_alloc(d.nbytes)

    block = (max(n // 2, 1), 1, 1)
    grid = (int(num_systems), 1, 1)
    shared_bytes = n * 5 * np.dtype(np.float32).itemsize

    kern(a_g, b_g, c_g, d_g, x_g, np.int32(n),
         block=block, grid=grid, shared=shared_bytes)

    a_g.free()
    b_g.free()
    c_g.free()
    d_g.free()
    x_g.free()

    drv.memcpy_dtoh(x.reshape(-1), x_g)
    return x


if __name__ == "__main__":
    n = 2 << 10

    generate_coefs(n)
    # num_systems = 1
    # rng = np.random.default_rng(0)
    #
    # a = rng.normal(size=(num_systems, n)).astype(np.float32); a[:,0] = 0.0
    # c = rng.normal(size=(num_systems, n)).astype(np.float32); c[:,-1] = 0.0
    # b = (np.abs(a) + np.abs(c) + 1.0).astype(np.float32)  # диагональное преобладание
    # d = rng.normal(size=(num_systems, n)).astype(np.float32)

    a = np.load("coefs/A.npy")
    b = np.load("coefs/B.npy")
    c = np.load("coefs/C.npy")
    d = np.load("coefs/D.npy")

    x_gpu = cr_solve_pycuda(a, b, c, d)

    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if i > 0:   A[i, i - 1] = a[0, i]
        A[i, i] = b[0, i]
        if i < n - 1: A[i, i + 1] = c[0, i]
    x_ref = np.linalg.solve(A, d[0])
    print("max abs err:", np.max(np.abs(x_gpu[0] - x_ref)))
