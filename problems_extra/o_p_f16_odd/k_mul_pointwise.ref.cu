#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const __half* x0, const __half* x1, __half* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1001) {
        int a0 = _gid;
        __half in0 = x0[a0];
        __half in1 = x1[a0];
        __half v0 = in0 * in1;
        mul[a0] = v0;
    }
}
