#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(128) void k_matmul_819b4f(const __half* x1, const __half* x0, __half* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 1024;
        int a1_b = (_gid / 128) % 8;
        int a0_u = (_gid / 16) % 8;
        int a1_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        __shared__ __align__(16) __half _a_smem[1024];
        __shared__ __align__(16) __half _b_smem[2048];
        for (int _ks = 0; _ks < 128; _ks += 128) {
            for (int _fa = a0_u * 16 + a1_u; _fa < 128; _fa += 128) {
                uint4 _v__a_copy0 = *reinterpret_cast<const uint4*>(&x0[(a0_b * 8 + _fa / 16) * 128 + (_ks + 8 * (_fa % 16))]);
                const __half* _v__a_copy0_h = reinterpret_cast<const __half*>(&_v__a_copy0);
                __half _a_copy0 = _v__a_copy0_h[0];
                __half _a_copy1 = _v__a_copy0_h[1];
                __half _a_copy2 = _v__a_copy0_h[2];
                __half _a_copy3 = _v__a_copy0_h[3];
                __half _a_copy4 = _v__a_copy0_h[4];
                __half _a_copy5 = _v__a_copy0_h[5];
                __half _a_copy6 = _v__a_copy0_h[6];
                __half _a_copy7 = _v__a_copy0_h[7];
                __half _vs__a_copy0_3a489368[8] = { _a_copy0, _a_copy1, _a_copy2, _a_copy3, _a_copy4, _a_copy5, _a_copy6, _a_copy7 };
                *reinterpret_cast<uint4*>(&_a_smem[_fa * 8]) = *reinterpret_cast<const uint4*>(_vs__a_copy0_3a489368);
            }
            for (int _fb = a0_u * 16 + a1_u; _fb < 256; _fb += 128) {
                uint4 _v__b_copy0 = *reinterpret_cast<const uint4*>(&x1[(_ks + _fb / 2) * 128 + (a1_b * 16 + 8 * (_fb % 2))]);
                const __half* _v__b_copy0_h = reinterpret_cast<const __half*>(&_v__b_copy0);
                __half _b_copy0 = _v__b_copy0_h[0];
                __half _b_copy1 = _v__b_copy0_h[1];
                __half _b_copy2 = _v__b_copy0_h[2];
                __half _b_copy3 = _v__b_copy0_h[3];
                __half _b_copy4 = _v__b_copy0_h[4];
                __half _b_copy5 = _v__b_copy0_h[5];
                __half _b_copy6 = _v__b_copy0_h[6];
                __half _b_copy7 = _v__b_copy0_h[7];
                __half _vs__b_copy0_f3cd8a12[8] = { _b_copy0, _b_copy1, _b_copy2, _b_copy3, _b_copy4, _b_copy5, _b_copy6, _b_copy7 };
                *reinterpret_cast<uint4*>(&_b_smem[_fb * 8]) = *reinterpret_cast<const uint4*>(_vs__b_copy0_f3cd8a12);
            }
            __syncthreads();
            for (int _ki = 0; _ki < 128; _ki++) {
                __half in1__c0_0 = _b_smem[_ki * 16 + (a1_b * 16 + a1_u - a1_b * 16)];
                __half in0__c0_0 = _a_smem[(a0_b * 8 + a0_u - a0_b * 8) * 128 + _ki];
                float acc0__v__c0_0 = __half2float(in1__c0_0) * __half2float(in0__c0_0);
                acc0__c0_0 += acc0__v__c0_0;
            }
            __syncthreads();
        }
        matmul[(a0_b * 8 + a0_u) * 128 + (a1_b * 16 + a1_u)] = __float2half(acc0__c0_0);
}
