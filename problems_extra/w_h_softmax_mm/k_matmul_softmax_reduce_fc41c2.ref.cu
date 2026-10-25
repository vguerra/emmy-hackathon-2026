#include <cuda_fp16.h>
static __device__ __forceinline__ void emmy_cp_async_cg(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem) : "memory");
}

template <int Bytes>
static __device__ __forceinline__ void emmy_cp_async_ca(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(addr), "l"(gmem), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void emmy_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void emmy_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_fc41c2(const __half* x, const __half* w, __half* softmax) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 512;
        int a5_b = (_gid / 256) % 2;
        int a0_u = (_gid / 32) % 8;
        int a5_u = (_gid) % 32;
        float acc4__c0_0 = 0.0f;
        float acc4__c1_0 = 0.0f;
        float acc4__c2_0 = 0.0f;
        float acc4__c3_0 = 0.0f;
        float acc4__c4_0 = 0.0f;
        float acc4__c5_0 = 0.0f;
        float acc4__c6_0 = 0.0f;
        float acc4__c7_0 = 0.0f;
        float acc4__c8_0 = 0.0f;
        float acc4__c9_0 = 0.0f;
        __shared__ __align__(16) __half _a_smem[10240];
        __shared__ __align__(16) __half _b_smem[4096];
        for (int _ks = 0; _ks < 128; _ks += 128) {
            emmy_cp_async_cg(&_a_smem[0 + ((a0_u * 32 + a5_u) * 8 / 128) * 128 + ((a0_u * 32 + a5_u) * 8 % 128)], &x[((a0_b * 80 + (a0_u * 32 + a5_u) / 16 < 64) ? (a0_b * 80 + (a0_u * 32 + a5_u) / 16) : (63)) * 128 + (_ks + 8 * ((a0_u * 32 + a5_u) % 16))]);
            emmy_cp_async_cg(&_a_smem[2048 + ((a0_u * 32 + a5_u) * 8 / 128) * 128 + ((a0_u * 32 + a5_u) * 8 % 128)], &x[((a0_b * 80 + ((a0_u * 32 + a5_u) / 16 + 16) < 64) ? (a0_b * 80 + ((a0_u * 32 + a5_u) / 16 + 16)) : (63)) * 128 + (_ks + 8 * ((a0_u * 32 + a5_u) % 16))]);
            emmy_cp_async_cg(&_a_smem[4096 + ((a0_u * 32 + a5_u) * 8 / 128) * 128 + ((a0_u * 32 + a5_u) * 8 % 128)], &x[((a0_b * 80 + ((a0_u * 32 + a5_u) / 16 + 32) < 64) ? (a0_b * 80 + ((a0_u * 32 + a5_u) / 16 + 32)) : (63)) * 128 + (_ks + 8 * ((a0_u * 32 + a5_u) % 16))]);
            emmy_cp_async_cg(&_a_smem[6144 + ((a0_u * 32 + a5_u) * 8 / 128) * 128 + ((a0_u * 32 + a5_u) * 8 % 128)], &x[((a0_b * 80 + ((a0_u * 32 + a5_u) / 16 + 48) < 64) ? (a0_b * 80 + ((a0_u * 32 + a5_u) / 16 + 48)) : (63)) * 128 + (_ks + 8 * ((a0_u * 32 + a5_u) % 16))]);
            emmy_cp_async_cg(&_a_smem[8192 + ((a0_u * 32 + a5_u) * 8 / 128) * 128 + ((a0_u * 32 + a5_u) * 8 % 128)], &x[((a0_b * 80 + ((a0_u * 32 + a5_u) / 16 + 64) < 64) ? (a0_b * 80 + ((a0_u * 32 + a5_u) / 16 + 64)) : (63)) * 128 + (_ks + 8 * ((a0_u * 32 + a5_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[0 + ((a0_u * 32 + a5_u) * 8 / 32) * 32 + ((a0_u * 32 + a5_u) * 8 % 32)], &w[(_ks + (a0_u * 32 + a5_u) / 4) * 64 + (a5_b * 32 + 8 * ((a0_u * 32 + a5_u) % 4))]);
            emmy_cp_async_cg(&_b_smem[2048 + ((a0_u * 32 + a5_u) * 8 / 32) * 32 + ((a0_u * 32 + a5_u) * 8 % 32)], &w[(_ks + ((a0_u * 32 + a5_u) / 4 + 64)) * 64 + (a5_b * 32 + 8 * ((a0_u * 32 + a5_u) % 4))]);
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            for (int _ki = 0; _ki < 128; _ki++) {
                __half in4__c0_0 = _b_smem[_ki * 32 + (a5_b * 32 + a5_u - a5_b * 32)];
                __half in5__c0_0 = _a_smem[(a0_b * 80 + a0_u * 10 - a0_b * 80) * 128 + _ki];
                float acc4__v__c0_0 = __half2float(in4__c0_0) * __half2float(in5__c0_0);
                acc4__c0_0 += acc4__v__c0_0;
                __half in5__c1_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 1 - a0_b * 80) * 128 + _ki];
                float acc4__v__c1_0 = __half2float(in4__c0_0) * __half2float(in5__c1_0);
                acc4__c1_0 += acc4__v__c1_0;
                __half in5__c2_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 2 - a0_b * 80) * 128 + _ki];
                float acc4__v__c2_0 = __half2float(in4__c0_0) * __half2float(in5__c2_0);
                acc4__c2_0 += acc4__v__c2_0;
                __half in5__c3_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 3 - a0_b * 80) * 128 + _ki];
                float acc4__v__c3_0 = __half2float(in4__c0_0) * __half2float(in5__c3_0);
                acc4__c3_0 += acc4__v__c3_0;
                __half in5__c4_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 4 - a0_b * 80) * 128 + _ki];
                float acc4__v__c4_0 = __half2float(in4__c0_0) * __half2float(in5__c4_0);
                acc4__c4_0 += acc4__v__c4_0;
                __half in5__c5_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 5 - a0_b * 80) * 128 + _ki];
                float acc4__v__c5_0 = __half2float(in4__c0_0) * __half2float(in5__c5_0);
                acc4__c5_0 += acc4__v__c5_0;
                __half in5__c6_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 6 - a0_b * 80) * 128 + _ki];
                float acc4__v__c6_0 = __half2float(in4__c0_0) * __half2float(in5__c6_0);
                acc4__c6_0 += acc4__v__c6_0;
                __half in5__c7_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 7 - a0_b * 80) * 128 + _ki];
                float acc4__v__c7_0 = __half2float(in4__c0_0) * __half2float(in5__c7_0);
                acc4__c7_0 += acc4__v__c7_0;
                __half in5__c8_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 8 - a0_b * 80) * 128 + _ki];
                float acc4__v__c8_0 = __half2float(in4__c0_0) * __half2float(in5__c8_0);
                acc4__c8_0 += acc4__v__c8_0;
                __half in5__c9_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 9 - a0_b * 80) * 128 + _ki];
                float acc4__v__c9_0 = __half2float(in4__c0_0) * __half2float(in5__c9_0);
                acc4__c9_0 += acc4__v__c9_0;
            }
            __syncthreads();
        }
        float acc3__c0_0 = 0.0f;
        __half acc1__c0_0 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__c0_0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                __half in1__c0_0 = x[(a0_b * 80 + a0_u * 10) % 64 * 128 + a2];
                __half in0__c0_0 = w[a2 * 64 + a1];
                __half v0__c0_0 = in0__c0_0 * in1__c0_0;
                acc0__c0_0 += __half2float(v0__c0_0);
            }
            __half v1__c0_0 = __float2half(acc0__c0_0);
            float one__c0_0 = 1.0f;
            float v1__gn__c0_0 = fmaxf(__half2float(acc1__c0_0), __half2float(v1__c0_0));
            float v1__dg__c0_0 = __half2float(acc1__c0_0) - v1__gn__c0_0;
            float v1__dg_o__c0_0 = __half2float(v1__c0_0) - v1__gn__c0_0;
            float v1__acc3_sa__c0_0 = acc3__c0_0 * expf(v1__dg__c0_0);
            float v1__acc3_sb__c0_0 = one__c0_0 * expf(v1__dg_o__c0_0);
            acc3__c0_0 = v1__acc3_sa__c0_0 + v1__acc3_sb__c0_0;
            acc1__c0_0 = __hmax(acc1__c0_0, v1__c0_0);
        }
        __half v8__c0_0 = __float2half(acc4__c0_0);
        __half v9__c0_0 = v8__c0_0 - acc1__c0_0;
        float v11__c0_0 = __half2float(expf(v9__c0_0)) * (1.0f / acc3__c0_0);
        if (a0_b * 80 + a0_u * 10 + 0 < 64) {
            softmax[(a0_b * 80 + a0_u * 10) % 64 * 64 + (a5_b * 32 + a5_u)] = __float2half(v11__c0_0);
        }
        float acc3__c1_0 = 0.0f;
        __half acc1__c1_0 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__c1_0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                __half in1__c1_0 = x[(a0_b * 80 + a0_u * 10 + 1) % 64 * 128 + a2];
                __half in0__c1_0 = w[a2 * 64 + a1];
                __half v0__c1_0 = in0__c1_0 * in1__c1_0;
                acc0__c1_0 += __half2float(v0__c1_0);
            }
            __half v1__c1_0 = __float2half(acc0__c1_0);
            float one__c1_0 = 1.0f;
            float v1__gn__c1_0 = fmaxf(__half2float(acc1__c1_0), __half2float(v1__c1_0));
            float v1__dg__c1_0 = __half2float(acc1__c1_0) - v1__gn__c1_0;
            float v1__dg_o__c1_0 = __half2float(v1__c1_0) - v1__gn__c1_0;
            float v1__acc3_sa__c1_0 = acc3__c1_0 * expf(v1__dg__c1_0);
            float v1__acc3_sb__c1_0 = one__c1_0 * expf(v1__dg_o__c1_0);
            acc3__c1_0 = v1__acc3_sa__c1_0 + v1__acc3_sb__c1_0;
            acc1__c1_0 = __hmax(acc1__c1_0, v1__c1_0);
        }
        __half v8__c1_0 = __float2half(acc4__c1_0);
        __half v9__c1_0 = v8__c1_0 - acc1__c1_0;
        float v11__c1_0 = __half2float(expf(v9__c1_0)) * (1.0f / acc3__c1_0);
        if (a0_b * 80 + a0_u * 10 + 1 < 64) {
            softmax[(a0_b * 80 + a0_u * 10 + 1) % 64 * 64 + (a5_b * 32 + a5_u)] = __float2half(v11__c1_0);
        }
        float acc3__c2_0 = 0.0f;
        __half acc1__c2_0 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__c2_0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                __half in1__c2_0 = x[(a0_b * 80 + a0_u * 10 + 2) % 64 * 128 + a2];
                __half in0__c2_0 = w[a2 * 64 + a1];
                __half v0__c2_0 = in0__c2_0 * in1__c2_0;
                acc0__c2_0 += __half2float(v0__c2_0);
            }
            __half v1__c2_0 = __float2half(acc0__c2_0);
            float one__c2_0 = 1.0f;
            float v1__gn__c2_0 = fmaxf(__half2float(acc1__c2_0), __half2float(v1__c2_0));
            float v1__dg__c2_0 = __half2float(acc1__c2_0) - v1__gn__c2_0;
            float v1__dg_o__c2_0 = __half2float(v1__c2_0) - v1__gn__c2_0;
            float v1__acc3_sa__c2_0 = acc3__c2_0 * expf(v1__dg__c2_0);
            float v1__acc3_sb__c2_0 = one__c2_0 * expf(v1__dg_o__c2_0);
            acc3__c2_0 = v1__acc3_sa__c2_0 + v1__acc3_sb__c2_0;
            acc1__c2_0 = __hmax(acc1__c2_0, v1__c2_0);
        }
        __half v8__c2_0 = __float2half(acc4__c2_0);
        __half v9__c2_0 = v8__c2_0 - acc1__c2_0;
        float v11__c2_0 = __half2float(expf(v9__c2_0)) * (1.0f / acc3__c2_0);
        if (a0_b * 80 + a0_u * 10 + 2 < 64) {
            softmax[(a0_b * 80 + a0_u * 10 + 2) % 64 * 64 + (a5_b * 32 + a5_u)] = __float2half(v11__c2_0);
        }
        float acc3__c3_0 = 0.0f;
        __half acc1__c3_0 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__c3_0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                __half in1__c3_0 = x[(a0_b * 80 + a0_u * 10 + 3) % 64 * 128 + a2];
                __half in0__c3_0 = w[a2 * 64 + a1];
                __half v0__c3_0 = in0__c3_0 * in1__c3_0;
                acc0__c3_0 += __half2float(v0__c3_0);
            }
            __half v1__c3_0 = __float2half(acc0__c3_0);
            float one__c3_0 = 1.0f;
            float v1__gn__c3_0 = fmaxf(__half2float(acc1__c3_0), __half2float(v1__c3_0));
            float v1__dg__c3_0 = __half2float(acc1__c3_0) - v1__gn__c3_0;
            float v1__dg_o__c3_0 = __half2float(v1__c3_0) - v1__gn__c3_0;
            float v1__acc3_sa__c3_0 = acc3__c3_0 * expf(v1__dg__c3_0);
            float v1__acc3_sb__c3_0 = one__c3_0 * expf(v1__dg_o__c3_0);
            acc3__c3_0 = v1__acc3_sa__c3_0 + v1__acc3_sb__c3_0;
            acc1__c3_0 = __hmax(acc1__c3_0, v1__c3_0);
        }
        __half v8__c3_0 = __float2half(acc4__c3_0);
        __half v9__c3_0 = v8__c3_0 - acc1__c3_0;
        float v11__c3_0 = __half2float(expf(v9__c3_0)) * (1.0f / acc3__c3_0);
        if (a0_b * 80 + a0_u * 10 + 3 < 64) {
            softmax[(a0_b * 80 + a0_u * 10 + 3) % 64 * 64 + (a5_b * 32 + a5_u)] = __float2half(v11__c3_0);
        }
        float acc3__c4_0 = 0.0f;
        __half acc1__c4_0 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__c4_0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                __half in1__c4_0 = x[(a0_b * 80 + a0_u * 10 + 4) % 64 * 128 + a2];
                __half in0__c4_0 = w[a2 * 64 + a1];
                __half v0__c4_0 = in0__c4_0 * in1__c4_0;
                acc0__c4_0 += __half2float(v0__c4_0);
            }
            __half v1__c4_0 = __float2half(acc0__c4_0);
            float one__c4_0 = 1.0f;
            float v1__gn__c4_0 = fmaxf(__half2float(acc1__c4_0), __half2float(v1__c4_0));
            float v1__dg__c4_0 = __half2float(acc1__c4_0) - v1__gn__c4_0;
            float v1__dg_o__c4_0 = __half2float(v1__c4_0) - v1__gn__c4_0;
            float v1__acc3_sa__c4_0 = acc3__c4_0 * expf(v1__dg__c4_0);
            float v1__acc3_sb__c4_0 = one__c4_0 * expf(v1__dg_o__c4_0);
            acc3__c4_0 = v1__acc3_sa__c4_0 + v1__acc3_sb__c4_0;
            acc1__c4_0 = __hmax(acc1__c4_0, v1__c4_0);
        }
        __half v8__c4_0 = __float2half(acc4__c4_0);
        __half v9__c4_0 = v8__c4_0 - acc1__c4_0;
        float v11__c4_0 = __half2float(expf(v9__c4_0)) * (1.0f / acc3__c4_0);
        if (a0_b * 80 + a0_u * 10 + 4 < 64) {
            softmax[(a0_b * 80 + a0_u * 10 + 4) % 64 * 64 + (a5_b * 32 + a5_u)] = __float2half(v11__c4_0);
        }
        float acc3__c5_0 = 0.0f;
        __half acc1__c5_0 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__c5_0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                __half in1__c5_0 = x[(a0_b * 80 + a0_u * 10 + 5) % 64 * 128 + a2];
                __half in0__c5_0 = w[a2 * 64 + a1];
                __half v0__c5_0 = in0__c5_0 * in1__c5_0;
                acc0__c5_0 += __half2float(v0__c5_0);
            }
            __half v1__c5_0 = __float2half(acc0__c5_0);
            float one__c5_0 = 1.0f;
            float v1__gn__c5_0 = fmaxf(__half2float(acc1__c5_0), __half2float(v1__c5_0));
            float v1__dg__c5_0 = __half2float(acc1__c5_0) - v1__gn__c5_0;
            float v1__dg_o__c5_0 = __half2float(v1__c5_0) - v1__gn__c5_0;
            float v1__acc3_sa__c5_0 = acc3__c5_0 * expf(v1__dg__c5_0);
            float v1__acc3_sb__c5_0 = one__c5_0 * expf(v1__dg_o__c5_0);
            acc3__c5_0 = v1__acc3_sa__c5_0 + v1__acc3_sb__c5_0;
            acc1__c5_0 = __hmax(acc1__c5_0, v1__c5_0);
        }
        __half v8__c5_0 = __float2half(acc4__c5_0);
        __half v9__c5_0 = v8__c5_0 - acc1__c5_0;
        float v11__c5_0 = __half2float(expf(v9__c5_0)) * (1.0f / acc3__c5_0);
        if (a0_b * 80 + a0_u * 10 + 5 < 64) {
            softmax[(a0_b * 80 + a0_u * 10 + 5) % 64 * 64 + (a5_b * 32 + a5_u)] = __float2half(v11__c5_0);
        }
        float acc3__c6_0 = 0.0f;
        __half acc1__c6_0 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__c6_0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                __half in1__c6_0 = x[(a0_b * 80 + a0_u * 10 + 6) % 64 * 128 + a2];
                __half in0__c6_0 = w[a2 * 64 + a1];
                __half v0__c6_0 = in0__c6_0 * in1__c6_0;
                acc0__c6_0 += __half2float(v0__c6_0);
            }
            __half v1__c6_0 = __float2half(acc0__c6_0);
            float one__c6_0 = 1.0f;
            float v1__gn__c6_0 = fmaxf(__half2float(acc1__c6_0), __half2float(v1__c6_0));
            float v1__dg__c6_0 = __half2float(acc1__c6_0) - v1__gn__c6_0;
            float v1__dg_o__c6_0 = __half2float(v1__c6_0) - v1__gn__c6_0;
            float v1__acc3_sa__c6_0 = acc3__c6_0 * expf(v1__dg__c6_0);
            float v1__acc3_sb__c6_0 = one__c6_0 * expf(v1__dg_o__c6_0);
            acc3__c6_0 = v1__acc3_sa__c6_0 + v1__acc3_sb__c6_0;
            acc1__c6_0 = __hmax(acc1__c6_0, v1__c6_0);
        }
        __half v8__c6_0 = __float2half(acc4__c6_0);
        __half v9__c6_0 = v8__c6_0 - acc1__c6_0;
        float v11__c6_0 = __half2float(expf(v9__c6_0)) * (1.0f / acc3__c6_0);
        if (a0_b * 80 + a0_u * 10 + 6 < 64) {
            softmax[(a0_b * 80 + a0_u * 10 + 6) % 64 * 64 + (a5_b * 32 + a5_u)] = __float2half(v11__c6_0);
        }
        float acc3__c7_0 = 0.0f;
        __half acc1__c7_0 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__c7_0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                __half in1__c7_0 = x[(a0_b * 80 + a0_u * 10 + 7) % 64 * 128 + a2];
                __half in0__c7_0 = w[a2 * 64 + a1];
                __half v0__c7_0 = in0__c7_0 * in1__c7_0;
                acc0__c7_0 += __half2float(v0__c7_0);
            }
            __half v1__c7_0 = __float2half(acc0__c7_0);
            float one__c7_0 = 1.0f;
            float v1__gn__c7_0 = fmaxf(__half2float(acc1__c7_0), __half2float(v1__c7_0));
            float v1__dg__c7_0 = __half2float(acc1__c7_0) - v1__gn__c7_0;
            float v1__dg_o__c7_0 = __half2float(v1__c7_0) - v1__gn__c7_0;
            float v1__acc3_sa__c7_0 = acc3__c7_0 * expf(v1__dg__c7_0);
            float v1__acc3_sb__c7_0 = one__c7_0 * expf(v1__dg_o__c7_0);
            acc3__c7_0 = v1__acc3_sa__c7_0 + v1__acc3_sb__c7_0;
            acc1__c7_0 = __hmax(acc1__c7_0, v1__c7_0);
        }
        __half v8__c7_0 = __float2half(acc4__c7_0);
        __half v9__c7_0 = v8__c7_0 - acc1__c7_0;
        float v11__c7_0 = __half2float(expf(v9__c7_0)) * (1.0f / acc3__c7_0);
        if (a0_b * 80 + a0_u * 10 + 7 < 64) {
            softmax[(a0_b * 80 + a0_u * 10 + 7) % 64 * 64 + (a5_b * 32 + a5_u)] = __float2half(v11__c7_0);
        }
        float acc3__c8_0 = 0.0f;
        __half acc1__c8_0 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__c8_0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                __half in1__c8_0 = x[(a0_b * 80 + a0_u * 10 + 8) % 64 * 128 + a2];
                __half in0__c8_0 = w[a2 * 64 + a1];
                __half v0__c8_0 = in0__c8_0 * in1__c8_0;
                acc0__c8_0 += __half2float(v0__c8_0);
            }
            __half v1__c8_0 = __float2half(acc0__c8_0);
            float one__c8_0 = 1.0f;
            float v1__gn__c8_0 = fmaxf(__half2float(acc1__c8_0), __half2float(v1__c8_0));
            float v1__dg__c8_0 = __half2float(acc1__c8_0) - v1__gn__c8_0;
            float v1__dg_o__c8_0 = __half2float(v1__c8_0) - v1__gn__c8_0;
            float v1__acc3_sa__c8_0 = acc3__c8_0 * expf(v1__dg__c8_0);
            float v1__acc3_sb__c8_0 = one__c8_0 * expf(v1__dg_o__c8_0);
            acc3__c8_0 = v1__acc3_sa__c8_0 + v1__acc3_sb__c8_0;
            acc1__c8_0 = __hmax(acc1__c8_0, v1__c8_0);
        }
        __half v8__c8_0 = __float2half(acc4__c8_0);
        __half v9__c8_0 = v8__c8_0 - acc1__c8_0;
        float v11__c8_0 = __half2float(expf(v9__c8_0)) * (1.0f / acc3__c8_0);
        if (a0_b * 80 + a0_u * 10 + 8 < 64) {
            softmax[(a0_b * 80 + a0_u * 10 + 8) % 64 * 64 + (a5_b * 32 + a5_u)] = __float2half(v11__c8_0);
        }
        float acc3__c9_0 = 0.0f;
        __half acc1__c9_0 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__c9_0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                __half in1__c9_0 = x[(a0_b * 80 + a0_u * 10 + 9) % 64 * 128 + a2];
                __half in0__c9_0 = w[a2 * 64 + a1];
                __half v0__c9_0 = in0__c9_0 * in1__c9_0;
                acc0__c9_0 += __half2float(v0__c9_0);
            }
            __half v1__c9_0 = __float2half(acc0__c9_0);
            float one__c9_0 = 1.0f;
            float v1__gn__c9_0 = fmaxf(__half2float(acc1__c9_0), __half2float(v1__c9_0));
            float v1__dg__c9_0 = __half2float(acc1__c9_0) - v1__gn__c9_0;
            float v1__dg_o__c9_0 = __half2float(v1__c9_0) - v1__gn__c9_0;
            float v1__acc3_sa__c9_0 = acc3__c9_0 * expf(v1__dg__c9_0);
            float v1__acc3_sb__c9_0 = one__c9_0 * expf(v1__dg_o__c9_0);
            acc3__c9_0 = v1__acc3_sa__c9_0 + v1__acc3_sb__c9_0;
            acc1__c9_0 = __hmax(acc1__c9_0, v1__c9_0);
        }
        __half v8__c9_0 = __float2half(acc4__c9_0);
        __half v9__c9_0 = v8__c9_0 - acc1__c9_0;
        float v11__c9_0 = __half2float(expf(v9__c9_0)) * (1.0f / acc3__c9_0);
        if (a0_b * 80 + a0_u * 10 + 9 < 64) {
            softmax[(a0_b * 80 + a0_u * 10 + 9) % 64 * 64 + (a5_b * 32 + a5_u)] = __float2half(v11__c9_0);
        }
}
