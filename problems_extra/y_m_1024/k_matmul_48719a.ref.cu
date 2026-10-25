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
__launch_bounds__(512) void k_matmul_48719a(const float* x1, const float* x0, float* matmul) {
    extern __shared__ __align__(16) unsigned char _smem_pool[];  // 65536 bytes
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 4096;
        int a1_b = (_gid / 512) % 8;
        int a0_u = (_gid / 64) % 8;
        int a1_u = (_gid) % 64;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c4_0 = 0.0f;
        float acc0__c5_0 = 0.0f;
        float acc0__c6_0 = 0.0f;
        float acc0__c7_0 = 0.0f;
        float* _a_smem = reinterpret_cast<float*>(_smem_pool + 0);
        float* _b_smem = reinterpret_cast<float*>(_smem_pool + 32768);
        emmy_cp_async_cg(&_a_smem[0 + ((a0_u * 64 + a1_u) * 4 / 32) * 32 + ((a0_u * 64 + a1_u) * 4 % 32)], &x0[(a0_b * 64 + (a0_u * 64 + a1_u) / 8) * 256 + 4 * ((a0_u * 64 + a1_u) % 8)]);
        emmy_cp_async_cg(&_b_smem[0 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &x1[(a0_u * 64 + a1_u) / 16 * 512 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
        emmy_cp_async_commit();
        emmy_cp_async_cg(&_a_smem[2048 + ((a0_u * 64 + a1_u) * 4 / 32) * 32 + ((a0_u * 64 + a1_u) * 4 % 32)], &x0[(a0_b * 64 + (a0_u * 64 + a1_u) / 8) * 256 + (32 + 4 * ((a0_u * 64 + a1_u) % 8))]);
        emmy_cp_async_cg(&_b_smem[2048 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &x1[(32 + (a0_u * 64 + a1_u) / 16) * 512 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
        emmy_cp_async_commit();
        emmy_cp_async_cg(&_a_smem[4096 + ((a0_u * 64 + a1_u) * 4 / 32) * 32 + ((a0_u * 64 + a1_u) * 4 % 32)], &x0[(a0_b * 64 + (a0_u * 64 + a1_u) / 8) * 256 + (64 + 4 * ((a0_u * 64 + a1_u) % 8))]);
        emmy_cp_async_cg(&_b_smem[4096 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &x1[(64 + (a0_u * 64 + a1_u) / 16) * 512 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
        emmy_cp_async_commit();
        for (int _ks = 0; _ks < 256; _ks += 32) {
            emmy_cp_async_cg(&_a_smem[(_ks / 32 + 3) % 4 * 64 * 32 + ((a0_u * 64 + a1_u) * 4 / 32) * 32 + ((a0_u * 64 + a1_u) * 4 % 32)], &x0[(a0_b * 64 + (a0_u * 64 + a1_u) / 8) * 256 + (((_ks + 96 < 256) ? (_ks + 96) : (224)) + 4 * ((a0_u * 64 + a1_u) % 8))]);
            emmy_cp_async_cg(&_b_smem[(_ks / 32 + 3) % 4 * 32 * 64 + ((a0_u * 64 + a1_u) * 4 / 64) * 64 + ((a0_u * 64 + a1_u) * 4 % 64)], &x1[(((_ks + 96 < 256) ? (_ks + 96) : (224)) + (a0_u * 64 + a1_u) / 16) * 512 + (a1_b * 64 + 4 * ((a0_u * 64 + a1_u) % 16))]);
            emmy_cp_async_commit();
            emmy_cp_async_wait<3>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 32; _ki++) {
                float in1__c0_0 = _b_smem[(_ks / 32 % 4 * 32 + _ki) * 64 + (a1_b * 64 + a1_u - a1_b * 64)];
                float in0__c0_0 = _a_smem[(_ks / 32 % 4 * 64 + (a0_b * 64 + a0_u * 8 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in0__c1_0 = _a_smem[(_ks / 32 % 4 * 64 + (a0_b * 64 + a0_u * 8 + 1 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float in0__c2_0 = _a_smem[(_ks / 32 % 4 * 64 + (a0_b * 64 + a0_u * 8 + 2 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float in0__c3_0 = _a_smem[(_ks / 32 % 4 * 64 + (a0_b * 64 + a0_u * 8 + 3 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                float in0__c4_0 = _a_smem[(_ks / 32 % 4 * 64 + (a0_b * 64 + a0_u * 8 + 4 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c4_0 = in1__c0_0 * in0__c4_0;
                acc0__c4_0 += acc0__v__c4_0;
                float in0__c5_0 = _a_smem[(_ks / 32 % 4 * 64 + (a0_b * 64 + a0_u * 8 + 5 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c5_0 = in1__c0_0 * in0__c5_0;
                acc0__c5_0 += acc0__v__c5_0;
                float in0__c6_0 = _a_smem[(_ks / 32 % 4 * 64 + (a0_b * 64 + a0_u * 8 + 6 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c6_0 = in1__c0_0 * in0__c6_0;
                acc0__c6_0 += acc0__v__c6_0;
                float in0__c7_0 = _a_smem[(_ks / 32 % 4 * 64 + (a0_b * 64 + a0_u * 8 + 7 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c7_0 = in1__c0_0 * in0__c7_0;
                acc0__c7_0 += acc0__v__c7_0;
            }
            __syncthreads();
        }
        matmul[(a0_b * 64 + a0_u * 8) * 512 + (a1_b * 64 + a1_u)] = acc0__c0_0;
        matmul[(a0_b * 64 + a0_u * 8 + 1) * 512 + (a1_b * 64 + a1_u)] = acc0__c1_0;
        matmul[(a0_b * 64 + a0_u * 8 + 2) * 512 + (a1_b * 64 + a1_u)] = acc0__c2_0;
        matmul[(a0_b * 64 + a0_u * 8 + 3) * 512 + (a1_b * 64 + a1_u)] = acc0__c3_0;
        matmul[(a0_b * 64 + a0_u * 8 + 4) * 512 + (a1_b * 64 + a1_u)] = acc0__c4_0;
        matmul[(a0_b * 64 + a0_u * 8 + 5) * 512 + (a1_b * 64 + a1_u)] = acc0__c5_0;
        matmul[(a0_b * 64 + a0_u * 8 + 6) * 512 + (a1_b * 64 + a1_u)] = acc0__c6_0;
        matmul[(a0_b * 64 + a0_u * 8 + 7) * 512 + (a1_b * 64 + a1_u)] = acc0__c7_0;
}
