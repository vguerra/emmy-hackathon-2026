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
__launch_bounds__(128) void k_matmul_288631(const float* x1, const float* x0, float* matmul) {
    extern __shared__ __align__(16) unsigned char _smem_pool[];  // 66048 bytes
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 128;
        int a1_b = (_gid / 128) % 1;
        int a0_u = (_gid / 128) % 1;
        int a1_u = (_gid) % 128;
        float acc0__c0_0 = 0.0f;
        float* _a_smem = reinterpret_cast<float*>(_smem_pool + 0);
        float* _b_smem = reinterpret_cast<float*>(_smem_pool + 512);
        for (int _fa = a0_u * 128 + a1_u; _fa < 16; _fa += 128) {
            emmy_cp_async_cg(&_a_smem[_fa * 4], &x0[(a0_b + _fa / 16) * 128 + 4 * (_fa % 16)]);
        }
        emmy_cp_async_cg(&_b_smem[0 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(a0_u * 128 + a1_u) * 4 + a1_b * 128]);
        emmy_cp_async_cg(&_b_smem[512 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 4) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[1024 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 8) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[1536 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 12) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[2048 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 16) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[2560 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 20) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[3072 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 24) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[3584 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 28) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[4096 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 32) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[4608 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 36) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[5120 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 40) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[5632 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 44) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[6144 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 48) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[6656 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 52) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[7168 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 56) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_cg(&_b_smem[7680 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[((a0_u * 128 + a1_u) / 32 + 60) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
        emmy_cp_async_commit();
        for (int _ks = 0; _ks < 128; _ks += 64) {
            for (int _fa = a0_u * 128 + a1_u; _fa < 16; _fa += 128) {
                emmy_cp_async_cg(&_a_smem[((_ks / 64 + 1) % 2 + _fa / 16) * 64 + 4 * (_fa % 16)], &x0[(a0_b + _fa / 16) * 128 + (((_ks + 64 < 128) ? (_ks + 64) : (64)) + 4 * (_fa % 16))]);
            }
            emmy_cp_async_cg(&_b_smem[(_ks / 64 + 1) % 2 * 64 * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + (a0_u * 128 + a1_u) / 32) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 4) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 4)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 8) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 8)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 12) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 12)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 16) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 16)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 20) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 20)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 24) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 24)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 28) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 28)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 32) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 32)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 36) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 36)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 40) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 40)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 44) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 44)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 48) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 48)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 52) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 52)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 56) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 56)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 1) % 2 * 64 + 60) * 128 + ((a0_u * 128 + a1_u) * 4 / 128) * 128 + ((a0_u * 128 + a1_u) * 4 % 128)], &x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + ((a0_u * 128 + a1_u) / 32 + 60)) * 128 + (a1_b * 128 + 4 * ((a0_u * 128 + a1_u) % 32))]);
            emmy_cp_async_commit();
            emmy_cp_async_wait<1>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 64; _ki++) {
                float in1__c0_0 = _b_smem[(_ks / 64 % 2 * 64 + _ki) * 128 + (a1_b * 128 + a1_u - a1_b * 128)];
                float in0__c0_0 = _a_smem[(_ks / 64 % 2 + (a0_b + a0_u - a0_b)) * 64 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
            }
            __syncthreads();
        }
        matmul[(a0_b + a0_u) * 128 + (a1_b * 128 + a1_u)] = acc0__c0_0;
}
