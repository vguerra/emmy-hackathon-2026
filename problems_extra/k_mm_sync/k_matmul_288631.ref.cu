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
            float4 _v__a_copy0 = *reinterpret_cast<const float4*>(&x0[(a0_b + _fa / 16) * 128 + 4 * (_fa % 16)]);
            float _a_copy0 = _v__a_copy0.x;
            float _a_copy1 = _v__a_copy0.y;
            float _a_copy2 = _v__a_copy0.z;
            float _a_copy3 = _v__a_copy0.w;
            float4 _vs__a_copy0_0bfc4020 = make_float4(_a_copy0, _a_copy1, _a_copy2, _a_copy3);
            *reinterpret_cast<float4*>(&_a_smem[_fa * 4]) = _vs__a_copy0_0bfc4020;
        }
        for (int _fb = a0_u * 128 + a1_u; _fb < 2048; _fb += 128) {
            float4 _v__b_copy0 = *reinterpret_cast<const float4*>(&x1[_fb * 4 + a1_b * 128]);
            float _b_copy0 = _v__b_copy0.x;
            float _b_copy1 = _v__b_copy0.y;
            float _b_copy2 = _v__b_copy0.z;
            float _b_copy3 = _v__b_copy0.w;
            float4 _vs__b_copy0_1c103c73 = make_float4(_b_copy0, _b_copy1, _b_copy2, _b_copy3);
            *reinterpret_cast<float4*>(&_b_smem[_fb * 4]) = _vs__b_copy0_1c103c73;
        }
        for (int _ks = 0; _ks < 128; _ks += 64) {
            for (int _fa = a0_u * 128 + a1_u; _fa < 16; _fa += 128) {
                float4 _v__a_copy0 = *reinterpret_cast<const float4*>(&x0[(a0_b + _fa / 16) * 128 + (((_ks + 64 < 128) ? (_ks + 64) : (64)) + 4 * (_fa % 16))]);
                float _a_copy0 = _v__a_copy0.x;
                float _a_copy1 = _v__a_copy0.y;
                float _a_copy2 = _v__a_copy0.z;
                float _a_copy3 = _v__a_copy0.w;
                float4 _vs__a_copy0_902f08e2 = make_float4(_a_copy0, _a_copy1, _a_copy2, _a_copy3);
                *reinterpret_cast<float4*>(&_a_smem[((_ks / 64 + 1) % 2 + _fa / 16) * 64 + 4 * (_fa % 16)]) = _vs__a_copy0_902f08e2;
            }
            for (int _fb = a0_u * 128 + a1_u; _fb < 2048; _fb += 128) {
                float4 _v__b_copy0 = *reinterpret_cast<const float4*>(&x1[(((_ks + 64 < 128) ? (_ks + 64) : (64)) + _fb / 32) * 128 + (a1_b * 128 + 4 * (_fb % 32))]);
                float _b_copy0 = _v__b_copy0.x;
                float _b_copy1 = _v__b_copy0.y;
                float _b_copy2 = _v__b_copy0.z;
                float _b_copy3 = _v__b_copy0.w;
                float4 _vs__b_copy0_b422a2da = make_float4(_b_copy0, _b_copy1, _b_copy2, _b_copy3);
                *reinterpret_cast<float4*>(&_b_smem[((_ks / 64 + 1) % 2 * 64 + _fb / 32) * 128 + 4 * (_fb % 32)]) = _vs__b_copy0_b422a2da;
            }
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
