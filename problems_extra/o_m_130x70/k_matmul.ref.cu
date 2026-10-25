extern "C" __global__
__launch_bounds__(128) void k_matmul(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 6528;
        int a0_b = (_gid / 384) % 17;
        int a1_b = (_gid / 128) % 3;
        int a0_u = (_gid / 16) % 8;
        int a1_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c0_2 = 0.0f;
        float acc0__c0_3 = 0.0f;
        #pragma unroll
        for (int a2 = 0; a2 < 35; a2++) {
            float in0__ar0 = x0[(a0_b * 8 + a0_u) % 130 * 70 + (_a2_ks * 35 + a2)];
            float in1__bc0 = x1[(_a2_ks * 35 + a2) * 130 + (a1_b * 64 + a1_u * 4) % 130];
            float v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += v0__c0_0;
            float in1__bc1 = x1[(_a2_ks * 35 + a2) * 130 + (a1_b * 64 + a1_u * 4 + 1) % 130];
            float v0__c0_1 = in0__ar0 * in1__bc1;
            acc0__c0_1 += v0__c0_1;
            float in1__bc2 = x1[(_a2_ks * 35 + a2) * 130 + (a1_b * 64 + a1_u * 4 + 2) % 130];
            float v0__c0_2 = in0__ar0 * in1__bc2;
            acc0__c0_2 += v0__c0_2;
            float in1__bc3 = x1[(_a2_ks * 35 + a2) * 130 + (a1_b * 64 + a1_u * 4 + 3) % 130];
            float v0__c0_3 = in0__ar0 * in1__bc3;
            acc0__c0_3 += v0__c0_3;
        }
        if (a0_b * 8 + a0_u * 1 + 0 < 130 && a1_b * 64 + a1_u * 4 + 0 < 130) {
            atomicAdd(&matmul[(a0_b * 8 + a0_u) % 130 * 130 + (a1_b * 64 + a1_u * 4) % 130], acc0__c0_0);
        }
        if (a0_b * 8 + a0_u * 1 + 0 < 130 && a1_b * 64 + a1_u * 4 + 1 < 130) {
            atomicAdd(&matmul[(a0_b * 8 + a0_u) % 130 * 130 + (a1_b * 64 + a1_u * 4 + 1) % 130], acc0__c0_1);
        }
        if (a0_b * 8 + a0_u * 1 + 0 < 130 && a1_b * 64 + a1_u * 4 + 2 < 130) {
            atomicAdd(&matmul[(a0_b * 8 + a0_u) % 130 * 130 + (a1_b * 64 + a1_u * 4 + 2) % 130], acc0__c0_2);
        }
        if (a0_b * 8 + a0_u * 1 + 0 < 130 && a1_b * 64 + a1_u * 4 + 3 < 130) {
            atomicAdd(&matmul[(a0_b * 8 + a0_u) % 130 * 130 + (a1_b * 64 + a1_u * 4 + 3) % 130], acc0__c0_3);
        }
}
