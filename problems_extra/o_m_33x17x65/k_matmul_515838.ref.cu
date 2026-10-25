extern "C" __global__
__launch_bounds__(256) void k_matmul_515838(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 726) {
        int a0_b = _gid / 22;
        int a1_b = (_gid) % 22;
        int a0_u = (_gid) % 1;
        int a1_u = (_gid) % 1;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c0_2 = 0.0f;
        #pragma unroll
        for (int a2 = 0; a2 < 17; a2++) {
            float in0__ar0 = x0[(a0_b + a0_u) * 17 + a2];
            float in1__bc0 = x1[a2 * 65 + (a1_b * 3 + a1_u * 3) % 65];
            float v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += v0__c0_0;
            float in1__bc1 = x1[a2 * 65 + (a1_b * 3 + a1_u * 3 + 1) % 65];
            float v0__c0_1 = in0__ar0 * in1__bc1;
            acc0__c0_1 += v0__c0_1;
            float in1__bc2 = x1[a2 * 65 + (a1_b * 3 + a1_u * 3 + 2) % 65];
            float v0__c0_2 = in0__ar0 * in1__bc2;
            acc0__c0_2 += v0__c0_2;
        }
        if (a1_b * 3 + a1_u * 3 + 0 < 65) {
            matmul[(a0_b + a0_u) * 65 + (a1_b * 3 + a1_u * 3) % 65] = acc0__c0_0;
        }
        if (a1_b * 3 + a1_u * 3 + 1 < 65) {
            matmul[(a0_b + a0_u) * 65 + (a1_b * 3 + a1_u * 3 + 1) % 65] = acc0__c0_1;
        }
        if (a1_b * 3 + a1_u * 3 + 2 < 65) {
            matmul[(a0_b + a0_u) * 65 + (a1_b * 3 + a1_u * 3 + 2) % 65] = acc0__c0_2;
        }
    }
}
