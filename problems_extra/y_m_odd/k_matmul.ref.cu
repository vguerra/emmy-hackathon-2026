extern "C" __global__
__launch_bounds__(256) void k_matmul(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 900) {
        int _a2_ks = _gid / 450;
        int a0_b = (_gid / 18) % 25;
        int a1_b = (_gid) % 18;
        int a0_u = (_gid) % 1;
        int a1_u = (_gid) % 1;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c0_2 = 0.0f;
        float acc0__c0_3 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c1_2 = 0.0f;
        float acc0__c1_3 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c2_1 = 0.0f;
        float acc0__c2_2 = 0.0f;
        float acc0__c2_3 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c3_1 = 0.0f;
        float acc0__c3_2 = 0.0f;
        float acc0__c3_3 = 0.0f;
        #pragma unroll
        for (int a2 = 0; a2 < 25; a2++) {
            float in0__ar0 = x0[(a0_b * 4 + a0_u * 4) * 50 + (_a2_ks * 25 + a2)];
            float in1__bc0 = x1[(_a2_ks * 25 + a2) * 70 + (a1_b * 4 + a1_u * 4) % 70];
            float v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += v0__c0_0;
            float in1__bc1 = x1[(_a2_ks * 25 + a2) * 70 + (a1_b * 4 + a1_u * 4 + 1) % 70];
            float v0__c0_1 = in0__ar0 * in1__bc1;
            acc0__c0_1 += v0__c0_1;
            float in1__bc2 = x1[(_a2_ks * 25 + a2) * 70 + (a1_b * 4 + a1_u * 4 + 2) % 70];
            float v0__c0_2 = in0__ar0 * in1__bc2;
            acc0__c0_2 += v0__c0_2;
            float in1__bc3 = x1[(_a2_ks * 25 + a2) * 70 + (a1_b * 4 + a1_u * 4 + 3) % 70];
            float v0__c0_3 = in0__ar0 * in1__bc3;
            acc0__c0_3 += v0__c0_3;
            float in0__ar1 = x0[(a0_b * 4 + a0_u * 4 + 1) * 50 + (_a2_ks * 25 + a2)];
            float v0__c1_0 = in0__ar1 * in1__bc0;
            acc0__c1_0 += v0__c1_0;
            float v0__c1_1 = in0__ar1 * in1__bc1;
            acc0__c1_1 += v0__c1_1;
            float v0__c1_2 = in0__ar1 * in1__bc2;
            acc0__c1_2 += v0__c1_2;
            float v0__c1_3 = in0__ar1 * in1__bc3;
            acc0__c1_3 += v0__c1_3;
            float in0__ar2 = x0[(a0_b * 4 + a0_u * 4 + 2) * 50 + (_a2_ks * 25 + a2)];
            float v0__c2_0 = in0__ar2 * in1__bc0;
            acc0__c2_0 += v0__c2_0;
            float v0__c2_1 = in0__ar2 * in1__bc1;
            acc0__c2_1 += v0__c2_1;
            float v0__c2_2 = in0__ar2 * in1__bc2;
            acc0__c2_2 += v0__c2_2;
            float v0__c2_3 = in0__ar2 * in1__bc3;
            acc0__c2_3 += v0__c2_3;
            float in0__ar3 = x0[(a0_b * 4 + a0_u * 4 + 3) * 50 + (_a2_ks * 25 + a2)];
            float v0__c3_0 = in0__ar3 * in1__bc0;
            acc0__c3_0 += v0__c3_0;
            float v0__c3_1 = in0__ar3 * in1__bc1;
            acc0__c3_1 += v0__c3_1;
            float v0__c3_2 = in0__ar3 * in1__bc2;
            acc0__c3_2 += v0__c3_2;
            float v0__c3_3 = in0__ar3 * in1__bc3;
            acc0__c3_3 += v0__c3_3;
        }
        if (a1_b * 4 + a1_u * 4 + 0 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4) * 70 + (a1_b * 4 + a1_u * 4) % 70], acc0__c0_0);
        }
        if (a1_b * 4 + a1_u * 4 + 1 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4) * 70 + (a1_b * 4 + a1_u * 4 + 1) % 70], acc0__c0_1);
        }
        if (a1_b * 4 + a1_u * 4 + 2 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4) * 70 + (a1_b * 4 + a1_u * 4 + 2) % 70], acc0__c0_2);
        }
        if (a1_b * 4 + a1_u * 4 + 3 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4) * 70 + (a1_b * 4 + a1_u * 4 + 3) % 70], acc0__c0_3);
        }
        if (a1_b * 4 + a1_u * 4 + 0 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 1) * 70 + (a1_b * 4 + a1_u * 4) % 70], acc0__c1_0);
        }
        if (a1_b * 4 + a1_u * 4 + 1 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 1) * 70 + (a1_b * 4 + a1_u * 4 + 1) % 70], acc0__c1_1);
        }
        if (a1_b * 4 + a1_u * 4 + 2 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 1) * 70 + (a1_b * 4 + a1_u * 4 + 2) % 70], acc0__c1_2);
        }
        if (a1_b * 4 + a1_u * 4 + 3 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 1) * 70 + (a1_b * 4 + a1_u * 4 + 3) % 70], acc0__c1_3);
        }
        if (a1_b * 4 + a1_u * 4 + 0 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 2) * 70 + (a1_b * 4 + a1_u * 4) % 70], acc0__c2_0);
        }
        if (a1_b * 4 + a1_u * 4 + 1 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 2) * 70 + (a1_b * 4 + a1_u * 4 + 1) % 70], acc0__c2_1);
        }
        if (a1_b * 4 + a1_u * 4 + 2 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 2) * 70 + (a1_b * 4 + a1_u * 4 + 2) % 70], acc0__c2_2);
        }
        if (a1_b * 4 + a1_u * 4 + 3 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 2) * 70 + (a1_b * 4 + a1_u * 4 + 3) % 70], acc0__c2_3);
        }
        if (a1_b * 4 + a1_u * 4 + 0 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 3) * 70 + (a1_b * 4 + a1_u * 4) % 70], acc0__c3_0);
        }
        if (a1_b * 4 + a1_u * 4 + 1 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 3) * 70 + (a1_b * 4 + a1_u * 4 + 1) % 70], acc0__c3_1);
        }
        if (a1_b * 4 + a1_u * 4 + 2 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 3) * 70 + (a1_b * 4 + a1_u * 4 + 2) % 70], acc0__c3_2);
        }
        if (a1_b * 4 + a1_u * 4 + 3 < 70) {
            atomicAdd(&matmul[(a0_b * 4 + a0_u * 4 + 3) * 70 + (a1_b * 4 + a1_u * 4 + 3) % 70], acc0__c3_3);
        }
    }
}
