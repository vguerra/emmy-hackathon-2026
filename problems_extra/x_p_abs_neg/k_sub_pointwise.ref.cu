extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* x0, const float* x1, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = x0[a0 * 2];
        float in1__u0 = x1[a0 * 2];
        float v1__u0 = fabsf(in0__u0) - in1__u0;
        float in0__u1 = x0[a0 * 2 + 1];
        float in1__u1 = x1[a0 * 2 + 1];
        float v1__u1 = fabsf(in0__u1) - in1__u1;
        float2 _vs_v1__u0_48f3a952 = make_float2(v1__u0, v1__u1);
        *reinterpret_cast<float2*>(&sub[a0 * 2]) = _vs_v1__u0_48f3a952;
}
