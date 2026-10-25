extern "C" __global__
__launch_bounds__(256) void k_log_pointwise(const float* add_c1, const float* x, float* log) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 2];
        float in0__u0 = add_c1[0];
        float v2__u0 = log((in0__u0 + (in1__u0 * in1__u0)));
        float in1__u1 = x[a0 * 2 + 1];
        float in0__u1 = add_c1[0];
        float v2__u1 = log((in0__u1 + (in1__u1 * in1__u1)));
        float2 _vs_v2__u0_4d2e1e49 = make_float2(v2__u0, v2__u1);
        *reinterpret_cast<float2*>(&log[a0 * 2]) = _vs_v2__u0_4d2e1e49;
}
