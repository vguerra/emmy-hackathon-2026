extern "C" __global__
__launch_bounds__(256) void k_mul_1_pointwise(const float* x, float* mul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float2 _v_in0__u0 = *reinterpret_cast<const float2*>(&x[a0 * 2]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float v1__u0 = in0__u0 * (in0__u0 * in0__u0);
        float v1__u1 = in0__u1 * (in0__u1 * in0__u1);
        float2 _vs_v1__u0_9ececc87 = make_float2(v1__u0, v1__u1);
        *reinterpret_cast<float2*>(&mul_1[a0 * 2]) = _vs_v1__u0_9ececc87;
}
