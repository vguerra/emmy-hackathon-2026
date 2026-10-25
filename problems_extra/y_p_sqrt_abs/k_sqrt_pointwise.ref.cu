extern "C" __global__
__launch_bounds__(256) void k_sqrt_pointwise(const float* x, float* sqrt) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float2 _v_in0__u0 = *reinterpret_cast<const float2*>(&x[a0 * 2]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float v1__u0 = sqrtf(fabsf(in0__u0));
        float v1__u1 = sqrtf(fabsf(in0__u1));
        float2 _vs_v1__u0_3148cbe6 = make_float2(v1__u0, v1__u1);
        *reinterpret_cast<float2*>(&sqrt[a0 * 2]) = _vs_v1__u0_3148cbe6;
}
