extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const double* add_c1, const double* x, double* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 2];
        float in0__u0 = add_c1[0];
        float v1__u0 = in0__u0 + (in1__u0 * in1__u0);
        float in1__u1 = x[a0 * 2 + 1];
        float in0__u1 = add_c1[0];
        float v1__u1 = in0__u1 + (in1__u1 * in1__u1);
        add[a0 * 2] = v1__u0;
        add[a0 * 2 + 1] = v1__u1;
}
