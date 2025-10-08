// iwp_nanobind_numpy.cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>   // ndarray/NumPy/CuPy/JAX/Tensorflow interop
#include <cstring>              // std::memcpy if you need it

namespace nb = nanobind;
using namespace nb::literals;

// Example: your raw CPU implementation (no Torch)
template <class T>
void iwp_forward_cpu(const T* x, const T* a, const T* b, const T* w,
                     T* y, std::size_t H, std::size_t W) {
    // ... fill y[...] from x,a,b,w (row-major) ...
    // (dummy example)
    for (std::size_t i = 0; i < H*W; ++i)
        y[i] = x[i] * a[0] + b[0] + w[0];
}

// Constrain inputs: CPU, 2D, C-contiguous, dtype float32/float64
template <typename T>
using MatrixIn = nb::ndarray<const T, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;

// Return a NumPy array (CPU, 2D, C-contiguous) owning its memory
template <typename T>
using MatrixOut = nb::ndarray<T, nb::numpy, nb::shape<-1, -1>, nb::c_contig>;

NB_MODULE(iwp_no_torch, m) {
    // float32 version
    m.def("iwp_forward_f32",
        [](MatrixIn<float> x, MatrixIn<float> a, MatrixIn<float> b, MatrixIn<float> w) {
            std::size_t H = x.shape(0), W = x.shape(1);

            // Allocate C++ storage the NumPy array will own
            auto* data = new float[H * W];

            // Wrap it as a NumPy ndarray with a destructor capsule
            MatrixOut<float> y(
                data,
                { (nb::ssize_t)H, (nb::ssize_t)W },
                { (nb::ssize_t)W * (nb::ssize_t)sizeof(float), (nb::ssize_t)sizeof(float) },
                nb::capsule(data, [](void* p) noexcept { delete[] (float*)p; })
            );

            iwp_forward_cpu<float>(x.data(), a.data(), b.data(), w.data(), y.data(), H, W);
            return y; // zero-copy into Python; Python owns `data`
        },
        "x"_a, "a"_a, "b"_a, "w"_a,
        "IWP forward (CPU, float32)");

    // float64 version
    m.def("iwp_forward_f64",
        [](MatrixIn<double> x, MatrixIn<double> a, MatrixIn<double> b, MatrixIn<double> w) {
            std::size_t H = x.shape(0), W = x.shape(1);
            auto* data = new double[H * W];
            MatrixOut<double> y(
                data,
                { (nb::ssize_t)H, (nb::ssize_t)W },
                { (nb::ssize_t)W * (nb::ssize_t)sizeof(double), (nb::ssize_t)sizeof(double) },
                nb::capsule(data, [](void* p) noexcept { delete[] (double*)p; })
            );
            iwp_forward_cpu<double>(x.data(), a.data(), b.data(), w.data(), y.data(), H, W);
            return y;
        },
        "x"_a, "a"_a, "b"_a, "w"_a,
        "IWP forward (CPU, float64)");
}
