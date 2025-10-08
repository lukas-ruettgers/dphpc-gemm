#pragma once
#include <iostream>
#include <sstream> // (L) sstream not included by default in Windows

#include <vector>
#include <random>
#include <type_traits>
#include <cstdint>
#include <cmath>
#include <limits>
#include "../globals.h"

enum class DistributionType {
    Uniform,
    Gaussian
};

template <typename T>
std::vector<T> generateRandomMatrix(int H, int W,
                                   DistributionType distType = DistributionType::Gaussian,
                                   std::uint64_t seed = GLOBAL_SEED,
                                   T param1 = T(0), T param2 = T(1)) {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point.");

    const auto n = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    std::vector<T> y(n);

    // Stable and reproducible RNG seed
    std::seed_seq seq{
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32),
        0x85EBCA6B, 0xC2B2AE35
    };
    std::mt19937_64 rng(seq);

    switch (distType) {
        case DistributionType::Uniform: {
            // param1 = lower, param2 = upper
            const T upper = std::nextafter(param2, std::numeric_limits<T>::max());
            std::uniform_real_distribution<T> dist(param1, upper);
            for (std::size_t i = 0; i < n; ++i)
                y[i] = dist(rng);
            break;
        }

        case DistributionType::Gaussian: {
            // param1 = mean, param2 = stddev
            std::normal_distribution<T> dist(param1, param2);
            for (std::size_t i = 0; i < n; ++i)
                y[i] = dist(rng);
            break;
        }

        default:
            throw std::invalid_argument("Unknown DistributionType.");
    }

    return y;
}

template <typename T>
void GEMM(T* A, T* B, T* C, T alpha, T beta, int M, int N, int K) {
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			T y = 0;
			for (int k = 0; k < K; k++) {
				y += A[m * K + k] * B[k * N + n];
			}
			C[m * N + n] += alpha * y + beta * C[m * N + n];
		}
	}
}


template <typename T> 
bool compare_results(T* result, T* groundTruth, int H, int W) {
	for (int h = 0; h < H; h++) {
		for (int w = 0; w < W; w++) {
			int i = h * W + w;
			if (abs(result[i] - groundTruth[i]) > 1e-3) {
				std::cout << "Error at: H=" << h << ", W=" << w << ", result=" << result[i] << ", groundTruth=" << groundTruth[i] << std::endl;
				return false;
			}
		}
	}

	return true;
}
