#include <cuda_device_runtime_api.h>
#include <thrust/detail/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cstdint>
#include <iostream>

#include "image_processing.hpp"

__global__ void d_initialize_A(float* d_A) {
  d_A[0] = 1.0 / 6.0;
  d_A[1] = 4.0 / 6.0;
  d_A[2] = 1.0 / 6.0;
  d_A[3] = 0.0 / 6.0;
  d_A[4] = -3.0 / 6.0;
  d_A[5] = 0.0 / 6.0;
  d_A[6] = 3.0 / 6.0;
  d_A[7] = 0.0 / 6.0;
  d_A[8] = 3.0 / 6.0;
  d_A[9] = -6.0 / 6.0;
  d_A[10] = 3.0 / 6.0;
  d_A[11] = 0.0 / 6.0;
  d_A[12] = -1.0 / 6.0;
  d_A[13] = 3.0 / 6.0;
  d_A[14] = -3.0 / 6.0;
  d_A[15] = 1.0 / 6.0;
}

__global__ void d_initialize_U(float* d_U, const uint32_t N) {
  const uint32_t row = blockIdx.x;
  const uint32_t power = threadIdx.x;
  const uint32_t P = blockDim.x;

  d_U[row * (P + 1) + power] = pow(float(row) / gridDim.x, power);
}

__global__ void d_evaluate_pixels(float* d_pixels, const float* d_U,
                                  const float* d_A, const float* d_P,
                                  const uint32_t P, const uint32_t N) {
  const uint32_t pixel_y = blockIdx.x;
  const uint32_t pixel_x = threadIdx.x;
  const uint32_t W = blockDim.x;

  uint32_t u_idx = pixel_x % (W / N) * (P + 1);
  uint32_t p_idx = pixel_y * (N * P + 1) + (pixel_x / (W / N)) * P;

  float val = 0.0;
  for (uint32_t i = 0; i < P + 1; ++i) {
    for (uint32_t j = 0; j < P + 1; ++j) {
      val += d_U[u_idx + i] * d_A[i * (P + 1) + j] * d_P[p_idx + j];
    }
  }
  d_pixels[pixel_y * W + pixel_x] = val;
}

ImageProcessing::ImageProcessing(const uint32_t w, const uint32_t h,
                                 const uint32_t P, const uint32_t N)
    : m_size{w, h}, m_P(P), m_N(N) {
  d_A.resize((P + 1) * (P + 1));
  d_initialize_A<<<1, 1>>>(thrust::raw_pointer_cast(d_A.data()));
  d_U.resize((w / N) * (P + 1));
  d_initialize_U<<<w / N, P>>>(thrust::raw_pointer_cast(d_U.data()), m_N);
  d_P.resize(h * (N * P + 1));
  d_pixels.resize(w * h);
}

void ImageProcessing::evaluate_pixels() {
  d_evaluate_pixels<<<m_size.y, m_size.x>>>(
      thrust::raw_pointer_cast(d_pixels.data()),
      thrust::raw_pointer_cast(d_U.data()),
      thrust::raw_pointer_cast(d_A.data()),
      thrust::raw_pointer_cast(d_P.data()), m_P, m_N);
  cudaDeviceSynchronize();
}

void ImageProcessing::set_p(float p) {
  thrust::fill(thrust::device, d_P.begin(), d_P.end(), p);
  cudaDeviceSynchronize();
}
