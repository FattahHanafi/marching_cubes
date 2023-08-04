#include <cuda_device_runtime_api.h>
#include <thrust/detail/copy.h>
#include <thrust/host_vector.h>

#include <cstdint>
#include <iomanip>
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

__global__ void d_evaluate_pixels(float* d_pixels, const float* d_U, const float* d_A, const float* d_P, const uint32_t P, const uint32_t N) {
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

__global__ void d_update_is_valid(bool* d_is_valid, const float* d_raw_pixels, const float th) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  d_is_valid[idx] = d_raw_pixels[idx] > th;
}

__global__ void d_evaluate_error(const bool* d_is_valid, const float* d_raw_pixels, const float* d_pixels, float* d_error) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  d_error[idx] = d_is_valid[idx] ? pow(d_pixels[idx] - d_raw_pixels[idx], 2.0f) : 0.0f;
}

__global__ void d_initialize_J(float* d_J, const float* d_U, const float* d_A, const uint32_t P) {
  const uint32_t u_idx = threadIdx.x * (P + 1);
  const uint32_t j_idx = u_idx * (P + 2);

  d_J[j_idx + 0] = pow(d_A[0] * d_U[u_idx + 0] + d_A[4] * d_U[u_idx + 1] + d_A[8] * d_U[u_idx + 2] + d_A[12] * d_U[u_idx + 3], 2.0) * 2.0;
  d_J[j_idx + 1] = (d_A[0] * d_U[u_idx + 0] + d_A[4] * d_U[u_idx + 1] + d_A[8] * d_U[u_idx + 2] + d_A[12] * d_U[u_idx + 3]) *
                   (d_A[1] * d_U[u_idx + 0] + d_A[5] * d_U[u_idx + 1] + d_A[9] * d_U[u_idx + 2] + d_A[13] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 2] = (d_A[0] * d_U[u_idx + 0] + d_A[4] * d_U[u_idx + 1] + d_A[8] * d_U[u_idx + 2] + d_A[12] * d_U[u_idx + 3]) *
                   (d_A[2] * d_U[u_idx + 0] + d_A[6] * d_U[u_idx + 1] + d_A[10] * d_U[u_idx + 2] + d_A[14] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 3] = (d_A[0] * d_U[u_idx + 0] + d_A[4] * d_U[u_idx + 1] + d_A[8] * d_U[u_idx + 2] + d_A[12] * d_U[u_idx + 3]) *
                   (d_A[3] * d_U[u_idx + 0] + d_A[7] * d_U[u_idx + 1] + d_A[11] * d_U[u_idx + 2] + d_A[15] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 4] = -2.0 * (d_A[0] * d_U[u_idx + 0] + d_A[4] * d_U[u_idx + 1] + d_A[8] * d_U[u_idx + 2] + d_A[12] * d_U[u_idx + 3]);  // * y
  d_J[j_idx + 5] = (d_A[0] * d_U[u_idx + 0] + d_A[4] * d_U[u_idx + 1] + d_A[8] * d_U[u_idx + 2] + d_A[12] * d_U[u_idx + 3]) *
                   (d_A[1] * d_U[u_idx + 0] + d_A[5] * d_U[u_idx + 1] + d_A[9] * d_U[u_idx + 2] + d_A[13] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 6] = pow(d_A[1] * d_U[u_idx + 0] + d_A[5] * d_U[u_idx + 1] + d_A[9] * d_U[u_idx + 2] + d_A[13] * d_U[u_idx + 3], 2.0) * 2.0;
  d_J[j_idx + 7] = (d_A[1] * d_U[u_idx + 0] + d_A[5] * d_U[u_idx + 1] + d_A[9] * d_U[u_idx + 2] + d_A[13] * d_U[u_idx + 3]) *
                   (d_A[2] * d_U[u_idx + 0] + d_A[6] * d_U[u_idx + 1] + d_A[10] * d_U[u_idx + 2] + d_A[14] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 8] = (d_A[1] * d_U[u_idx + 0] + d_A[5] * d_U[u_idx + 1] + d_A[9] * d_U[u_idx + 2] + d_A[13] * d_U[u_idx + 3]) *
                   (d_A[3] * d_U[u_idx + 0] + d_A[7] * d_U[u_idx + 1] + d_A[11] * d_U[u_idx + 2] + d_A[15] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 9] = -2.0 * (d_A[1] * d_U[u_idx + 0] + d_A[5] * d_U[u_idx + 1] + d_A[9] * d_U[u_idx + 2] + d_A[13] * d_U[u_idx + 3]);  // * y
  d_J[j_idx + 10] = (d_A[0] * d_U[u_idx + 0] + d_A[4] * d_U[u_idx + 1] + d_A[8] * d_U[u_idx + 2] + d_A[12] * d_U[u_idx + 3]) *
                    (d_A[2] * d_U[u_idx + 0] + d_A[6] * d_U[u_idx + 1] + d_A[10] * d_U[u_idx + 2] + d_A[14] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 11] = (d_A[1] * d_U[u_idx + 0] + d_A[5] * d_U[u_idx + 1] + d_A[9] * d_U[u_idx + 2] + d_A[13] * d_U[u_idx + 3]) *
                    (d_A[2] * d_U[u_idx + 0] + d_A[6] * d_U[u_idx + 1] + d_A[10] * d_U[u_idx + 2] + d_A[14] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 12] = pow(d_A[2] * d_U[u_idx + 0] + d_A[6] * d_U[u_idx + 1] + d_A[10] * d_U[u_idx + 2] + d_A[14] * d_U[u_idx + 3], 2.0) * 2.0;
  d_J[j_idx + 13] = (d_A[2] * d_U[u_idx + 0] + d_A[6] * d_U[u_idx + 1] + d_A[10] * d_U[u_idx + 2] + d_A[14] * d_U[u_idx + 3]) *
                    (d_A[3] * d_U[u_idx + 0] + d_A[7] * d_U[u_idx + 1] + d_A[11] * d_U[u_idx + 2] + d_A[15] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 14] = -2.0 * (d_A[2] * d_U[u_idx + 0] + d_A[6] * d_U[u_idx + 1] + d_A[10] * d_U[u_idx + 2] + d_A[14] * d_U[u_idx + 3]);  // * y
  d_J[j_idx + 15] = (d_A[0] * d_U[u_idx + 0] + d_A[4] * d_U[u_idx + 1] + d_A[8] * d_U[u_idx + 2] + d_A[12] * d_U[u_idx + 3]) *
                    (d_A[3] * d_U[u_idx + 0] + d_A[7] * d_U[u_idx + 1] + d_A[11] * d_U[u_idx + 2] + d_A[15] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 16] = (d_A[1] * d_U[u_idx + 0] + d_A[5] * d_U[u_idx + 1] + d_A[9] * d_U[u_idx + 2] + d_A[13] * d_U[u_idx + 3]) *
                    (d_A[3] * d_U[u_idx + 0] + d_A[7] * d_U[u_idx + 1] + d_A[11] * d_U[u_idx + 2] + d_A[15] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 17] = (d_A[2] * d_U[u_idx + 0] + d_A[6] * d_U[u_idx + 1] + d_A[10] * d_U[u_idx + 2] + d_A[14] * d_U[u_idx + 3]) *
                    (d_A[3] * d_U[u_idx + 0] + d_A[7] * d_U[u_idx + 1] + d_A[11] * d_U[u_idx + 2] + d_A[15] * d_U[u_idx + 3]) * 2.0;
  d_J[j_idx + 18] = pow(d_A[3] * d_U[u_idx + 0] + d_A[7] * d_U[u_idx + 1] + d_A[11] * d_U[u_idx + 2] + d_A[15] * d_U[u_idx + 3], 2.0) * 2.0;
  d_J[j_idx + 19] = -2.0 * (d_A[3] * d_U[u_idx + 0] + d_A[7] * d_U[u_idx + 1] + d_A[11] * d_U[u_idx + 2] + d_A[15] * d_U[u_idx + 3]);  // * y
}

ImageProcessing::ImageProcessing(const uint32_t w, const uint32_t h, const uint32_t P, const uint32_t N) : m_size{w, h}, m_P(P), m_N(N) {
  d_A.resize((P + 1) * (P + 1));
  d_initialize_A<<<1, 1>>>(thrust::raw_pointer_cast(d_A.data()));
  cudaDeviceSynchronize();
  d_U.resize((w / N) * (P + 1));
  d_initialize_U<<<w / N, P>>>(thrust::raw_pointer_cast(d_U.data()), m_N);
  cudaDeviceSynchronize();
  d_P.resize(h * (N * P + 1));
  d_pixels.resize(w * h);
  d_raw_pixels.resize(w * h);
  h_raw_pixels.resize(w * h);
  d_is_valid.resize(w * h);
  d_error.resize(w * h);
  d_J.resize((w / N) * (P + 1) * (P + 2));
  d_initialize_J<<<1, w / N>>>(thrust::raw_pointer_cast(d_J.data()), thrust::raw_pointer_cast(d_U.data()), thrust::raw_pointer_cast(d_A.data()), P);
  cudaDeviceSynchronize();

  thrust::host_vector<float> h_J;
  h_J.resize(d_J.size());
  thrust::copy(d_J.begin(), d_J.end(), h_J.begin());
  for (uint32_t k = 0; k < 80; ++k) {
    std::cout << "i = " << k << "=============\n";
    for (uint32_t i = k * 4; i < (k + 1) * 4; ++i) {
      for (uint32_t j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(4) << h_J[i * 5 + j] << " ";
      std::cout << '\n';
    }
  }
}

void ImageProcessing::evaluate_pixels() {
  d_evaluate_pixels<<<m_size.y, m_size.x>>>(thrust::raw_pointer_cast(d_pixels.data()), thrust::raw_pointer_cast(d_U.data()),
                                            thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_P.data()), m_P, m_N);
  cudaDeviceSynchronize();
}

void ImageProcessing::set_p(float p) { thrust::fill(thrust::device, d_P.begin(), d_P.end(), p); }

void ImageProcessing::update_raw_pixels() { thrust::copy(h_raw_pixels.begin(), h_raw_pixels.end(), d_raw_pixels.begin()); }

void ImageProcessing::update_is_valid(float th) {
  d_update_is_valid<<<m_size.y, m_size.x>>>(thrust::raw_pointer_cast(d_is_valid.data()), thrust::raw_pointer_cast(d_raw_pixels.data()), th);
  cudaDeviceSynchronize();
}

float ImageProcessing::evaluate_error() {
  d_evaluate_error<<<m_size.y, m_size.x>>>(thrust::raw_pointer_cast(d_is_valid.data()), thrust::raw_pointer_cast(d_raw_pixels.data()),
                                           thrust::raw_pointer_cast(d_pixels.data()), thrust::raw_pointer_cast(d_error.data()));
  cudaDeviceSynchronize();
  return thrust::reduce(d_error.begin(), d_error.end(), 0.0f);
}

void ImageProcessing::fill_raw_pixels(float value) {
  // thrust::fill(h_raw_pixels.begin(), h_raw_pixels.end(), value);
  update_raw_pixels();
}
