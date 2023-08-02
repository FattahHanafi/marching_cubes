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

__global__ void d_initialize_U(float* d_U, const uint32_t w) {
  uint32_t row = threadIdx.x;
  uint32_t power = threadIdx.y;

  d_U[row * P + power] = pow(row * float(w) / N, power);
}

ImageProcessing::ImageProcessing(const uint32_t w, const uint32_t h)
    : m_size{w, h} {
  d_A.resize((P + 1) * (P + 1));
  d_initialize_A<<<1, 1>>>(thrust::raw_pointer_cast(d_A.data()));
  d_U.resize((m_size.x / N) * (P + 1));
  d_initialize_U<<<1, dim3(m_size.x, P + 1)>>>(
      thrust::raw_pointer_cast(d_U.data()), m_size.y);
  d_P.resize(h * (N * P + 1));
}
