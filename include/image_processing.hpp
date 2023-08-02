#pragma once
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <cstdint>

#include "Vec2.hpp"
const uint32_t P = 3;
const uint32_t N = 10;

class ImageProcessing {
 public:
  ImageProcessing(const uint32_t width, const uint32_t height);

 private:
  const Vec2<uint32_t> m_size;
  thrust::device_vector<float> d_values;

  thrust::device_vector<float> d_U;
  thrust::device_vector<float> d_A;
  thrust::device_vector<float> d_P;
};
