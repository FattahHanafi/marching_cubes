#pragma once
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "Vec2.hpp"

class ImageProcessing {
 public:
  ImageProcessing(const uint32_t width, const uint32_t height, const uint32_t P,
                  const uint32_t N);
  void evaluate_pixels();
  void set_p(float p);

 private:
  const Vec2<uint32_t> m_size;
  const uint32_t m_P;
  const uint32_t m_N;

  thrust::device_vector<float> d_pixels;

  thrust::device_vector<float> d_U;
  thrust::device_vector<float> d_A;
  thrust::device_vector<float> d_P;
};
