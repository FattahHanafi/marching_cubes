#pragma once
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <algorithm>

#include "Vec2.hpp"

class ImageProcessing {
 public:
  ImageProcessing(const uint32_t width, const uint32_t height, const uint32_t P, const uint32_t N);
  void evaluate_pixels();
  void set_p(float p);
  void update_raw_pixels();
  void update_is_valid(float th);
  float evaluate_error();

  void fill_raw_pixels(float value);

 private:
  const Vec2<uint32_t> m_size;
  const uint32_t m_P;
  const uint32_t m_N;

  thrust::device_vector<float> d_pixels;
  thrust::device_vector<bool> d_is_valid;
  thrust::device_vector<float> d_raw_pixels;
  thrust::device_vector<float> d_J;
  thrust::device_vector<float> d_error;

  thrust::host_vector<float> h_raw_pixels;

  thrust::device_vector<float> d_U;
  thrust::device_vector<float> d_A;
  thrust::device_vector<float> d_P;
};
