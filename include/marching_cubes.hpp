#pragma once
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cstdint>
#include <vector>

#include "Vec3.hpp"

class MarchingCubes {
 public:
  MarchingCubes(const uint32_t x_count, const uint32_t y_count,
                const uint32_t z_count, const double size);
  void set_vertex(Vec3<uint32_t>* index, bool value);
  void set_heights_gpu(double height);
  void update_vertices_gpu();
  void update_cubes_gpu();
  double update_volumes_gpu();
  // void set_vertices(Vec3<uint32_t>* index, uint32_t len);
  void print();
  bool get_vertex(const size_t i) const;
  void build_cubes();

  uint32_t add();
  ~MarchingCubes();

 private:
  const Vec3<uint32_t> m_count;
  const Vec3<double> m_cube_size;
  const Vec3<double> m_size;
  std::vector<bool>* m_vertices;
  std::vector<uint8_t>* m_cubes;

  thrust::device_vector<bool> d_vertices;
  thrust::device_vector<uint8_t> d_cubes;
  thrust::device_vector<double> d_heights;
  thrust::device_vector<double> d_volumes;
  thrust::device_vector<double> d_const_volumes;

  cudaError_t m_cudaStat;
  cublasStatus_t m_stat;
  cublasHandle_t m_handle;

  size_t vertex2idx(const Vec3<uint32_t>* index) const;
  size_t cube2idx(const Vec3<uint32_t>* index) const;
};
