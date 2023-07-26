#pragma once
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "Vec3.hpp"

class MarchingCubes {
 public:
  MarchingCubes(const uint32_t x_count, const uint32_t y_count,
                const uint32_t z_count, const float size);
  void set_vertex(Vec3<uint32_t>* index, bool value);
  void set_vertices_gpu(const bool value);
  // __global__ void d_set_vertices(bool* d_vertices, const bool value);
  void set_vertices(Vec3<uint32_t>* index, uint32_t len);
  void print();
  bool get_vertex(const size_t i) const;
  void build_cubes();

  ~MarchingCubes();

 private:
  const Vec3<uint32_t> m_count;
  const Vec3<float> m_cube_size;
  const Vec3<float> m_size;
  std::vector<bool>* m_vertices;
  std::vector<uint8_t>* m_cubes;

  bool* d_vertices;
  uint8_t* d_cubes;

  cudaError_t m_cudaStat;
  cublasStatus_t m_stat;
  cublasHandle_t m_handle;

  size_t vertex2idx(const Vec3<uint32_t>* index) const;
  size_t cube2idx(const Vec3<uint32_t>* index) const;
};
