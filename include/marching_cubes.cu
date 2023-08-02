#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sys/types.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/detail/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <ostream>
#include <vector>

#include "marching_cubes.hpp"

struct is_valid {
  __device__ bool operator()(const float x) { return (x > 0); }
};

__global__ void d_set_zero(float* d_triangles) {
  uint32_t idx = 15 * (blockIdx.x * gridDim.x * blockDim.x);
  for (uint32_t i = 0; i < 15; ++i) {
    d_triangles[idx + i] = 0.0f;
  }
}

__global__ void d_update_vertices(bool* d_vertices, double* d_heights,
                                  const double size) {
  uint32_t vertex_idx = blockIdx.x * gridDim.y * blockDim.x +
                        blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t height_idx = blockIdx.x * gridDim.y + blockIdx.y;
  d_vertices[vertex_idx] = (threadIdx.x * size) < d_heights[height_idx];
}

__global__ void d_update_cubes(uint8_t* d_cubes, bool* d_vertices) {
  uint32_t cube_idx = blockIdx.x * gridDim.y * blockDim.x +
                      blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t vertex_idx[8];

  vertex_idx[0] = (blockIdx.x + 0) * (gridDim.y + 1) * (blockDim.x + 1) +
                  (blockIdx.y + 0) * (blockDim.x + 1) + (threadIdx.x + 0);
  vertex_idx[1] = (blockIdx.x + 1) * (gridDim.y + 1) * (blockDim.x + 1) +
                  (blockIdx.y + 0) * (blockDim.x + 1) + (threadIdx.x + 0);
  vertex_idx[2] = (blockIdx.x + 1) * (gridDim.y + 1) * (blockDim.x + 1) +
                  (blockIdx.y + 1) * (blockDim.x + 1) + (threadIdx.x + 0);
  vertex_idx[3] = (blockIdx.x + 0) * (gridDim.y + 1) * (blockDim.x + 1) +
                  (blockIdx.y + 1) * (blockDim.x + 1) + (threadIdx.x + 0);
  vertex_idx[4] = (blockIdx.x + 0) * (gridDim.y + 1) * (blockDim.x + 1) +
                  (blockIdx.y + 0) * (blockDim.x + 1) + (threadIdx.x + 1);
  vertex_idx[5] = (blockIdx.x + 1) * (gridDim.y + 1) * (blockDim.x + 1) +
                  (blockIdx.y + 0) * (blockDim.x + 1) + (threadIdx.x + 1);
  vertex_idx[6] = (blockIdx.x + 1) * (gridDim.y + 1) * (blockDim.x + 1) +
                  (blockIdx.y + 1) * (blockDim.x + 1) + (threadIdx.x + 1);
  vertex_idx[7] = (blockIdx.x + 0) * (gridDim.y + 1) * (blockDim.x + 1) +
                  (blockIdx.y + 1) * (blockDim.x + 1) + (threadIdx.x + 1);

  uint8_t cube = 0;

  for (uint8_t i = 0; i < 8; ++i) cube += (1 << i) * d_vertices[vertex_idx[i]];
  d_cubes[cube_idx] = cube;
}

__global__ void d_update_volumes(double* d_volumes, const uint8_t* d_cubes,
                                 const double* d_mc_volumes,
                                 const double cube_volume) {
  uint32_t cube_idx = blockIdx.x * gridDim.y * blockDim.x +
                      blockIdx.y * blockDim.x + threadIdx.x;
  d_volumes[cube_idx] = d_mc_volumes[d_cubes[cube_idx]] * cube_volume;
}

__global__ void d_update_triangles(float* d_triangles,
                                   uint8_t* d_triangles_number,
                                   const uint8_t* d_cubes,
                                   int8_t* d_mc_triangles, const double x_size,
                                   const double y_size, const double z_size) {
  uint32_t cube_idx = blockIdx.x * gridDim.y * blockDim.x +
                      blockIdx.y * blockDim.x + threadIdx.x;
  const uint8_t cube_type = d_cubes[cube_idx];

  int8_t* v = &d_mc_triangles[cube_type * 16];
  float x = x_size;
  float y = y_size;
  float z = z_size;

  uint8_t j = 0;
  while (v[j] != -1) {
    switch (v[j]) {
      case 0:
        x *= 0.5;
        y *= 0.0;
        z *= 0.0;
        break;
      case 1:
        x *= 1.0;
        y *= 0.5;
        z *= 0.0;
        break;
      case 2:
        x *= 0.5;
        y *= 1.0;
        z *= 0.0;
        break;
      case 3:
        x *= 0.0;
        y *= 0.5;
        z *= 0.0;
        break;
      case 4:
        x *= 0.5;
        y *= 0.0;
        z *= 1.0;
        break;
      case 5:
        x *= 1.0;
        y *= 0.5;
        z *= 1.0;
        break;
      case 6:
        x *= 0.5;
        y *= 1.0;
        z *= 1.0;
        break;
      case 7:
        x *= 0.0;
        y *= 0.5;
        z *= 1.0;
        break;
      case 8:
        x *= 0.0;
        y *= 0.0;
        z *= 0.5;
        break;
      case 9:
        x *= 1.0;
        y *= 0.0;
        z *= 0.5;
        break;
      case 10:
        x *= 1.0;
        y *= 1.0;
        z *= 0.5;
        break;
      case 11:
        x *= 0.0;
        y *= 1.0;
        z *= 0.5;
        break;
    }
    d_triangles[cube_idx * 5 * 3 * 3 + j * 3 + 0] = blockIdx.x * x_size + x;
    d_triangles[cube_idx * 5 * 3 * 3 + j * 3 + 1] = blockIdx.y * y_size + y;
    d_triangles[cube_idx * 5 * 3 * 3 + j * 3 + 2] = threadIdx.x * z_size + z;
    ++j;
  }
  d_triangles_number[cube_idx] = j;
}

MarchingCubes::MarchingCubes(const uint32_t x_count, const uint32_t y_count,
                             const uint32_t z_count, const double size)
    : m_count{x_count, y_count, z_count},
      m_cube_size{size, size, size},
      m_size{x_count * size, y_count * size, z_count * size} {
  d_vertices.resize((x_count + 1) * (y_count + 1) * (z_count + 1));
  d_cubes.resize(x_count * y_count * z_count);
  d_heights.resize((x_count + 1) * (y_count + 1));
  d_volumes.resize(x_count * y_count * z_count);
  d_triangles.resize(
      m_count.x * m_count.y * m_count.z * 5 * 3 *
      3);  // up to 5 faces each containing 3 vertices each containg x,y,z
  d_triangles_number.resize(m_count.x * m_count.y * m_count.z);

  d_mc_volumes.resize(m_mc_volumes.size());
  thrust::copy(m_mc_volumes.cbegin(), m_mc_volumes.cend(), d_mc_volumes.data());
  m_mc_volumes.clear();
  m_mc_volumes.shrink_to_fit();

  d_mc_triangles.resize(m_mc_triangles.size());
  thrust::copy(m_mc_triangles.begin(), m_mc_triangles.end(),
               d_mc_triangles.data());
  m_mc_triangles.clear();
  m_mc_triangles.shrink_to_fit();

  thrust::fill(d_heights.begin(), d_heights.end(), 0.0f);
  thrust::fill(d_vertices.begin(), d_vertices.end(), false);
  thrust::fill(d_cubes.begin(), d_cubes.end(), 0);
  thrust::fill(d_volumes.begin(), d_volumes.end(), 0.0);
}

void MarchingCubes::set_heights_gpu(double height) {
  thrust::fill(d_heights.begin(), d_heights.end(), height);
}

double MarchingCubes::update_volumes_gpu() {
  d_update_volumes<<<dim3(m_count.x, m_count.y, 1), m_count.z>>>(
      thrust::raw_pointer_cast(d_volumes.data()),
      thrust::raw_pointer_cast(d_cubes.data()),
      thrust::raw_pointer_cast(d_mc_volumes.data()), m_cube_size.x);
  cudaDeviceSynchronize();
  return thrust::reduce(d_volumes.begin(), d_volumes.end(), double(0));
}

void MarchingCubes::update_vertices_gpu() {
  d_update_vertices<<<dim3(m_count.x + 1, m_count.y + 1, 1), m_count.z + 1>>>(
      thrust::raw_pointer_cast(d_vertices.data()),
      thrust::raw_pointer_cast(d_heights.data()),
      m_cube_size.x * m_cube_size.y * m_cube_size.z);
  cudaDeviceSynchronize();
}

void MarchingCubes::update_cubes_gpu() {
  d_update_cubes<<<dim3(m_count.x, m_count.y, 1), m_count.z>>>(
      thrust::raw_pointer_cast(d_cubes.data()),
      thrust::raw_pointer_cast(d_vertices.data()));
  cudaDeviceSynchronize();
}

void MarchingCubes::update_triangles_gpu() {
  d_update_triangles<<<dim3(m_count.x, m_count.y, 1), m_count.z>>>(
      thrust::raw_pointer_cast(d_triangles.data()),
      thrust::raw_pointer_cast(d_triangles_number.data()),
      thrust::raw_pointer_cast(d_cubes.data()),
      thrust::raw_pointer_cast(d_mc_triangles.data()), m_cube_size.x,
      m_cube_size.y, m_cube_size.z);
  cudaDeviceSynchronize();
}

void MarchingCubes::shrink_triangles_gpu() {
  set_zero_kernel();
  auto s = thrust::reduce(thrust::device, d_triangles_number.begin(),
                          d_triangles_number.end(), 0);
  d_triangles_shrinked.resize(s * 3);  // 3 vertex per each face
  thrust::copy_if(thrust::device, d_triangles.cbegin(), d_triangles.cend(),
                  d_triangles_shrinked.begin(), is_valid());
}

void MarchingCubes::copy_triangles_from_gpu() {
  h_triangles_shrinked.resize(d_triangles_shrinked.size());
  thrust::copy(d_triangles_shrinked.cbegin(), d_triangles_shrinked.cend(),
               h_triangles_shrinked.begin());
}

void MarchingCubes::set_zero_kernel() {
  d_set_zero<<<dim3(m_count.x, m_count.y, 1), m_count.z>>>(
      thrust::raw_pointer_cast(d_triangles.data()));
  cudaDeviceSynchronize();
}
