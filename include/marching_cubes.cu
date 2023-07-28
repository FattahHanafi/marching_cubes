#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <sys/types.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#include "marching_cubes.hpp"

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
                                 const double* d_const_volumes) {
  uint32_t cube_idx = blockIdx.x * gridDim.y * blockDim.x +
                      blockIdx.y * blockDim.x + threadIdx.x;
  d_volumes[cube_idx] = d_volumes[d_cubes[cube_idx]];
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
  d_const_volumes.resize(256);

  thrust::fill(d_heights.begin(), d_heights.end(), 0.0f);
  thrust::fill(d_vertices.begin(), d_vertices.end(), false);
  thrust::fill(d_cubes.begin(), d_cubes.end(), 0);
  thrust::fill(d_volumes.begin(), d_volumes.end(), 0.0);
  thrust::fill(d_volumes.begin(), d_volumes.end(), 0.5);  // UPDATE Values
}

MarchingCubes::~MarchingCubes() {}

void MarchingCubes::set_vertex(Vec3<uint32_t>* index, bool value) {
  uint32_t idx = vertex2idx(index);
  m_vertices->at(idx) = value;
}

uint32_t MarchingCubes::add() {
  return thrust::reduce(thrust::device, d_vertices.begin(), d_vertices.end(),
                        0);
}

bool MarchingCubes::get_vertex(const size_t i) const {
  return m_vertices->at(i);
}

size_t MarchingCubes::vertex2idx(const Vec3<uint32_t>* index) const {
  size_t idx = index->z;
  idx += index->y * (m_count.z + 1);
  idx += index->x * (m_count.z + 1) * (m_count.y + 1);
  return idx;
}

size_t MarchingCubes::cube2idx(const Vec3<uint32_t>* index) const {
  size_t idx = index->z;
  idx += index->y * m_count.z;
  idx += index->x * m_count.z * m_count.y;
  return idx;
}

void MarchingCubes::set_heights_gpu(double height) {
  thrust::fill(d_heights.begin(), d_heights.end(), height);
}

double MarchingCubes::update_volumes_gpu() {
  d_update_volumes<<<dim3(m_count.x, m_count.y, 1), m_count.z>>>(
      thrust::raw_pointer_cast(d_volumes.data()),
      thrust::raw_pointer_cast(d_cubes.data()),
      thrust::raw_pointer_cast(d_const_volumes.data()));
  cudaDeviceSynchronize();
  return thrust::reduce(d_volumes.begin(), d_volumes.end(), 0.0);
}

void MarchingCubes::update_vertices_gpu() {
  d_update_vertices<<<dim3(m_count.x + 1, m_count.y + 1, 1), m_count.z + 1>>>(
      thrust::raw_pointer_cast(d_vertices.data()),
      thrust::raw_pointer_cast(d_heights.data()), m_cube_size.z);
  cudaDeviceSynchronize();
}

void MarchingCubes::update_cubes_gpu() {
  d_update_cubes<<<dim3(m_count.x, m_count.y, 1), m_count.z>>>(
      thrust::raw_pointer_cast(d_cubes.data()),
      thrust::raw_pointer_cast(d_vertices.data()));
  cudaDeviceSynchronize();
}

void MarchingCubes::print() {
  m_size.print();
  m_cube_size.print();
  m_size.print();
}
