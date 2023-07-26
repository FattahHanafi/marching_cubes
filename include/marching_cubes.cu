#include <cuda_device_runtime_api.h>
#include <vector_types.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#include "marching_cubes.hpp"

__global__ void d_set_vertices(bool* d_vertices, const bool value) {
  uint32_t idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x +
                 threadIdx.x;
  d_vertices[idx] = value;
}

MarchingCubes::MarchingCubes(const uint32_t x_count, const uint32_t y_count,
                             const uint32_t z_count, const float size)
    : m_count{x_count, y_count, z_count},
      m_cube_size{size, size, size},
      m_size{x_count * size, y_count * size, z_count * size} {
  m_vertices = new std::vector<bool>;
  m_vertices->resize((x_count + 1) * (y_count + 1) * (z_count + 1), false);
  m_cubes = new std::vector<uint8_t>;
  m_cubes->resize(x_count * y_count * z_count, 0);

  m_cudaStat =
      cudaMalloc((void**)&d_vertices,
                 m_vertices->size() * sizeof(typeof(m_vertices->at(0))));
  if (m_cudaStat != cudaSuccess) {
    printf("device memory allocation failed\n");
    exit(CUBLAS_STATUS_ALLOC_FAILED);
  }
  cudaMalloc((void**)&d_cubes,
             m_cubes->size() * sizeof(typeof(m_cubes->at(0))));
}

MarchingCubes::~MarchingCubes() {
  delete m_vertices;
  delete m_cubes;
  cudaFree(d_vertices);
  cudaFree(d_cubes);
}

void MarchingCubes::set_vertex(Vec3<uint32_t>* index, bool value) {
  uint32_t idx = vertex2idx(index);
  m_vertices->at(idx) = value;
}

void MarchingCubes::set_vertices_gpu(const bool value) {
  d_set_vertices<<<dim3(m_count.x, m_count.y), m_count.z>>>(d_vertices, value);
  cudaDeviceSynchronize();
}

void MarchingCubes::set_vertices(Vec3<uint32_t>* index, uint32_t len) {
  index->z = 0;
  uint32_t idx = vertex2idx(index);
  for (uint32_t i = 0; i < m_count.z; ++i) {
    m_vertices->at(idx + i) = i < len;
  }
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
void MarchingCubes::build_cubes() {
  uint8_t cube = 0;
  Vec3<uint32_t> index{0, 0, 0};
  for (uint32_t i = 0; i < m_count.x; ++i)
    for (uint32_t j = 0; j < m_count.y; ++j)
      for (uint32_t k = 0; k < m_count.z; ++k) {
        cube = 0;
        index.x = i;
        index.y = j;
        index.z = k;
        cube += m_vertices->at(vertex2idx(&index)) * (1 << 0);
        index.x = i + 1;
        cube += m_vertices->at(vertex2idx(&index)) * (1 << 1);
        index.y = j + 1;
        cube += m_vertices->at(vertex2idx(&index)) * (1 << 2);
        index.x = i;
        cube += m_vertices->at(vertex2idx(&index)) * (1 << 3);
        index.y = j;
        index.z = k + 1;
        cube += m_vertices->at(vertex2idx(&index)) * (1 << 4);
        index.x = i + 1;
        cube += m_vertices->at(vertex2idx(&index)) * (1 << 5);
        index.y = j + 1;
        cube += m_vertices->at(vertex2idx(&index)) * (1 << 6);
        index.x = i;
        cube += m_vertices->at(vertex2idx(&index)) * (1 << 7);
        index.x = i;
        index.y = j;
        index.z = k;
        m_cubes->at(cube2idx(&index)) = cube;
      }
}

void MarchingCubes::print() {
  m_size.print();
  m_cube_size.print();
  m_size.print();
}
