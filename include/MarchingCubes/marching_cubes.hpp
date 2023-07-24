#pragma once
#include <Vec3.hpp>
#include <cstdint>
#include <vector>

class MarchingCubes {
 public:
  MarchingCubes(const uint32_t x_count, const uint32_t y_count, const uint32_t z_count, const float size);
  void set_vertex(Vec3<uint32_t>* index, bool value);

  const void print();

 private:
  Vec3<uint32_t> m_count;
  Vec3<float> m_cube_size;
  Vec3<float> m_size;
  std::vector<bool> m_vertices;

  const uint32_t vec2idx(Vec3<uint32_t>* index);
};
