#include "marching_cubes.hpp"

#include <algorithm>
#include <cstdint>

MarchingCubes::MarchingCubes(const uint32_t x_count, const uint32_t y_count, const uint32_t z_count, const float size) {
  m_count.set_x(x_count);
  m_count.set_y(y_count);
  m_count.set_z(z_count);

  m_cube_size.set_all(size);

  m_size.set_x(x_count * size);
  m_size.set_y(y_count * size);
  m_size.set_z(z_count * size);

  m_vertices.resize((x_count + 1) * (y_count + 1) * (z_count + 1), false);
}

void MarchingCubes::set_vertex(Vec3<uint32_t>* index, bool value) {
  uint32_t idx = vec2idx(index);
  m_vertices[idx] = value;
}
const uint32_t MarchingCubes::vec2idx(Vec3<uint32_t>* index) {
  return index->z() + index->y() * (m_count.z() + 1) + index->x() * ((m_count.z() + 1) * (m_count.y() + 1));
}

const void MarchingCubes::print() {
  m_size.print();
  m_cube_size.print();
  m_size.print();
}
