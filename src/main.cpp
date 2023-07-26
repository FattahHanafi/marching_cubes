#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>

#include "../include/marching_cubes.hpp"

#define X 800
#define Y 200
#define Z 100

int main() {
  auto mc = MarchingCubes(X, Y, Z, 0.1f);

  Vec3<uint32_t> idx{0, 0, 0};
  idx.z = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < X + 1; ++i) {
    idx.x = i;
    for (uint32_t j = 0; j < Y + 1; ++j) {
      idx.y = j;
      mc.set_vertices(&idx, Z / 2);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  auto t3 = std::chrono::high_resolution_clock::now();
  mc.build_cubes();
  auto t4 = std::chrono::high_resolution_clock::now();
  auto duration2 =
      std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

  auto t5 = std::chrono::high_resolution_clock::now();
  mc.set_vertices_gpu(true);
  auto t6 = std::chrono::high_resolution_clock::now();

  auto duration3 =
      std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

  std::cout << "It took : " << duration << " ms\n";
  std::cout << "It took : " << duration2 << " ms\n";
  std::cout << "It took : " << duration3 << " ms\n";
}
