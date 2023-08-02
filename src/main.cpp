#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>

#include "../include/marching_cubes.hpp"
#include "../include/timer.hpp"

#define X 200
#define Y 100
#define Z 100

int main() {
  MarchingCubes mc = MarchingCubes(X, Y, Z, 1.0);

  for (int i = 0; i < 10; ++i) {
    std::cout << "===========================\n";
    {
      Timer t("Set height", TimeUnit::TIME_UNIT_μS);
      mc.set_heights_gpu(0.5);
    }

    {
      Timer t("Update vertices", TimeUnit::TIME_UNIT_μS);
      mc.update_vertices_gpu();
    }

    {
      Timer t("Update cubes", TimeUnit::TIME_UNIT_μS);
      mc.update_cubes_gpu();
    }

    {
      Timer t("Update volumes", TimeUnit::TIME_UNIT_μS);
      double vol = mc.update_volumes_gpu();
    }

    {
      Timer t("Update triangles", TimeUnit::TIME_UNIT_μS);
      mc.update_triangles_gpu();
    }

    {
      Timer t("Shrink triangles", TimeUnit::TIME_UNIT_μS);
      mc.shrink_triangles_gpu();
    }

    {
      Timer t("Copy triangles", TimeUnit::TIME_UNIT_μS);
      mc.copy_triangles_from_gpu();
    }
  }
}
