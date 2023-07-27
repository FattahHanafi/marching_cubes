#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>

#include "../include/marching_cubes.hpp"
#include "../include/timer.hpp"

#define X 1000
#define Y 150
#define Z 100

int main() {
  MarchingCubes mc = MarchingCubes(X, Y, Z, 2.0);

  for (uint32_t i = 0; i < 1000; ++i) {
    {
      Timer t("set height", TimeUnit::TIME_UNIT_μS);
      mc.set_heights_gpu(1.1);
    }

    {
      Timer t("update vertices", TimeUnit::TIME_UNIT_μS);
      // mc.update_vertices_gp();
    }

    {
      Timer t("update cubes", TimeUnit::TIME_UNIT_μS);
      // mc.update_cubes();
    }
  }
}
