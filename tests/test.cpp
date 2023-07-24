#include <Vec3.hpp>
#include <iostream>
#include <marching_cubes.hpp>

#include "marching_cubes.hpp"

int main(int argc, char *argv[]) {
  auto v = Vec3<float>();
  v.set_all(3.5f);
  v.print();

  auto mc = MarchingCubes(3, 3, 3, 1.5f);
  mc.print();
  return 0;
}
