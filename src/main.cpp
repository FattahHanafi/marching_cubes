#include <iostream>

#include "../include/image_processing.hpp"
#include "../include/marching_cubes.hpp"
#include "../include/timer.hpp"

#define X 200
#define Y 100
#define Z 100

int main() {
  // MarchingCubes mc = MarchingCubes(X, Y, Z, 1.0);
  ImageProcessing ip = ImageProcessing(800, 480, 3, 10);

  for (int i = 0; i < 1; ++i) {
    std::cout << "===========================\n";
    {
      Timer t("Set P", TimeUnit::TIME_UNIT_μS);
      ip.set_p(1.26f);
    }

    {
      Timer t("Set raw", TimeUnit::TIME_UNIT_μS);
      ip.fill_raw_pixels(1.25f);
    }

    {
      Timer t("update mask", TimeUnit::TIME_UNIT_μS);
      ip.update_is_valid(0.0f);
    }

    {
      Timer t("Evaluating Pixels", TimeUnit::TIME_UNIT_μS);
      ip.evaluate_pixels();
    }

    {
      Timer t("Calculate Errors", TimeUnit::TIME_UNIT_μS);
      float er = ip.evaluate_error();
      std::cout << "err = " << er << std::endl;
    }
  }
}
