#pragma once

template <typename T>
class Vec3 {
 public:
  T x = T(0);
  T y = T(0);
  T z = T(0);

  Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
};
