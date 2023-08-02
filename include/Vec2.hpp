#pragma once

template <typename T>
class Vec2 {
 public:
  T x = T(0);
  T y = T(0);

  Vec2(T x, T y) : x(x), y(y) {}
};
