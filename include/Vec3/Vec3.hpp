#pragma once
#include <cstdint>
#include <iostream>
#include <vector>

template <typename T>
class Vec3 {
 public:
  Vec3();
  Vec3(const T x, const T y, const T z);

  const T x();
  const T y();
  const T z();

  void set_x(const T x);
  void set_y(const T y);
  void set_z(const T z);
  void set_all(const T s);

  const void print();

 private:
  std::vector<T> m_values;
};

template <typename T>
Vec3<T>::Vec3() {
  m_values.resize(3);
  m_values.at(0) = T(0);
  m_values.at(1) = T(0);
  m_values.at(2) = T(0);
}

template <typename T>
Vec3<T>::Vec3(const T x, const T y, const T z) {
  m_values.resize(3);
  m_values.at(0) = x;
  m_values.at(1) = y;
  m_values.at(2) = z;
}

template <typename T>
const T Vec3<T>::x() {
  return m_values.at(0);
}

template <typename T>
const T Vec3<T>::y() {
  return m_values.at(1);
}

template <typename T>
const T Vec3<T>::z() {
  return m_values.at(2);
}

template <typename T>
void Vec3<T>::set_x(const T x) {
  m_values.at(0) = x;
}

template <typename T>
void Vec3<T>::set_y(const T y) {
  m_values.at(1) = y;
}

template <typename T>
void Vec3<T>::set_z(const T z) {
  m_values.at(2) = z;
}

template <typename T>
void Vec3<T>::set_all(const T s) {
  m_values.at(0) = s;
  m_values.at(1) = s;
  m_values.at(2) = s;
}

template <typename T>
const void Vec3<T>::print() {
  std::cout << "x : " << m_values.at(0) << ", y : " << m_values.at(1) << ", z : " << m_values.at(2) << std::endl;
}
