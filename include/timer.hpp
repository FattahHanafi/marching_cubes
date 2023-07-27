#include <chrono>
#include <iostream>
#include <string>

enum class TimeUnit {
  TIME_UNIT_S,
  TIME_UNIT_MS,
  TIME_UNIT_μS,
  TIME_UNIT_NS,
};

class Timer {
 public:
  Timer(std::string name, TimeUnit time_unit)
      : m_name(name), m_time_unit(time_unit) {
    this->reset();
  }

  ~Timer() {
    stop();
    print();
  }

  void reset() {
    m_start = std::chrono::steady_clock::now();
    m_end = m_start;
  }

  void stop() { m_end = std::chrono::steady_clock::now(); }

  void print() {
    switch (m_time_unit) {
      case TimeUnit::TIME_UNIT_S:
        m_duration =
            std::chrono::duration_cast<std::chrono::seconds>(m_end - m_start)
                .count();
        std::cout << m_name << " took " << m_duration << " s\n";
        break;
      case TimeUnit::TIME_UNIT_MS:
        m_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                         m_end - m_start)
                         .count();
        std::cout << m_name << " took " << m_duration << " ms\n";
        break;
      case TimeUnit::TIME_UNIT_μS:
        m_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                         m_end - m_start)
                         .count();
        std::cout << m_name << " took " << m_duration << " μs\n";
        break;
      case TimeUnit::TIME_UNIT_NS:
        m_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         m_end - m_start)
                         .count();
        std::cout << m_name << " took " << m_duration << " ns\n";
        break;
    }
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> m_start;
  std::chrono::time_point<std::chrono::steady_clock> m_end;
  TimeUnit m_time_unit;
  long long m_duration = -1;
  std::string m_name;
};
