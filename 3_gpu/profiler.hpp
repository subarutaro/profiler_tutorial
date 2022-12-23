#include <iostream>
#include <string>
#include <chrono>

class Profiler{
  std::chrono::time_point<std::chrono::steady_clock> t0,t1;
  double duration;
  std::string name;
public:
  Profiler(std::string name_) : name(name_) { clear(); }
  void start(){
    t0 = std::chrono::steady_clock::now();
  }
  void end(){
    t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = t1 - t0;
    duration += diff.count();
  }
  void clear(){
    duration = 0.0;
  }
  double getTime() const {
    return duration;
  }
  void print(std::ostream& os = std::cout) {
    os << name << " " << duration << std::endl;
  }
};
