#include "Utils.h"
#include <iostream>

arax_vaccel_s *getControllVAC() {
  static arax_vaccel_s *vac = 0;

  if (!vac)
    vac = (arax_vaccel_s *)arax_accel_acquire_type(ANY);

  return vac;
}

LogTracer ::LogTracer(std::string msg) : msg(msg) {
  std::cerr << "In  " << msg << std::endl;
}
LogTracer ::~LogTracer() { std::cerr << "Out " << msg << std::endl; }
