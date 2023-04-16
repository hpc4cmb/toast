
#include <iostream>
#include <omp.h>

int main() {
  double s;
#pragma omp target teams distribute parallel for reduction(+:s) map(tofrom:s)
  for (int idx = 0; idx < 1000; ++idx) s+= idx;
  std::cout << s << std::endl;
}

