
#include <omp.h>
//#include "omptarget.h"
#include <iostream>
  
int main(int argc, char **argv) {
  int ndev = omp_get_num_devices();
  std::cout << "OMP found " << ndev << " available target offload devices" << std::endl;
  int target = ndev - 1;
  int host = omp_get_initial_device();
  int defdev = omp_get_default_device();
  std::cout << "OMP initial host device = " << host << std::endl;
  std::cout << "OMP target device = " << target << std::endl;
  std::cout << "OMP default device = " << defdev << std::endl;
  return 0;
}

