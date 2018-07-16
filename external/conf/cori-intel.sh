loadedgnu=`${MODULESHOME}/bin/modulecmd sh -t list 2>&1 | grep PrgEnv-gnu`
loadedintel=`${MODULESHOME}/bin/modulecmd sh -t list 2>&1 | grep PrgEnv-intel`
loadedcray=`${MODULESHOME}/bin/modulecmd sh -t list 2>&1 | grep PrgEnv-cray`
loadeddarshan=`${MODULESHOME}/bin/modulecmd sh -t list 2>&1 | grep darshan`
if [ "x${loadedintel}" = x ]; then
    if [ "x${loadedcray}" != x ]; then
      module swap PrgEnv-cray PrgEnv-intel
    fi
    if [ "x${loadedgnu}" != x ]; then
      module swap PrgEnv-gnu PrgEnv-intel
    fi
fi
module swap intel intel/18.0.1.163
module load gcc/7.3.0
module load git
module load cmake/3.8.2
export CRAYPE_LINK_TYPE=dynamic

