
python3 configure.py\
                         --prob galpause \
                         --coord spherical_polar \
                         --eos adiabatic \
                         --flux hllc \
                         --cxx g++ \
                         #--ccmd /nas/longleaf/apps/hdf5/1.10.4/bin/h5cc \
                         --cflag="-std=c++11" \
                         --ns 2 \
                         -hdf5 \
                         -de \
                         -exp
                          

