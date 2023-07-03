
python3 configure.py\
                         --prob galpause \
                         --coord spherical_polar \
                         --eos adiabatic \
                         --flux hllc \
                         --cxx icc \
                         --cflag="DH5_HAVE_PARALLEL -std=c++11" \
                         --ccmd /nas/longleaf/apps-dogwood/hdf5/1.10.2/openmpi/bin/h5pcc \
                         --ns 3 \
                         -hdf5 \
                         -mpi \
                         -rec \
                         -exp
                          

