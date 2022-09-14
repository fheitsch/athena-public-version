python3 configure.py\
                         --prob blast \
                         --coord cartesian \
                         --eos adiabatic \
                         --flux hllc \
                         --cxx icc \
                         --cflag="DH5_HAVE_PARALLEL -std=c++11" \
                         --ccmd /nas/longleaf/apps-dogwood/hdf5/1.10.2/openmpi/bin/h5pcc \
                         --nghost=4 \
                         -hdf5 \
                         -mpi

