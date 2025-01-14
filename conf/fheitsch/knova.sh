
python3 configure.py\
                         --prob knova \
                         --coord cartesian \
                         --eos adiabatic \
                         --flux hlle \
                         --cxx icc \
                         --cflag="DH5_HAVE_PARALLEL -std=c++11" \
                         --ccmd /nas/longleaf/apps-dogwood/hdf5/1.10.2/openmpi/bin/h5pcc \
                         --ns 3 \
                         --ng 4 \
                         -exp \
                         -de \
                         -mpi \
                         -hdf5 
                          

