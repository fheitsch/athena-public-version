python3 configure.py\
                         --prob exp_advect \
                         --coord cartesian \
                         --eos adiabatic \
                         --flux hlld \
                         --cxx icc \
                         --cflag="DH5_HAVE_PARALLEL -std=c++11" \
                         --ccmd /nas/longleaf/apps-dogwood/hdf5/1.10.2/openmpi/bin/h5pcc \
                         --ng 4 \
                         -hdf5 \
                         -debug \
                         -exp \
                         -b

