# Easy to edit configure script for athena++ 
python configure.py\
			 --prob blast\
			 --flux hllc\
			 --include /usr/include/hdf5/serial/ \
			 --nghost=3 \
			 -hdf5 \
			 -de 
#			 --hdf5_path=/usr/lib/x86_64-linux-gnu/