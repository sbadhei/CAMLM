echo "Setting up the environment, it will take few minutes to complete........"
echo "Y" | sudo apt-get update
if test $? -eq 0
then
	echo "apt-get update [ok]"
else
	echo "apt-get update [Not ok]"
fi
echo "Y" | sudo apt-get install python3-pip python3-dev 
if test $? -eq 0
then
	echo "Installation of python3-pip python3-dev [ok]"
else
	echo "Installation of python3-pip python3-dev [Not ok]"
        exit 1
fi
echo "Y" | sudo apt install python3
if test $? -eq 0
then
	echo "Installation python3 [ok]"
else
	echo "Installation python3 [Not ok]"
        exit 1
fi
echo "Y" | sudo apt-get install build-essential cmake git unzip pkg-config libopenblas-dev  liblapack-dev
if test $? -eq 0
then
	echo "Installation of build-essential cmake git unzip pkg-config libopenblas-dev liblapack-dev [ok]"
else
	echo "Installation of build-essential cmake git unzip pkg-config libopenblas-dev liblapack-dev [Not ok]"
        exit 1
fi
echo "Y" | sudo apt-get install python3-numpy python3-scipy python3-matplotlib
if test $? -eq 0
then
	echo "Installation of python3-numpy python3-scipy python3-matplotlib [ok]"
else
	echo "Installation of python3-numpy python3-scipy python3-matplotlib [Not ok]"
        exit 1
fi
echo "Y" | pip3 install pandas
if test $? -eq 0
then
	echo "Installation of pandas [ok]"
else
	echo "Installation of pandas [Not ok]"
        exit 1
fi
echo "Y" | sudo pip3 install tensorflow-gpu
if test $? -eq 0
then
	echo "Installation of tensorflow-gpu [ok]"
else
	echo "Installation of tensorflow-gpu [Not ok]"
        exit 1
fi
echo "Y" | sudo apt-get install libhdf5-serial-dev
if test $? -eq 0
then
	echo "Installation of libhdf5-serial-dev [ok]"
else
	echo "Installation of libhdf5-serial-dev [Not ok]"
        exit 1
fi
echo "Y" | sudo apt-get install python3-h5py
if test $? -eq 0
then
	echo "Installation of python3-h5py [ok]"
else
	echo "Installation of python3-h5py [Not ok]"
        exit 1
fi
echo "Y" | pip install -U scikit-learn
if test $? -eq 0
then
	echo "Installation of scikit-learn [ok]"
else
	echo "Installation of scikit-learn [Not ok]"
        exit 1
fi
echo "Y" | sudo apt-get install nmon
if test $? -eq 0
then
	echo "Installation of nmon [ok]"
else
	echo "Installation of nmon [Not ok]"
        exit 1
fi
ln -s /usr/lib/x86_64-linux-gnu/libcudart.so /lib64/libcudart.so.11.0
if test $? -eq 0
then
	echo "Created symbolic link for libcudart.so.11.0 [ok]"
else
	echo "Created symbolic link for libcudart.so.11.0 [Not ok]"
fi
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/lib64
echo "Set up is ready to perform the experiment....."
python3 fmnist.py
