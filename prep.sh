#!/usr/bin/env bash

sudo apt-get update
sudo apt install cmake-curses-gui g++

cd arax/build
git checkout controller_integration
cmake -DARAX_OBJECT_NAME_SIZE=128 -DBUILD_TESTS=OFF -Dasync_architecture=spin ..
make -j `nproc`
sudo make install
cd -

cd OneDnn/build
cmake ..
make -j `nproc`
cd -

cd ivshmem-uio
make -j `nproc`
sudo make install
sudo depmod
sudo modprobe uio_ivshmem
