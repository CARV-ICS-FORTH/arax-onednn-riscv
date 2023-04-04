#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

NAME=$1

OS_IMAGE_URL=https://cdimage.ubuntu.com/releases/22.10/release/ubuntu-22.10-preinstalled-server-riscv64+unmatched.img.xz
OS_IMAGE_COMPRESSED=$(basename $OS_IMAGE_URL)
OS_IMAGE=${OS_IMAGE_COMPRESSED%.xz}

if [ ! -f ${OS_IMAGE_COMPRESSED} ]; then
    echo "Downloading compressed OS image..."
    curl -OL ${OS_IMAGE_URL}
fi

if [ ! -f ${OS_IMAGE} ]; then
    echo "Uncompressing OS image..."
    xz -dk ${OS_IMAGE_COMPRESSED}
fi

if [ ! -f ${NAME}.img ]; then
    echo "Copying OS iamge to {NAME}.img"
    cp ${OS_IMAGE} ${NAME}.img
    qemu-img resize -f raw ${NAME}.img +8G
fi

# The following files were copied over from an Ubuntu install:
# * fw_jump.elf from /usr/lib/riscv64-linux-gnu/opensbi/generic/fw_jump.elf
# * uboot.elf from /usr/lib/u-boot/qemu-riscv64_smode/uboot.elf

HOST_PORT=8888

if [ "$NAME" == "client" ]; then
	HOST_PORT=9999
fi

qemu-system-riscv64 \
    -machine virt -nographic -m 8192 -smp 4 \
    -bios fw_jump.elf \
    -kernel uboot.elf \
    -device virtio-net-device,netdev=eth0 -netdev user,id=eth0,hostfwd=tcp::${HOST_PORT}-:22 \
    -drive file=${NAME}.img,format=raw,if=virtio \
    -object memory-backend-file,size=64M,share,mem-path=/dev/shm/ivshmem,id=hostmem \
    -device ivshmem-plain,memdev=hostmem 
