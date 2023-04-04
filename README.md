# Create server/client vms/images

`./riscv.sh server`
`./riscv.sh client`

One booted, login to each vm, user & pass:ubuntu, you will be promted to change password.
(Due to qemu, booting will be slow, 

# Enable ssh

`sudo systemctl enable ssh`
`sudo systemctl start ssh`

# Clone dependencies

After that, on the host, run `./clone_deps.sh server` and `./clone_deps.sh client`.
This will clone repos localy and then scp sources and configuration files to the VMs.
You will be asked for the VM password 7-8 times.
(If you recreate the VM images, you might have to delete on host keys from ~/.ssh/known_hosts)

# Build

In each VM, call `bash prep.sh`, this will build arax and OneDnn.

# Run

## Install IVSHMEM module

Insert the ivsheme kernel module:

`sudo modprobe uio_ivshmem`

## IVSHMEM permisions

Use `sudo chmod 777 /dev/uio0` to allow controller and applications to mmap the shared segment.

## Start Controller

`arax_controller conf.json`

Controller should start and print:

```
cpu:CPU:0 :: Initialization successful
cpu:CPU:0 :: Ready
```

## Run example

To test the setup we can run a OneDnn example:
`~/OneDnn/build/tests/getting_started`

It should run and print 'Example passed on CPU.'
