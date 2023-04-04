# OneDNN convolution example

This folder contains an application that uses oneDNN to compute a convolution with fixed arguments.
After compiling with `make', run the application following this bash script.

```bash
make
export DNN_RISCV_DISABLE_LONGVEC=1
export VEHAVE_VECTOR_LENGTH=8192
export VEHAVE_DEBUG_LEVEL=0
LD_PRELOAD=../vehave/libvehave.so ./example
```

The benchdnn driver script sets the DNN_RISCV_DISABLE_LONGVEC environment variable.
This statement disables experimental primitive kernels still under development on oneDNN.
We will remove this variable later but, for now, use it whenever interacting with oneDNN.

The statement regarding the VEHAVE_DEBUG_LEVEL variable, supresses the terminal output of VEHAVE, which can be a lot on large convolutions.
The VEHAVE_VECTOR_LENGTH determines the maximum vector length of the platform.
The DNN_RISCV_DISABLE_LONGVEC disables experimental convolution kernels that are still under development.

If VEHAVE_VECTOR_LENGTH is not set, the application may crash due to segmentation faults.