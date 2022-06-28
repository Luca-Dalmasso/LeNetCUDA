# LeNetCUDA

The purpose of this project is to accelerate a simple Convolutional Neural Network forward propagation algorithm on a Nvidia GPU and to show what are the possible architectural choices that can be used to speed up a code running on a GPU.

## Project overview

The following folders contains the source code:
<br>
[header files](./inc)
<br>
[source files](./src)
<br>
[main application](./LeNet.cu)
<br>
[The profiling script](./profile_app.sh) is the one used for profiling the application, from it is possible to select different metrics and events to be profiled using **nvprof** command line profiler available in the *NVIDIA TOOLKIT*. The script will generate three files (_exhautive, _medium, _light)containing more and more detailed profiling information about the application. [Here](./tmp/) and [Here](./rept/) you can find some examples.<br>
More details about the project can be found in the [report](./report.pdf).

## Hoe to compile, run & profile

If you want to use the application you need to install the Nvidia Toolkit on your machine and of course have a Nvidia GPU available.<br>
Instructions on how to install the Toolkit can be found [here](https://docs.nvidia.com/cuda/index.html).

### compile

You can compile the sourcefiles using [make](./makefile) <br>

    make

and clean the compilation files using

    make clean

I suggest to change the following flag  *GPU_ARCHITECTURE=sm_53* according to your GPU. The above flag is suited for my **NVIDIA Jetson NANO board** with a **Tegra X1** GPU which is a Mawxell architecture.
<br>
Higly suggest to take a look at the references i used for the CNN documentation, you can find them in the report.
