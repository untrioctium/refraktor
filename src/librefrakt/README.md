# `librefrakt` - real-time flame fractal rendering with CUDA
`librefrakt` is a C++/CUDA library for rendering fractal flames with real-time performance. The goals of this project are:
* __Reusability__ - Beyond the dependency on CUDA, the library should not be tied to any operating system, graphics API, windowing system, framework, or compiler; it should also be easy to integrate into any C++ application.
* __Tweakability__ - The underlying CUDA code should be data to be modified and improved, even at run-time. New variations can be added and existing ones edited without rebuilding the library or the application.
* __Portability__ - 

## Extensions to the fractal flame algorithm
`librefrakt` is designed to be compatible with existing `flam3` flames while adding a few generalizations and extensions. The main change is the vChain system, which generalizes pre- and post- variations; an Xform consists of one ore more functions called vLinks that are called sequentially and have their own affine transform and set of variations. This allows using any variation in a pre- or post- style 

## Denoising with Optix
NVIDIA's Optix library includes a denoiser for raytraced images 

## Glossary
* _Efficiency_ - The percent of iteration passes that resulted in a draw on the accumulator. 
* _Quality_ - Average number of samples output per pixel on the image. A 1280x720 image with quality 100 means that 1280 * 720 * 100 samples were drawn on the image.
* _vChain_ - A collection of functions in an Xform that compute the new position of an iterator.
* _vLink_ - An individual function in a vChain. It consists of an affine transform and then a summation of any number of weighted variations. It is essentially what an Xform is in traditional flame renderers.