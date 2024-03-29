# `librefrakt` - real-time flame fractal rendering with CUDA
`librefrakt` is a C++ library for rendering fractal flames with real-time performance. The goals of this project are:
* __Reusability__ - The library should not be tied to any operating system, graphics API, windowing system, framework, or compiler; it should also be easy to integrate into any C++ application that uses CMake.
* __Tweakability__ - The underlying flame iteration code should be data to be modified and improved, even at run-time. New variations can be added and existing ones edited without rebuilding the library or the application.
* __Portability__ - 

## Extensions to the fractal flame algorithm
`librefrakt` is designed to be compatible with existing `flam3` flames while adding a few generalizations and extensions. The main change is the vChain system, which generalizes pre- and post- variations; an Xform consists of one or more functions called vLinks that are called sequentially and have their own affine transform and set of variations. This allows using any variation in a pre- or post- style, and this nesting may be done to any depth.

## Performance
While rendering should work on any modern Nvidia or AMD GPU, real-time performance requires high end RTX 3000/4000 Nvidia GPUs. The RTX 4000 series in particular will see a large performance boost due to the generous L2 cache size. 

## Denoising
`librefrakt` does not use traditional density estimation as in other flame renderers; instead, AI denosing is used for performance reasons. Two denoisers are provided:
* Nvidia's Optix AI denoiser - Available only on Nvidia GPUs, this denoiser is very fast and produces acceptable results. It is the default denoiser for real-time rendering. It also provides an upscaling mode that can greatly accelerate 4K rendering for almost no quality loss. The upscaler can also be used to provide a "super-sampling" mode when rendering at native resolution (similar in idea to Nvidia's DLAA technology). The main downside is that the weights are controlled by Nvidia and results may change between driver versions.
* Intel's Open Image Denoiser - Available on all platforms, this denoiser is slower than Optix but produces better results. It is the default denoiser for offline rendering. This denoiser uses custom weights trained on a large dataset of noisy flame renders at various quality levels, so it will provide more accurate results.

The difference between the two can be seen in especially low qualiy images. The Optix denoiser, being trained on raytraced images, will assume that black pixels part of the image and will blend actual samples with them, giving darker results in low density areas. The OIDN denoiser, being trained on flame renders, will better recognize areas of low density

## Glossary
* _Efficiency_ - The percent of iteration passes that resulted in a draw on the accumulator. 
* _Quality_ - Average number of samples output per pixel on the image. A 1280x720 image with quality 100 means that 1280 x 720 x 100 samples were drawn on the image. This is in contrast to other flame renderers, which define quality as the number of iteration passes per pixel regardless of whether a draw occurred.
* _vChain_ - A collection of functions in an Xform that compute the new position of an iterator.
* _vLink_ - An individual function in a vChain. It consists of an affine transform and then a summation of any number of weighted variations. It is essentially what an Xform is in traditional flame renderers.