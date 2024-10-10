# refraktor

# Building
To build, run these from the repo's root
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Binaries must be put in the repo's root to run, but running with Visual Studio will automatically set the correct paths.

`get_runtime_deps.py` in build_scripts will fetch required binaries (DLLs, fonts, etc).
