set -ex
cmake -S .. -G "Ninja" \
         -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/clang++ \
         -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/clang \
         -DCMAKE_INSTALL_PREFIX=./ \
         -DCMAKE_INSTALL_RPATH=./lib \
         -DLLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm \
         -DCUDAQ_ENABLE_PYTHON=TRUE \
         -DCMAKE_CXX_STANDARD=20 
ninja install 
export PYTHONPATH=$PYTHONPATH:$PWD
python -m pytest ../python/tests/unittests
