set -ex
cmake -S .. -G "Ninja" \
         -DCMAKE_INSTALL_PREFIX=./ \
         -DCMAKE_INSTALL_RPATH=./lib \
         -DLLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm \
         -DCUDAQ_ENABLE_PYTHON=TRUE \
         -DGIT_SUBMODULE=FALSE \
         -DCMAKE_CXX_STANDARD=17 
ninja install 
export PYTHONPATH=$PYTHONPATH:$PWD
python -m pytest ../python/tests/unittests
