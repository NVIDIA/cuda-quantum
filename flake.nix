{
  description = "CUDA Quantum development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-cmake.url = "github:NixOS/nixpkgs?rev=c9eb8d14da4455de9f05ce9429324e5b1b2bc638";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, nixpkgs-cmake, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        pkgs-cmake = import nixpkgs-cmake {
          inherit system;
        };
        cmake_3_26 = pkgs-cmake.cmake;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # CUDA Toolkit (required for GPU acceleration)
            cudaPackages.cuda_cudart
            cudaPackages.cuda_nvrtc
            cudaPackages.cuda_nvcc
            cudaPackages.libcusolver
            cudaPackages.libcublas
            cudaPackages.libnvjitlink

            # CUDA-aware MPI
            openmpi

            # Core build tools
            cmake_3_26
            ninja
            gcc11
            gfortran11

            # Python dependencies
            python3
            python3Packages.pip
            python3Packages.pybind11
            python3Packages.numpy
            python3Packages.pytest
            python3Packages.setuptools
            python3Packages.wheel
            python3Packages.fastapi
            python3Packages.uvicorn
            python3Packages.llvmlite

            # Build tools
            git
            gnupg
            wget
            gnumake
            automake
            libtool
            unzip
            gtest
            autoconf

            # Dev tools
            pre-commit

            # Prerequisites
            pkgsStatic.openblas
            ncurses
            libz.dev
          ];

          shellHook = ''
            # CUDA setup
            export CUDA_PATH="${pkgs.cudaPackages.cuda_cudart}"
            export CUDA_HOME="${pkgs.cudaPackages.cuda_nvcc}"

            # MPI setup
            export MPI_PATH="${pkgs.openmpi}"
            export OMPI_MCA_opal_warn_on_missing_libcuda=0

            # Prerequisites - nix-managed
            export BLAS_INSTALL_PREFIX="${pkgs.pkgsStatic.openblas}"

            # Prerequisites - unmanaged
            export ZLIB_INSTALL_PREFIX=/usr/local/zlib
            export LLVM_INSTALL_PREFIX=/usr/local/llvm
            export OPENSSL_INSTALL_PREFIX=/usr/local/openssl
            export CURL_INSTALL_PREFIX=/usr/local/curl
            export AWS_INSTALL_PREFIX=/usr/local/aws
            export CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
            export CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum
            export CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor

            # Python setup
            export PYTHONPATH="${pkgs.python3Packages.pybind11}/${pkgs.python3.sitePackages}:$PYTHONPATH"

            # Compiler setup
            export CC="${pkgs.gcc11}/bin/gcc"
            export CXX="${pkgs.gcc11}/bin/g++"
            export FC="${pkgs.gfortran11}/bin/gfortran"

            # LLVM setup
            export PATH="$PATH:$LLVM_INSTALL_PREFIX/bin/"

            echo "CUDA Quantum development environment loaded"
            echo "CUDA available at: $CUDA_PATH"
            echo "MPI available at: $MPI_PATH"
          '';

          # Required for CUDA development
          LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [
            cudaPackages.cuda_cudart
            cudaPackages.cuda_nvrtc
            cudaPackages.libcusolver
            cudaPackages.libcublas
            cudaPackages.libnvjitlink
            openmpi
          ];
        };
      });
}
