import os
from setuptools import setup, find_packages
from cuda_autodetect import infer_best_package, bdist_wheel

package_ver = '1.0.0'
release_ver = '1.0.0'
package_name = 'cudaq'

# get project long description
with open("README.rst") as f:
    long_description = f.read()

if os.environ.get('CUDAQ_META_WHEEL_BUILD', '0') == '1':
    install_requires = []
    data_files = [('', ['cuda_autodetect.py',])]
    cmdclass = {}
else:
    install_requires = [f"{infer_best_package(package_name)}=={release_ver}"]
    data_files = []
    cmdclass = {'bdist_wheel': bdist_wheel} if bdist_wheel is not None else {}

setup(
    name=package_name,
    version=package_ver,
    description="A CUDA version autodetecting package for cudaq",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/NVIDIA/cuda-quantum",
    author="NVIDIA Corporation",
    author_email="",
    license="NVIDIA Proprietary Software",
    license_files=('LICENSE',),
    keywords=["cuda", "cudaq"],
    include_package_data=True,
    zip_safe=False,
    data_files=data_files,
    setup_requires=[
        "setuptools",
        "wheel",
    ],
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU :: NVIDIA CUDA :: 11",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
    ],
    cmdclass=cmdclass,
)