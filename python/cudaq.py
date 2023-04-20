# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from _pycudaq import *
from typing import List
import sys, subprocess, os

def nvqpp(): 
    command = os.path.join(os.path.dirname(__file__), "bin/nvq++")
    subprocess.call([command] + sys.argv[1:])

def cudaq_quake():
    installDir = os.path.dirname(__file__)
    # Get lib/clang resource dir, it has a version name sub-folder
    # get that version, its the only subfolder in the directory
    clangVer = os.listdir(os.path.join(installDir, "lib/clang"))[0]
    resourceDir = os.path.join(installDir, "lib/clang/{}".format(clangVer))
    command = os.path.join(installDir, "bin/cudaq-quake")
    subprocess.call([command] + sys.argv[1:] +['-resource-dir', resourceDir])

def cudaq_opt(): 
    command = os.path.join(os.path.dirname(__file__), "bin/cudaq-opt")
    subprocess.call([command] + sys.argv[1:])

def quake_translate(): 
    command = os.path.join(os.path.dirname(__file__), "bin/quake-translate")
    subprocess.call([command] + sys.argv[1:])

def quake_synth(): 
    command = os.path.join(os.path.dirname(__file__), "bin/quake-synth")
    subprocess.call([command] + sys.argv[1:])

initKwargs = {'qpu': 'qpp', 'platform':'default'}

if '-qpu' in sys.argv:
    initKwargs['qpu'] = sys.argv[sys.argv.index('-qpu')+1]

if '--qpu' in sys.argv:
    initKwargs['qpu'] = sys.argv[sys.argv.index('--qpu')+1]

if '-platform' in sys.argv:
    initKwargs['platform'] = sys.argv[sys.argv.index('-platform')+1]

if '--platform' in sys.argv:
    initKwargs['platform'] = sys.argv[sys.argv.index('--platform')+1]

initialize_cudaq(**initKwargs)
