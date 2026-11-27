/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file unified_dispatch_kernel_image.cpp
/// @brief Embeds the unified dispatch kernel's LTO-IR fatbin, which the build
/// produces from unified_dispatch_kernel.cu, between two hidden symbols.

#ifndef CUDAQ_REALTIME_UNIFIED_KERNEL_FATBIN
#error "CUDAQ_REALTIME_UNIFIED_KERNEL_FATBIN must name the kernel's fatbin"
#endif

// nvJitLink reads the fatbin's 8-byte header fields in place.
asm(".section .rodata.cudaq_unified_dispatch_kernel_image,\"a\"\n"
    ".balign 16\n"
    ".globl cudaq_unified_dispatch_kernel_image_begin\n"
    ".hidden cudaq_unified_dispatch_kernel_image_begin\n"
    "cudaq_unified_dispatch_kernel_image_begin:\n"
    ".incbin \"" CUDAQ_REALTIME_UNIFIED_KERNEL_FATBIN "\"\n"
    ".globl cudaq_unified_dispatch_kernel_image_end\n"
    ".hidden cudaq_unified_dispatch_kernel_image_end\n"
    "cudaq_unified_dispatch_kernel_image_end:\n"
    ".previous\n");
