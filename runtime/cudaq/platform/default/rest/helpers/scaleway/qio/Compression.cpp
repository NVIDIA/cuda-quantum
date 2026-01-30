/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2026 Scaleway                                                     *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "Compression.h"
#include <stdexcept>
#include <zlib.h>

using namespace cudaq::qio;

std::string cudaq::qio::gzipCompress(const std::string &input) {
    if (input.empty())
        return {};

    z_stream zs{};
    if (deflateInit2(
            &zs,
            Z_BEST_COMPRESSION,
            Z_DEFLATED,
            15 + 16,        // 15 = window bits, +16 = gzip
            8,
            Z_DEFAULT_STRATEGY) != Z_OK) {
        throw std::runtime_error("deflateInit2 failed");
    }

    zs.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(input.data()));
    zs.avail_in = static_cast<uInt>(input.size());

    std::string output;
    char buffer[32768];

    int ret;
    do {
        zs.next_out = reinterpret_cast<Bytef *>(buffer);
        zs.avail_out = sizeof(buffer);

        ret = deflate(&zs, Z_FINISH);

        if (output.size() < zs.total_out)
            output.append(buffer, zs.total_out - output.size());
    } while (ret == Z_OK);

    deflateEnd(&zs);

    if (ret != Z_STREAM_END)
        throw std::runtime_error("deflate failed");

    return output;
}


std::string cudaq::qio::gzipDecompress(const std::string &input) {
    if (input.empty())
        return {};

    z_stream zs{};
    if (inflateInit2(&zs, 15 + 16) != Z_OK) {
        throw std::runtime_error("inflateInit2 failed");
    }

    zs.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(input.data()));
    zs.avail_in = static_cast<uInt>(input.size());

    std::string output;
    char buffer[32768];

    int ret;
    do {
        zs.next_out = reinterpret_cast<Bytef *>(buffer);
        zs.avail_out = sizeof(buffer);

        ret = inflate(&zs, 0);

        if (output.size() < zs.total_out)
            output.append(buffer, zs.total_out - output.size());
    } while (ret == Z_OK);

    inflateEnd(&zs);

    if (ret != Z_STREAM_END)
        throw std::runtime_error("inflate failed");

    return output;
}
