/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "Compression.h"
#include <stdexcept>
#include <zlib.h>
#include <string.h>
#include <iomanip>
#include <sstream>

using namespace cudaq::qio;

std::string cudaq::qio::gzipCompress(const std::string &input) {
    int compressionlevel = Z_BEST_COMPRESSION;
    z_stream zs; // z_stream is zlib's control structure
    memset(&zs, 0, sizeof(zs));

    if (deflateInit(&zs, compressionlevel) != Z_OK)
        throw(std::runtime_error("deflateInit failed while compressing."));

    zs.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
    zs.avail_in = input.size(); // set the z_stream's input

    int ret;
    char outbuffer[10240];
    std::string outstring;

    // retrieve the compressed bytes blockwise
    do {
        zs.next_out = const_cast<Bytef*>(reinterpret_cast<Bytef*>(outbuffer));
        zs.avail_out = sizeof(outbuffer);

        ret = deflate(&zs, Z_FINISH);

        if (outstring.size() < zs.total_out) {
            // append the block to the output string
            outstring.append(outbuffer,
                             zs.total_out - outstring.size());
        }
    } while (ret == Z_OK);

    deflateEnd(&zs);

    if (ret != Z_STREAM_END) {          // an error occurred that was not EOF
        std::ostringstream oss;
        oss << "Exception during zlib compression: (" << ret << ") " << zs.msg;
        throw(std::runtime_error(oss.str()));
    }

    return outstring;
}


std::string cudaq::qio::gzipDecompress(const std::string &input) {
    z_stream zs;                        // z_stream is zlib's control structure
    memset(&zs, 0, sizeof(zs));

    if (inflateInit(&zs) != Z_OK)
        throw(std::runtime_error("inflateInit failed while decompressing."));

    // zs.next_in = (Bytef*)input.data();
    zs.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
    zs.avail_in = input.size();

    int ret;
    char outbuffer[10240];
    std::string outstring;

    // get the decompressed bytes blockwise using repeated calls to inflate
    do {
        zs.next_out = const_cast<Bytef*>(reinterpret_cast<Bytef*>(outbuffer));
        zs.avail_out = sizeof(outbuffer);

        ret = inflate(&zs, 0);

        if (outstring.size() < zs.total_out) {
            outstring.append(outbuffer,
                             zs.total_out - outstring.size());
        }

    } while (ret == Z_OK);

    inflateEnd(&zs);

    if (ret != Z_STREAM_END) {          // an error occurred that was not EOF
        std::ostringstream oss;
        oss << "Exception during zlib decompression: (" << ret << ") "
            << zs.msg;
        throw(std::runtime_error(oss.str()));
    }

    return outstring;
}
