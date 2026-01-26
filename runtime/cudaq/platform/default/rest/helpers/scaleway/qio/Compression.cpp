/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2026 Scaleway                                                     *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "Compression.h"

std::string gzipCompress(const std::string &input) {
  z_stream zs{};
  if (deflateInit2(&zs, Z_BEST_COMPRESSION, Z_DEFLATED, 15 + 16, 8,
                   Z_DEFAULT_STRATEGY) != Z_OK)
    throw std::runtime_error("deflateInit failed");

  zs.next_in = (Bytef *)input.data();
  zs.avail_in = input.size();

  char outbuffer[32768];
  std::string out;

  int ret;
  do {
    zs.next_out = reinterpret_cast<Bytef *>(outbuffer);
    zs.avail_out = sizeof(outbuffer);

    ret = deflate(&zs, Z_FINISH);
    out.append(outbuffer, sizeof(outbuffer) - zs.avail_out);
  } while (ret == Z_OK);

  deflateEnd(&zs);
  return out;
}

std::string gzipDecompress(const std::string &input) {
  z_stream zs{};
  zs.zalloc = Z_NULL;
  zs.zfree = Z_NULL;
  zs.opaque = Z_NULL;

  zs.next_in = (Bytef *)input.data();
  zs.avail_in = input.size();

  if (inflateInit2(&zs, 15 + 32) != Z_OK) {
    throw std::runtime_error("qio: gzipDecompress inflateInit failed");
  }

  char outbuffer[32768];
  std::string out;
  int ret;

  do {
    zs.next_out = reinterpret_cast<Bytef *>(outbuffer);
    zs.avail_out = sizeof(outbuffer);

    ret = inflate(&zs, Z_NO_FLUSH);

    if (out.size() < zs.total_out) {
      out.append(outbuffer, zs.total_out - out.size());
    }

  } while (ret == Z_OK);

  inflateEnd(&zs);

  if (ret != Z_STREAM_END) {
    throw std::runtime_error(
        "qio: gzipDecompress failed (error code: " + std::to_string(ret) + ")");
  }

  return out;
}

// std::string base64Encode(const std::string &input) {
//   return base64::to_base64(input);
// }

// std::string base64Decode(const std::string &input) {
//   return base64::from_base64(input);
// }
