/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "compression.h"
#include <zlib.h>
#include <stdexcept>

namespace qio::compression {

std::string gzipCompress(const std::string &input) {
  z_stream zs{};
  if (deflateInit2(&zs, Z_BEST_COMPRESSION, Z_DEFLATED,
                   15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
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

static const char table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64Encode(const std::string &in) {
  std::string out;
  int val = 0, valb = -6;
  for (uint8_t c : in) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(table[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6) out.push_back(table[((val << 8) >> (valb + 8)) & 0x3F]);
  while (out.size() % 4) out.push_back('=');
  return out;
}

}
