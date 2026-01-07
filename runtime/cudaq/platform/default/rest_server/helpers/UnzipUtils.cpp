/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/UnzipUtils.h"
#include "common/FmtCore.h"
#include "unzip.h"
#include <fstream>
namespace cudaq {
namespace utils {

static std::string getUnzipErrorString(int errorCode) {
  if (errorCode == UNZ_END_OF_LIST_OF_FILE)
    return "END OF LIST OF FILE";
  if (errorCode == UNZ_PARAMERROR)
    return "PARAM ERROR";
  if (errorCode == UNZ_BADZIPFILE)
    return "BAD ZIP FILE";
  if (errorCode == UNZ_INTERNALERROR)
    return "INTERNAL ERROR";
  if (errorCode == UNZ_CRCERROR)
    return "CRC ERROR";
  return "UNKNOWN ERROR";
}

#define HANDLE_MINIZIP_ERROR(x)                                                \
  do {                                                                         \
    const auto err = x;                                                        \
    if (err != UNZ_OK) {                                                       \
      throw std::runtime_error(fmt::format("[minizip] %{} in {} (line {})",    \
                                           getUnzipErrorString(err),           \
                                           __FUNCTION__, __LINE__));           \
    }                                                                          \
  } while (false)

void unzip(const std::filesystem::path &zipFile,
           const std::filesystem::path &outputDir) {
  const char *zipFileName = zipFile.c_str();
  ::unzFile unzipFileDesc = ::unzOpen(zipFileName);
  if (unzipFileDesc == nullptr) {
    throw std::runtime_error("Failed to open zip file " + zipFile.string());
  }

  ::unz_global_info64 unzipGlobalInfo;
  HANDLE_MINIZIP_ERROR(::unzGetGlobalInfo64(unzipFileDesc, &unzipGlobalInfo));
  const auto numFiles = unzipGlobalInfo.number_entry;
  char readBuffer[8192];
  if (!std::filesystem::exists(outputDir))
    std::filesystem::create_directories(outputDir);

  for (std::size_t fileId = 0; fileId < numFiles; ++fileId) {
    ::unz_file_info fileInfo;
    char fileNameInZip[4096]; // Linux PATH_MAX
    HANDLE_MINIZIP_ERROR(::unzGetCurrentFileInfo(
        unzipFileDesc, &fileInfo, fileNameInZip, sizeof(fileNameInZip),
        /*extraField=*/NULL, /*extraFieldBufferSize=*/0, /*szComment=*/NULL,
        /*commentBufferSize=*/0));
    HANDLE_MINIZIP_ERROR(::unzOpenCurrentFile(unzipFileDesc));
    const auto unzipFilePath =
        outputDir / std::filesystem::path(fileNameInZip).filename();
    std::ofstream unzipFile;
    try {
      unzipFile.open(unzipFilePath.string(),
                     std::ios_base::binary | std::ios::out);
    } catch (std::exception &e) {
      throw std::runtime_error(
          fmt::format("Failed to create the unzip file '{}'. Error: {}",
                      unzipFilePath.string(), e.what()));
    }

    for (;;) {
      // Break inside
      const int bytesReadOrErrorCode =
          ::unzReadCurrentFile(unzipFileDesc, readBuffer, sizeof(readBuffer));
      // Returns 0 if the end of file was reached
      if (bytesReadOrErrorCode == 0)
        break;

      if (bytesReadOrErrorCode > 0)
        // Read some bytes
        unzipFile.write(readBuffer, bytesReadOrErrorCode);

      if (bytesReadOrErrorCode < 0)
        HANDLE_MINIZIP_ERROR(bytesReadOrErrorCode);
    }
    unzipFile.close();
    HANDLE_MINIZIP_ERROR(::unzCloseCurrentFile(unzipFileDesc));
    if ((fileId + 1) != numFiles)
      HANDLE_MINIZIP_ERROR(::unzGoToNextFile(unzipFileDesc));
  }
  HANDLE_MINIZIP_ERROR(::unzClose(unzipFileDesc));
}
} // namespace utils
} // namespace cudaq
