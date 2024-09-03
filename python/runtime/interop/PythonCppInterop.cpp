#include "PythonCppInterop.h"
#include "cudaq.h"

namespace cudaq {

std::string getKernelName(std::string &input) {
  size_t pos = 0;
  std::string result = "";
  while (true) {
    // Find the next occurrence of "func.func @"
    size_t start = input.find("func.func @", pos) + 11;

    if (start == std::string::npos)
      break;

    // Find the position of the first "(" after "func.func @"
    size_t end = input.find("(", start);

    if (end == std::string::npos)
      break;

    // Extract the substring
    result = input.substr(start, end - start);

    // Check if the substring doesn't contain ".thunk"
    if (result.find(".thunk") == std::string::npos)
      break;

    // Move the position to continue searching
    pos = end;
  }
  return result;
}

std::string extractSubstring(const std::string &input,
                             const std::string &startStr,
                             const std::string &endStr) {
  size_t startPos = input.find(startStr);
  if (startPos == std::string::npos) {
    return ""; // Start string not found
  }

  startPos += startStr.length(); // Move to the end of the start string
  size_t endPos = input.find(endStr, startPos);
  if (endPos == std::string::npos) {
    return ""; // End string not found
  }

  return input.substr(startPos, endPos - startPos);
}

std::tuple<std::string, std::string>
getMLIRCodeAndName(const std::string &name) {
  auto cppMLIRCode = cudaq::get_quake("entryPoint");
  auto kernelName = cudaq::getKernelName(cppMLIRCode);
  cppMLIRCode = "module {\nfunc.func @" + kernelName +
                cudaq::extractSubstring(cppMLIRCode, "func.func @" + kernelName,
                                        "func.func") +
                "\n}";
  return std::make_tuple(kernelName, cppMLIRCode);
}
} // namespace cudaq