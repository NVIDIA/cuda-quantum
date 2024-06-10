#include "RequestValidator.h"
#include <iostream>

namespace cudaq {
RequestValidator::RequestValidator() {
  unsafePatterns = std::regex(
      R"(\b(__import__|eval|exec|compile|open|imput|os\.|sys\.|subprocess\.|shutil\.|Popen|system|getattr|setattr|delattr|globals|locals|vars|exit|quit|file|open|read|write|close|unlink|remove|rmdir|mkdir|chmod|chown|chdir|pathlib|tempfile|signal|threading|multiprocessing|socket|ctypes|ffi|pickle|marshal|builtins|xml|json|yaml|base64|webbrowser|urllib|requests|http|ftplib|poplib|smtplib|telnetlib|imaplib|nntplib|requests|cgi|random|secrets|hashlib|inspect|ast|imp|resource|crypt|pwd|grp)\b)");
}

bool RequestValidator::validateString(const std::string &sourceCode) {
  return std::regex_search(sourceCode, unsafePatterns);
}

bool RequestValidator::validateJsonValue(const std::string &key) {
  return !validateString(key);
}

bool RequestValidator::validateJsonValue(const json &value) {
  if (value.is_string())
    return !validateString(value.get<std::string>());

  if (value.is_object()) {
    return validateNamespace(value);
  }
  if (value.is_array()) {
    for (const auto &item : value) {
      if (!validateJsonValue(item)) {
        return false;
      }
    }
  }
  return true;
}

bool RequestValidator::validateNamespace(const json &namespaceDict) {
  if (!namespaceDict.is_object()) {
    return false;
  }

  for (auto it = namespaceDict.begin(); it != namespaceDict.end(); it++) {
    if (!validateJsonValue(it.key()) || !validateJsonValue(it.value())) {
      return false;
    }
  }

  return true;
}

bool RequestValidator::validateRequest(
    const SerializedCodeExecutionContext &serializedCodeExecutionContext,
    std::string &outValidationMessage) {
  try {
    std::string sourceCode = serializedCodeExecutionContext.code;
    json localsNamespace = serializedCodeExecutionContext.locals;
    json globalsNamespace = serializedCodeExecutionContext.globals;

    if (!validateString(sourceCode)) {
      outValidationMessage = "Invalid source code.";
      return false;
    }
    if (!validateNamespace(localsNamespace) ||
        !validateNamespace(globalsNamespace)) {
      outValidationMessage = "Invalid namespace.";
      return false;
    }
  } catch (const std::exception &e) {
    outValidationMessage = "Invalid request field format.";
    return false;
  }

  return true;
}
} // namespace cudaq