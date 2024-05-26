#pragma once

#include <regex>
#include <string>
#include <nlohmann/json.hpp>
#include "JsonConvert.h"

using json = nlohmann::json;

namespace cudaq
{
class RequestValidator {
private:
    std::regex unsafePatterns;
    bool validateJsonValue(const json &value);
    bool validateJsonValue(const std::string &key);

public:
    RequestValidator();
    bool validateString(const std::string &sourceCode);
    bool validateNamespace(const json &namespace_dict);
    bool validateRequest(const SerializedCodeExecutionContext &serializedCodeExecutionContext,
                             std::string &outValidationMessage);
};
} // namespace cudaq

