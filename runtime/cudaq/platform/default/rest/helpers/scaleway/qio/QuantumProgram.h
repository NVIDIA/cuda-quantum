#pragma once
#include <string>
#include <nlohmann/json.hpp>

namespace qio {

class QuantumProgram {
public:
  enum class SerializationFormat {
    UNKOWN_SERIALIZATION_FORMAT = 0
    QASM_V1 = 1
    QASM_V2 = 2
    QASM_V3 = 3
    QIR_V1 = 4
  };

 enum class CompressionFormat {
    UNKNOWN_COMPRESSION_FORMAT = 0
    NONE = 1
    ZLIB_BASE64_V1 = 2
  };

  QioQuantumProgram(const cudaq::Kernel &kernel,
                    SerializationFormat serializationFormat,
                    CompressionFormat compressionFormat);

  nlohmann::json toJson() const;

private:
  std::string m_serialization;
  SerializationFormat m_serializationFormat;
  CompressionFormat m_compressionFormat;
};

}
