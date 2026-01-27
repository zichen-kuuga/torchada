// torchada C++ operator overrides
//
// This header provides the infrastructure for registering custom ATen operator
// implementations that override the default PrivateUse1 (MUSA) implementations.
//
// Usage:
//   1. Include this header in your .cpp file
//   2. Use TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) to register overrides
//   3. The extension will be built and loaded automatically by torchada
//
// Example:
//   #include "ops.h"
//
//   at::Tensor my_custom_add(const at::Tensor& self, const at::Tensor& other,
//                            const at::Scalar& alpha) {
//       // Custom implementation
//       auto result = at::empty_like(self);
//       result.copy_(self);
//       result.add_(other, alpha);
//       return result;
//   }
//
//   TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
//       m.impl("add.Tensor", my_custom_add);
//   }

#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>

namespace torchada {

// Version information
constexpr const char* VERSION = "0.1.0";

// Check if operator override is enabled via environment variable
inline bool is_override_enabled(const char* op_name) {
    // Check TORCHADA_DISABLE_OP_OVERRIDE_<OP_NAME> environment variable
    std::string env_var = "TORCHADA_DISABLE_OP_OVERRIDE_";
    env_var += op_name;
    const char* val = std::getenv(env_var.c_str());
    if (val != nullptr && std::string(val) == "1") {
        return false;
    }
    return true;
}

// Logging helper for debugging
inline void log_op_call(const char* op_name) {
    const char* debug = std::getenv("TORCHADA_DEBUG_CPP_OPS");
    if (debug != nullptr && std::string(debug) == "1") {
        std::cout << "[torchada] " << op_name << " called" << std::endl;
    }
}

}  // namespace torchada

