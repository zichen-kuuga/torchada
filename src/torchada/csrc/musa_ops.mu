// torchada MUSA operator overrides
//
// This file contains MUSA kernel implementations that override torch_musa's
// default ATen operator implementations.

#include "ops.h"
#include <ATen/musa/MUSAContext.h>

namespace torchada {

// ============================================================================
// MUSA kernel: neg (negation)
// Overrides aten::neg for PrivateUse1 (MUSA) tensors
// ============================================================================

template <typename scalar_t>
__global__ void neg_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = -input[idx];
    }
}

at::Tensor neg_musa_impl(const at::Tensor& self) {
    log_op_call("neg");

    // Ensure contiguous tensor
    auto self_contig = self.contiguous();

    // Allocate output tensor
    auto output = at::empty_like(self_contig);

    if (self_contig.numel() == 0) {
        return output;
    }

    // Get MUSA stream
    musaStream_t stream = at::musa::getCurrentMUSAStream();

    // Launch kernel
    const int64_t numel = self_contig.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        self_contig.scalar_type(), "neg_musa", [&] {
            neg_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                output.data_ptr<scalar_t>(),
                self_contig.data_ptr<scalar_t>(),
                numel);
        });

    // Check for launch errors
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        TORCH_CHECK(false, "MUSA kernel launch failed: ", musaGetErrorString(err));
    }

    return output;
}

}  // namespace torchada

// ============================================================================
// Register operator overrides for PrivateUse1 (MUSA)
// ============================================================================

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("neg", torchada::neg_musa_impl);
}
