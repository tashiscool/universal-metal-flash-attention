#pragma once

#include <ATen/Tensor.h>

namespace metal_sdpa {
namespace mps_utils {

bool is_mps_tensor(const at::Tensor& tensor);

// Returns opaque pointer to id<MTLBuffer> (or nullptr on failure)
void* get_mtl_buffer_handle(const at::Tensor& tensor);

// Returns storage offset in bytes for the tensor within its MTLBuffer
size_t get_storage_offset_bytes(const at::Tensor& tensor);

// Synchronize MPS command queue — flushes all pending GPU ops and waits
void synchronize_mps();

} // namespace mps_utils
} // namespace metal_sdpa

