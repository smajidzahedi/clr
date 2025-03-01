#pragma once

#include <hip/hip_runtime.h>
#include <thread>
#include <mutex>
#include <iostream>
#include "thread/thread.hpp"
#include "hip_internal.hpp"
#include "hip_mrfs.hpp"

namespace hip {

extern std::thread* hipMemCpyThread;
extern std::mutex threadMutex;

// Forward declarations of the original sync functions we need to call
extern hipError_t ihipDeviceSynchronize();
extern hipError_t ihipStreamSynchronize(hipStream_t stream);
extern hipError_t ihipMemcpy_validate(void* dst, const void* src, size_t sizeBytes,
                                     hipMemcpyKind kind);
// extern hipError_t ihipMemcpy(void* dst, const void* src, size_t sizeBytes,
//                            hipMemcpyKind kind, hip::Stream& stream, bool isAsync);

// Our custom async implementation
hipError_t hipMemcpyAsync_mrfs(void* dst, const void* src, size_t sizeBytes, 
                             hipMemcpyKind kind, hip::Stream& stream);

// Our intercepted synchronization functions
hipError_t interceptedHipDeviceSynchronize();
hipError_t interceptedHipStreamSynchronize(hipStream_t stream);

} // namespace hip