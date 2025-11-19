# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.13)

# CMake invokes the toolchain file twice during the first build, but only once
# during subsequent rebuilds. This was causing the various flags to be added
# twice on the first build, and on a rebuild ninja would see only one set of the
# flags and rebuild the world.
# https://github.com/android-ndk/ndk/issues/323
if(ARM_TOOLCHAIN_INCLUDED)
  return()
endif(ARM_TOOLCHAIN_INCLUDED)
set(ARM_TOOLCHAIN_INCLUDED true)

set(CMAKE_SYSTEM_PROCESSOR arm)

set(ARM_HOST_TAG linux)

set(ARM_TOOL_PATH "$ENV{HOME}/arm" CACHE PATH "RISC-V tool path")

set(ARM_TOOLCHAIN_ROOT "${ARM_TOOL_PATH}/toolchain/clang/${ARM_HOST_TAG}/ARM" CACHE PATH "RISC-V compiler path")
set(ARM_TOOLCHAIN_PREFIX "arm64-unknown-linux-gnu-" CACHE STRING "RISC-V toolchain prefix")
set(CMAKE_FIND_ROOT_PATH ${ARM_TOOLCHAIN_ROOT})
list(APPEND CMAKE_PREFIX_PATH "${ARM_TOOLCHAIN_ROOT}")

set(CMAKE_C_COMPILER "${ARM_TOOLCHAIN_ROOT}/bin/clang")
set(CMAKE_CXX_COMPILER "${ARM_TOOLCHAIN_ROOT}/bin/clang++")
set(CMAKE_AR "${ARM_TOOLCHAIN_ROOT}/bin/llvm-ar")
set(CMAKE_RANLIB "${ARM_TOOLCHAIN_ROOT}/bin/llvm-ranlib")
set(CMAKE_STRIP "${ARM_TOOLCHAIN_ROOT}/bin/llvm-strip")

set(ARM_COMPILER_FLAGS)
set(ARM_COMPILER_FLAGS_CXX)
set(ARM_COMPILER_FLAGS_DEBUG)
set(ARM_COMPILER_FLAGS_RELEASE)
set(ARM_LINKER_FLAGS)
set(ARM_LINKER_FLAGS_EXE "" CACHE STRING "Linker flags for ARM executables")

if (ARM_CPU MATCHES "generic")
  set(CMAKE_SYSTEM_NAME Generic)
  set(CMAKE_SYSTEM_LIBRARY_PATH "${ARM_TOOLCHAIN_ROOT}/${ARM_TOOLCHAIN_PREFIX}/lib/")
  set(CMAKE_CROSSCOMPILING ON CACHE BOOL "")
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_C_EXTENSIONS OFF)     # Force the usage of _ISOC11_SOURCE
  set(IREE_BUILD_BINDINGS_TFLITE OFF CACHE BOOL "" FORCE)
  set(IREE_BUILD_BINDINGS_TFLITE_JAVA OFF CACHE BOOL "" FORCE)
  set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
  set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)
  set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF CACHE BOOL "" FORCE)
  set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON CACHE BOOL "" FORCE)
  set(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE ON CACHE BOOL "" FORCE)
  set(IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS OFF CACHE BOOL "" FORCE)
  set(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF ON CACHE BOOL "" FORCE)
  set(IREE_ENABLE_THREADING OFF CACHE BOOL "" FORCE)
elseif(ARM_CPU MATCHES "linux")
  set(CMAKE_SYSTEM_NAME Linux)
endif()

if(ARM_CPU MATCHES "arm_64")
  set(CMAKE_SYSTEM_PROCESSOR arm64)
elseif(ARM_CPU MATCHES "arm_32")
  set(CMAKE_SYSTEM_PROCESSOR arm32)
endif()

if(ARM_CPU STREQUAL "linux-arm_64")
  set(CMAKE_SYSTEM_LIBRARY_PATH "${ARM_TOOLCHAIN_ROOT}/sysroot/lib64/lp64")
  set(CMAKE_SYSROOT "${ARM_TOOLCHAIN_ROOT}/sysroot")
  # Specify ISP spec for march=rv64gc. This is to resolve the mismatch between
  # llvm and binutil ISA version.
  set(ARM_COMPILER_FLAGS "${ARM_COMPILER_FLAGS} \
      -march=armv8-a+simd -mtune=cortex-a72 -mabi=lp64")
  set(ARM_LINKER_FLAGS "${ARM_LINKER_FLAGS} -lstdc++ -lpthread -lm -ldl")
  set(ARM64_TEST_DEFAULT_LLVM_FLAGS
    "--iree-llvmcpu-target-triple=aarch64"
    "--iree-llvmcpu-target-abi=lp64"
    "--iree-llvmcpu-target-cpu=generic"
    "--iree-llvmcpu-target-cpu-features=+neon,+fp-armv8"
    CACHE INTERNAL "Default llvm codegen flags for testing purposes")
elseif(ARM_CPU STREQUAL "generic-arm_64")
  # Specify ISP spec for march=rv64gc. This is to resolve the mismatch between
  # llvm and binutil ISA version.
  set(ARM_COMPILER_FLAGS "${ARM_COMPILER_FLAGS} \
      -march=rv64i2p0ma2p0f2p0d2p0c2p0 -mabi=lp64 -DIREE_PLATFORM_GENERIC=1 -DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1 \
      -DIREE_FILE_IO_ENABLE=0 -DIREE_TIME_NOW_FN=\"\{ return 0; \}\" -DIREE_DEVICE_SIZE_T=uint64_t -DPRIdsz=PRIu64")
  set(ARM_LINKER_FLAGS "${ARM_LINKER_FLAGS} -lm")
elseif(ARM_CPU STREQUAL "linux-arm_32")
  list(APPEND CMAKE_SYSTEM_LIBRARY_PATH
    "${ARM_TOOLCHAIN_ROOT}/sysroot/usr/lib32"
    "${ARM_TOOLCHAIN_ROOT}/sysroot/usr/lib32/ilp32d"
  )
  set(CMAKE_SYSROOT "${ARM_TOOLCHAIN_ROOT}/sysroot")
  # Specify ISP spec for march=rv32gc. This is to resolve the mismatch between
  # llvm and binutil ISA version.
  set(ARM_COMPILER_FLAGS "${ARM_COMPILER_FLAGS} \
      -march=rv32i2p1ma2p1f2p2d2p2c2p0 -mabi=ilp32d \
      -Wno-atomic-alignment")
  set(ARM_LINKER_FLAGS "${ARM_LINKER_FLAGS} -lstdc++ -lpthread -lm -ldl -latomic")
  set(ARM32_TEST_DEFAULT_LLVM_FLAGS
    "--iree-llvmcpu-target-triple=armv7-linux-gnueabihf"
    "--iree-llvmcpu-target-abi=aapcs"
    "--iree-llvmcpu-target-cpu-features=+neon,+vfp4"
    CACHE INTERNAL "Default llvm codegen flags for testing purposes")
elseif(ARM_CPU STREQUAL "generic-arm_32")
  set(CMAKE_SYSROOT "${ARM_TOOLCHAIN_ROOT}/${ARM_TOOLCHAIN_PREFIX}")
  # Specify ISP spec for march=rv32imf. This is to resolve the mismatch between
  # llvm and binutil ISA version.
  set(ARM_COMPILER_FLAGS "${ARM_COMPILER_FLAGS} \
      -march=rv32i2p1mf2p2 -mabi=ilp32 -DIREE_PLATFORM_GENERIC=1 -DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1 \
      -DIREE_FILE_IO_ENABLE=0 -DIREE_TIME_NOW_FN=\"\{ return 0; \}\" -DIREE_DEVICE_SIZE_T=uint32_t -DPRIdsz=PRIu32")
  set(ARM_LINKER_FLAGS "${ARM_LINKER_FLAGS} -lm")
endif()

set(CMAKE_C_FLAGS             "${ARM_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS           "${ARM_COMPILER_FLAGS} ${ARM_COMPILER_FLAGS_CXX} ${CMAKE_CXX_FLAGS}")
set(CMAKE_ASM_FLAGS           "${ARM_COMPILER_FLAGS} ${CMAKE_ASM_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG       "${ARM_COMPILER_FLAGS_DEBUG} ${CMAKE_C_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_DEBUG     "${ARM_COMPILER_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_ASM_FLAGS_DEBUG     "${ARM_COMPILER_FLAGS_DEBUG} ${CMAKE_ASM_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_RELEASE     "${ARM_COMPILER_FLAGS_RELEASE} ${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELEASE   "${ARM_COMPILER_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_ASM_FLAGS_RELEASE   "${ARM_COMPILER_FLAGS_RELEASE} ${CMAKE_ASM_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS "${ARM_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${ARM_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS    "${ARM_LINKER_FLAGS} ${ARM_LINKER_FLAGS_EXE} ${CMAKE_EXE_LINKER_FLAGS}")
