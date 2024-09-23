cmake_minimum_required(VERSION 3.22)
set(CMAKE_C_COMPILER "/usr/bin/clang-18")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-18")
SET(CMAKE_TOOLCHAIN_FILE "/home/fanghz/lib/vcpkg/scripts/buildsystems/vcpkg.cmake")
project(Agent)
set(CMAKE_CXX_STANDARD 20)
include_directories(SYSTEM "/home/fanghz/lib/vcpkg/installed/x64-linux/include")
SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_BUILD_TYPE "Release")
find_package(Threads REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
set(absl_LIBRARIES
absl::algorithm
absl::base
absl::debugging
absl::flat_hash_map
absl::memory
absl::meta
absl::numeric
absl::str_format
absl::strings
absl::synchronization
absl::time
absl::flags
absl::strings
absl::log
absl::flags_internal
absl::log_initialize
absl::config
absl::base
absl::flags_parse
absl::utility)

# 设置检测程序的源文件
set(CPU_CHECK_SOURCE "${CMAKE_SOURCE_DIR}/avx_checker.cc")

# 尝试编译并运行检测程序
try_run(
    RUN_RESULT        # 运行结果变量
    CPU_CHECK_OUTPUT  # 输出结果变量
    ${CMAKE_BINARY_DIR}  # 运行目录
    ${CPU_CHECK_SOURCE}  # 源文件路径
)

file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/*.cc)
list(REMOVE_ITEM SOURCES ${CMAKE_SOURCE_DIR}/avx_checker.cc)
add_executable(match_engine ${SOURCES})

# 检查返回值是否为1，表示支持 AVX 和 AVX512F
if(RUN_RESULT EQUAL 1)
    message(STATUS "CPU supports AVX and AVX512F, adding avx512 compilation flags.")
    target_compile_options(match_engine PRIVATE -mavx512f -mavx512dq -mevex512)
else()
    message(STATUS "CPU does not support AVX512F, add maxv only.")
    target_compile_options(match_engine PRIVATE -mavx)
endif()


target_link_libraries(match_engine ${absl_LIBRARIES})
target_link_libraries(match_engine fmt::fmt)
target_link_libraries(match_engine TBB::tbb TBB::tbbmalloc)
target_link_libraries(match_engine spdlog::spdlog_header_only)
