cmake_minimum_required(VERSION 3.15)
project(xad_ladder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Same compiler flags auto-differentiation/ad-benchmarks uses for its own
# XAD rows, so this measurement is held to the same bar as the vendor's own
# published numbers.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -DNDEBUG -mavx2 -mfma)
endif()

include(FetchContent)
set(XAD_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(xad
    GIT_REPOSITORY https://github.com/auto-differentiation/xad.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(xad)

add_executable(xad_ladder xad_ladder.cpp)
target_link_libraries(xad_ladder PRIVATE xad)
