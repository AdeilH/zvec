cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
project(zvec)
set(CC_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=return-type")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror=return-type")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

if(NOT DEFINED PROJECT_ROOT_DIR OR NOT PROJECT_ROOT_DIR)
    set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR} CACHE PATH "Root directory of the project" FORCE)
endif()

message(STATUS "PROJECT_ROOT_DIR = ${PROJECT_ROOT_DIR}")

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(GNUInstallDirs)

include_directories(${PROJECT_ROOT_DIR}/src/include)
include_directories(${PROJECT_ROOT_DIR}/src)

option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)
message(STATUS "BUILD_PYTHON_BINDINGS:${BUILD_PYTHON_BINDINGS}")

option(BUILD_TOOLS "Build tools" ON)
message(STATUS "BUILD_TOOLS:${BUILD_TOOLS}")

option(BUILD_MONOLITH_LIBS "Build monolithic zvec static/shared libraries" OFF)
message(STATUS "BUILD_MONOLITH_LIBS:${BUILD_MONOLITH_LIBS}")

if(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|x64")
    set(PC_FROM_UCONTEXT "uc_mcontext.gregs[REG_RIP]" CACHE STRING "glog ucontext pc field" FORCE)
endif()

if(BUILD_PYTHON_BINDINGS AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|x64")
    set(_PY_NO_AVX512_FLAGS
        "-mno-avx512f -mno-avx512cd -mno-avx512dq -mno-avx512bw -mno-avx512vl -mno-avx512ifma -mno-avx512vbmi"
    )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_PY_NO_AVX512_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_PY_NO_AVX512_FLAGS}")
endif()

if(BUILD_MONOLITH_LIBS)
    # Monolithic shared library
    add_library(zvec_all SHARED src/monolith/zvec_all_stub.cc)
    if(UNIX AND NOT APPLE)
        target_link_libraries(zvec_all PRIVATE
            -Wl,--whole-archive
            zvec_c_static
            zvec_db
            zvec_core
            zvec_ailego
            -Wl,--no-whole-archive
        )
    elseif(APPLE)
        target_link_libraries(zvec_all PRIVATE
            -Wl,-force_load,$<TARGET_FILE:zvec_c_static>
            -Wl,-force_load,$<TARGET_FILE:zvec_db>
            -Wl,-force_load,$<TARGET_FILE:zvec_core>
            -Wl,-force_load,$<TARGET_FILE:zvec_ailego>
        )
    else()
        target_link_libraries(zvec_all PRIVATE
            zvec_c_static
            zvec_db
            zvec_core
            zvec_ailego
        )
    endif()

    # Monolithic static archive (merged)
    set(ZVEC_MONOLITH_STATIC "${CMAKE_BINARY_DIR}/lib/libzvec_all.a")
    set(ZVEC_MONOLITH_LIB_LIST_FILE "${CMAKE_BINARY_DIR}/lib/zvec_all_libs.txt")
    set(ZVEC_MONOLITH_ZVEC_LIBS
        $<TARGET_FILE:zvec_c_static>
        $<TARGET_FILE:zvec_ailego>
        $<TARGET_FILE:zvec_core>
        $<TARGET_FILE:zvec_db>
    )
    file(GENERATE
        OUTPUT "${ZVEC_MONOLITH_LIB_LIST_FILE}"
        CONTENT "$<JOIN:${ZVEC_MONOLITH_ZVEC_LIBS},\n>\n"
    )
    add_custom_command(
        OUTPUT "${ZVEC_MONOLITH_STATIC}"
        COMMAND "${CMAKE_COMMAND}"
            "-DOUT=${ZVEC_MONOLITH_STATIC}"
            "-DAR=${CMAKE_AR}"
            "-DRANLIB=${CMAKE_RANLIB}"
            "-DIS_APPLE=${APPLE}"
            "-DLIBTOOL=${CMAKE_LIBTOOL}"
            "-DLIBS_FILE=${ZVEC_MONOLITH_LIB_LIST_FILE}"
            "-DLIB_DIRS=${CMAKE_BINARY_DIR}/external/usr/local/lib"
            -P "${PROJECT_ROOT_DIR}/cmake/merge_static_libs.cmake"
        DEPENDS
            zvec_c_static
            zvec_db
            zvec_core zvec_ailego
    )
    add_custom_target(zvec_all_static_archive DEPENDS "${ZVEC_MONOLITH_STATIC}")
    add_library(zvec_all_static STATIC IMPORTED GLOBAL)
    set_target_properties(zvec_all_static PROPERTIES IMPORTED_LOCATION "${ZVEC_MONOLITH_STATIC}")
    add_dependencies(zvec_all_static zvec_all_static_archive)

    install(TARGETS zvec_all
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
    install(FILES "${ZVEC_MONOLITH_STATIC}"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
endif()

install(DIRECTORY "${PROJECT_ROOT_DIR}/src/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h"
)

cc_directory(thirdparty)
cc_directories(src)
cc_directories(tests)


if(BUILD_TOOLS)
    cc_directories(tools)
endif()

git_version(GIT_SRCS_VER ${PROJECT_ROOT_DIR})
set(CPACK_PACKAGE_VERSION ${GIT_SRCS_VER})
set(CPACK_PACKAGE_NAME zvec)
include(CPack)

if(BUILD_PYTHON_BINDINGS)
    if(APPLE)
        set(CMAKE_STRIP "")
        message(STATUS "Disabled strip on macOS to preserve code signature")
    endif()

    include(GNUInstallDirs)
    if(DEFINED SKBUILD_PLATLIB_DIR)
        set(ZVEC_PY_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}")
    elseif(DEFINED Python_SITEARCH)
        set(ZVEC_PY_INSTALL_DIR "${Python_SITEARCH}")
    else()
        set(ZVEC_PY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    endif()

    MESSAGE(STATUS "Zvec install path: ${ZVEC_PY_INSTALL_DIR}")
    install(TARGETS _zvec LIBRARY DESTINATION ${ZVEC_PY_INSTALL_DIR})
endif()


cc_directories(examples/c++)

install(TARGETS zvec_c zvec_c_static
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
