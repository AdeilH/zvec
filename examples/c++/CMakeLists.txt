cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
project(zvec-example-c++)
set(CMAKE_CXX_STANDARD 17)

# Enable compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Control whether to force-link static archives to keep registration objects.
option(ZVEC_EXAMPLES_WHOLE_ARCHIVE "Force-link zvec core static libs" ON)

# --- Paths to Zvec and dependencies ---
# Allow overrides via -DZVEC_ROOT, -DZVEC_INCLUDE_DIR, -DZVEC_LIB_DIR, -DZVEC_DEPENDENCY_LIB_DIR
if(NOT DEFINED ZVEC_ROOT OR ZVEC_ROOT STREQUAL "")
    get_filename_component(ZVEC_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
endif()

if(NOT DEFINED ZVEC_INCLUDE_DIR OR ZVEC_INCLUDE_DIR STREQUAL "")
    set(ZVEC_INCLUDE_DIR "${ZVEC_ROOT}/src/include")
endif()
if(NOT DEFINED ZVEC_LIB_DIR OR ZVEC_LIB_DIR STREQUAL "")
    set(ZVEC_LIB_DIR "${ZVEC_ROOT}/build/lib")
endif()
if(NOT DEFINED ZVEC_DEPENDENCY_LIB_DIR OR ZVEC_DEPENDENCY_LIB_DIR STREQUAL "")
    set(ZVEC_DEPENDENCY_LIB_DIR "${ZVEC_ROOT}/build/external/usr/local/lib")
endif()


# Add include and library search paths
include_directories(${ZVEC_INCLUDE_DIR})
link_directories(${ZVEC_LIB_DIR} ${ZVEC_DEPENDENCY_LIB_DIR})

# --- Determine debug/release library names ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(GLOG_LIB glogd)
    set(GFLAGS_LIB gflags_nothreads_debug)
    set(PROTOBUF_LIB protobufd)
else()
    set(GLOG_LIB glog)
    set(GFLAGS_LIB gflags_nothreads)
    set(PROTOBUF_LIB protobuf)
endif()

# --- Arrow/Parquet resolution ---
set(ARROW_LIB_TARGET "")
set(PARQUET_LIB_TARGET "")
set(ARROW_DEPENDS_TARGET "")
set(ARROW_COMPUTE_TARGET "")
set(ARROW_ACERO_TARGET "")
set(ARROW_DATASET_TARGET "")

if(TARGET Arrow::arrow_static)
    set(ARROW_LIB_TARGET Arrow::arrow_static)
endif()
if(TARGET Arrow::parquet_static)
    set(PARQUET_LIB_TARGET Arrow::parquet_static)
endif()
if(TARGET Arrow::arrow_depends)
    set(ARROW_DEPENDS_TARGET Arrow::arrow_depends)
endif()
if(TARGET Arrow::arrow_compute)
    set(ARROW_COMPUTE_TARGET Arrow::arrow_compute)
endif()
if(TARGET Arrow::arrow_acero)
    set(ARROW_ACERO_TARGET Arrow::arrow_acero)
endif()
if(TARGET Arrow::arrow_dataset)
    set(ARROW_DATASET_TARGET Arrow::arrow_dataset)
endif()

if(ARROW_LIB_TARGET STREQUAL "")
    find_library(ARROW_LIB_TARGET NAMES arrow PATHS ${ZVEC_DEPENDENCY_LIB_DIR} NO_DEFAULT_PATH)
endif()
if(PARQUET_LIB_TARGET STREQUAL "")
    find_library(PARQUET_LIB_TARGET NAMES parquet PATHS ${ZVEC_DEPENDENCY_LIB_DIR} NO_DEFAULT_PATH)
endif()
if(ARROW_DEPENDS_TARGET STREQUAL "")
    find_library(ARROW_DEPENDS_TARGET NAMES arrow_bundled_dependencies PATHS ${ZVEC_DEPENDENCY_LIB_DIR} NO_DEFAULT_PATH)
endif()
if(ARROW_COMPUTE_TARGET STREQUAL "")
    find_library(ARROW_COMPUTE_TARGET NAMES arrow_compute PATHS ${ZVEC_DEPENDENCY_LIB_DIR} NO_DEFAULT_PATH)
endif()
if(ARROW_ACERO_TARGET STREQUAL "")
    find_library(ARROW_ACERO_TARGET NAMES arrow_acero PATHS ${ZVEC_DEPENDENCY_LIB_DIR} NO_DEFAULT_PATH)
endif()
if(ARROW_DATASET_TARGET STREQUAL "")
    find_library(ARROW_DATASET_TARGET NAMES arrow_dataset PATHS ${ZVEC_DEPENDENCY_LIB_DIR} NO_DEFAULT_PATH)
endif()

# --- Dependency groups ---
find_package(Threads REQUIRED)

set(zvec_ailego_deps
    ${ARROW_LIB_TARGET}
    ${PARQUET_LIB_TARGET}
    ${ARROW_DEPENDS_TARGET}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
)

set(zvec_core_deps
    # empty
)

set(zvec_db_deps
    roaring
    rocksdb
    ${ARROW_LIB_TARGET}
    ${ARROW_ACERO_TARGET}
    ${ARROW_DEPENDS_TARGET}
    ${ARROW_COMPUTE_TARGET}
    ${ARROW_DATASET_TARGET}
    ${PARQUET_LIB_TARGET}
    antlr4-runtime
    ${GLOG_LIB}
    ${GFLAGS_LIB}
    ${PROTOBUF_LIB}
    lz4
)

# --- Create INTERFACE targets for Zvec components ---

# zvec_ailego: links libzvec_ailego.a + its deps
add_library(zvec-ailego INTERFACE)
target_link_libraries(zvec-ailego INTERFACE
    ${ZVEC_LIB_DIR}/libzvec_ailego.a
    ${zvec_ailego_deps}
)

# zvec_core: links libzvec_core.a via special flags (handled externally), but declare logical deps
add_library(zvec-core INTERFACE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(ZVEC_EXAMPLES_WHOLE_ARCHIVE)
        target_link_libraries(zvec-core INTERFACE
            -Wl,--whole-archive
            ${ZVEC_LIB_DIR}/libzvec_core.a
            -Wl,--no-whole-archive
            -Wl,--start-group
            ${ZVEC_LIB_DIR}/libzvec_ailego.a
            ${zvec_core_deps}
            ${zvec_ailego_deps}
            -Wl,--end-group
        )
    else()
        target_link_libraries(zvec-core INTERFACE
            ${ZVEC_LIB_DIR}/libzvec_core.a
            -Wl,--start-group
            ${ZVEC_LIB_DIR}/libzvec_ailego.a
            ${zvec_core_deps}
            ${zvec_ailego_deps}
            -Wl,--end-group
        )
    endif()
elseif(APPLE)
    if(ZVEC_EXAMPLES_WHOLE_ARCHIVE)
        target_link_libraries(zvec-core INTERFACE
            -Wl,-force_load ${ZVEC_LIB_DIR}/libzvec_core.a
            ${ZVEC_LIB_DIR}/libzvec_ailego.a
            ${zvec_core_deps}
            ${zvec_ailego_deps}
        )
    else()
        target_link_libraries(zvec-core INTERFACE
            ${ZVEC_LIB_DIR}/libzvec_core.a
            ${ZVEC_LIB_DIR}/libzvec_ailego.a
            ${zvec_core_deps}
            ${zvec_ailego_deps}
        )
    endif()
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# zvec_db: links libzvec_db.a + all deps
add_library(zvec-db INTERFACE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(zvec-db INTERFACE
        ${ZVEC_LIB_DIR}/libzvec_db.a
        zvec-core
        zvec-ailego
        -Wl,--start-group
        ${zvec_db_deps}
        -Wl,--end-group
    )
elseif(APPLE)
    target_link_libraries(zvec-db INTERFACE
        ${ZVEC_LIB_DIR}/libzvec_db.a
        zvec-core
        zvec-ailego
        ${zvec_db_deps}
    )
endif()


# --- Main executable ---
add_executable(db-example db/main.cc)
target_link_libraries(db-example PRIVATE
    zvec-db
)

add_executable(core-example core/main.cc)
target_link_libraries(core-example PRIVATE
    zvec-core
)

add_executable(ailego-example ailego/main.cc)
target_link_libraries(ailego-example PRIVATE
    zvec-ailego
)
