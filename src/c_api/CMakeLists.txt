include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
    NAME zvec_c STATIC SHARED STRICT
    SRCS zvec_c.cc
    INCS ${PROJECT_ROOT_DIR}/src/include
    PUBINCS ${PROJECT_ROOT_DIR}/src/include
    LIBS zvec_db zvec_core zvec_ailego
    VERSION "${PROXIMA_ZVEC_VERSION}"
)
