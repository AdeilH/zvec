cmake_policy(SET CMP0007 NEW)

if(NOT DEFINED OUT)
  message(FATAL_ERROR "OUT not set")
endif()
if(NOT DEFINED LIBS)
  set(LIBS "")
endif()
if(DEFINED LIBS_FILE AND NOT LIBS_FILE STREQUAL "")
  if(NOT EXISTS "${LIBS_FILE}")
    message(FATAL_ERROR "LIBS_FILE not found: ${LIBS_FILE}")
  endif()
  file(READ "${LIBS_FILE}" _libs_file_content)
  string(REPLACE "\r\n" "\n" _libs_file_content "${_libs_file_content}")
  string(REPLACE "\n" ";" _libs_file_content "${_libs_file_content}")
  list(REMOVE_ITEM _libs_file_content "")
  set(LIBS "${_libs_file_content}")
endif()
if(NOT DEFINED LIB_DIRS)
  set(LIB_DIRS "")
endif()
if(NOT DEFINED AR)
  set(AR "ar")
endif()
if(NOT DEFINED RANLIB)
  set(RANLIB "")
endif()
if(NOT DEFINED IS_APPLE)
  set(IS_APPLE FALSE)
endif()
if(NOT DEFINED LIBTOOL)
  set(LIBTOOL "libtool")
endif()

get_filename_component(_out_dir "${OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${_out_dir}")

set(_all_libs "")

foreach(_lib IN LISTS LIBS)
  if(EXISTS "${_lib}")
    list(APPEND _all_libs "${_lib}")
  else()
    message(FATAL_ERROR "Missing static library: ${_lib}")
  endif()
endforeach()

foreach(_dir IN LISTS LIB_DIRS)
  if(NOT IS_DIRECTORY "${_dir}")
    message(FATAL_ERROR "Library directory not found: ${_dir}")
  endif()
  file(GLOB _dir_libs "${_dir}/*.a")
  list(APPEND _all_libs ${_dir_libs})
endforeach()

list(REMOVE_DUPLICATES _all_libs)

if(_all_libs STREQUAL "")
  message(FATAL_ERROR "No static libraries found to merge.")
endif()

file(REMOVE "${OUT}")

if(IS_APPLE)
  execute_process(
      COMMAND "${LIBTOOL}" -static -o "${OUT}" ${_all_libs}
      RESULT_VARIABLE _merge_res
      ERROR_VARIABLE _merge_err
  )
  if(NOT _merge_res EQUAL 0)
    message(FATAL_ERROR "libtool failed: ${_merge_err}")
  endif()
else()
  set(_mri "${OUT}.mri")
  file(WRITE "${_mri}" "CREATE ${OUT}\n")
  foreach(_lib IN LISTS _all_libs)
    file(APPEND "${_mri}" "ADDLIB ${_lib}\n")
  endforeach()
  file(APPEND "${_mri}" "SAVE\nEND\n")

  execute_process(
      COMMAND "${AR}" -M
      INPUT_FILE "${_mri}"
      RESULT_VARIABLE _merge_res
      ERROR_VARIABLE _merge_err
  )
  if(NOT _merge_res EQUAL 0)
    message(FATAL_ERROR "ar -M failed: ${_merge_err}")
  endif()

  if(RANLIB AND NOT RANLIB STREQUAL "")
    execute_process(COMMAND "${RANLIB}" "${OUT}")
  endif()
endif()
