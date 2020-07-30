if(USE_VITIS_AI)
  set(PYXIR_SHARED_LIB libpyxir.so)
  find_package(PythonInterp 3.6 REQUIRED)
  if(NOT PYTHON)
    find_program(PYTHON NAMES python3 python3.6)
  endif()
  if(PYTHON)
    execute_process(COMMAND "${PYTHON}" "-c"
      "import pyxir as px; print(px.get_include_dir()); print(px.get_lib_dir());"
      RESULT_VARIABLE __result
      OUTPUT_VARIABLE __output
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(__result MATCHES 0)
      string(REGEX REPLACE ";" "\\\\;" __values ${__output})
      string(REGEX REPLACE "\r?\n" ";"    __values ${__values})
      list(GET __values 0 PYXIR_INCLUDE_DIR)
      list(GET __values 1 PYXIR_LIB_DIR)
    endif()

  else()
  message(STATUS "To find Pyxir, Python interpreter is required to be found.")
  endif()

message(STATUS "Build with contrib.vitisai")
include_directories(${PYXIR_INCLUDE_DIR})  
file(GLOB VAI_CONTRIB_SRC src/runtime/contrib/vitis_ai/*.cc)
link_directories(${PYXIR_LIB_DIR})
list(APPEND TVM_RUNTIME_LINKER_LIBS "pyxir")
list(APPEND RUNTIME_SRCS ${VAI_CONTRIB_SRC})
endif(USE_VAI)

