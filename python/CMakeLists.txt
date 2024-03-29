if(NOT HAVE_PYTHON)
  message(STATUS "Python interface is disabled or not all required dependencies found. Building without it...")
  return()
endif()

file(GLOB_RECURSE python_srcs ${PROJECT_SOURCE_DIR}/python/*.cpp)

add_library(pycaffe MODULE ${python_srcs})
caffe_default_properties(pycaffe)
set_target_properties(pycaffe PROPERTIES PREFIX "" OUTPUT_NAME "_caffe")
if(MSVC)
  set_target_properties(pycaffe PROPERTIES SUFFIX ".pyd")
endif()

if(MSVC)
  if(NOT "${PYTHON_VERSION_STRING}" VERSION_LESS "3.8.0")
    add_compile_options(
        $<$<CONFIG:>:/MD> #---------|
        $<$<CONFIG:Release>:/MD> #--|
        $<$<CONFIG:Debug>:/MDd> #---|
    )
  endif()
endif()

if(NOT "${PYTHON_VERSION_STRING}" VERSION_LESS "3.8.0")
  target_include_directories(pycaffe PUBLIC ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${Caffe_INCLUDE_DIR})
  target_link_libraries(pycaffe PUBLIC ${Caffe_LINKER_LIBS} ${Caffe_LINK} ${PYTHON_LIBRARIES})
else()
  target_include_directories(pycaffe PUBLIC ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
  target_link_libraries(pycaffe PUBLIC ${Caffe_LINK} ${PYTHON_LIBRARIES})
endif()

if(UNIX OR APPLE)
    set(__linkname "${PROJECT_SOURCE_DIR}/python/caffe/_caffe.so")
    add_custom_command(TARGET pycaffe POST_BUILD
                       COMMAND ln -sf $<TARGET_LINKER_FILE:pycaffe> "${__linkname}"
                       COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/python/caffe/proto
                       COMMAND touch ${PROJECT_SOURCE_DIR}/python/caffe/proto/__init__.py
                       COMMAND cp ${proto_gen_folder}/*.py ${PROJECT_SOURCE_DIR}/python/caffe/proto/
                       COMMENT "Creating symlink ${__linkname} -> ${PROJECT_BINARY_DIR}/lib/_caffe${Caffe_POSTFIX}.so")
elseif(WIN32)
    add_custom_command(TARGET pycaffe POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:pycaffe> ${PROJECT_SOURCE_DIR}/python/caffe
                       COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/python/caffe/proto
                       COMMAND ${CMAKE_COMMAND} -E touch ${PROJECT_SOURCE_DIR}/python/caffe/proto/__init__.py
                       COMMAND (robocopy  "\"${proto_gen_folder}\" \"${PROJECT_SOURCE_DIR}/python/caffe/proto\" *.py") ^& IF %ERRORLEVEL% LEQ 4 set ERRORLEVEL=0
                       COMMENT "Creating symlink ${__linkname} -> ${PROJECT_BINARY_DIR}/lib/_caffe.pyd")
endif()

if(MSVC AND COPY_PREREQUISITES)
  caffe_copy_prerequisites(pycaffe DESTINATION ${PROJECT_SOURCE_DIR}/python/caffe USE_HARD_LINKS)
endif()

# ---[ Install
# scripts
file(GLOB python_files *.py requirements.txt)
install(FILES ${python_files} DESTINATION python)

# module
install(DIRECTORY caffe
    DESTINATION python
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "ilsvrc_2012_mean.npy"
    PATTERN "test" EXCLUDE
    )

# _caffe.so
install(TARGETS pycaffe  DESTINATION python/caffe)

if(MSVC AND INSTALL_PREREQUISITES)
  caffe_install_prerequisites(pycaffe DESTINATION python/caffe)
endif()


