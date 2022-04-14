# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# ---[the workaround for Boost throw exception trigger LNK fatal error
SET(TPN_WIN32 "/EHsc")

# ---[MSVC static linking
if(MSVC)
    add_compile_options(
        $<$<CONFIG:>:/MT> #---------|
        $<$<CONFIG:Debug>:/MTd> #---|-- Statically link the runtime libraries
        $<$<CONFIG:Release>:/MD> #--|
    )
endif()

# Build each source file independently
foreach(source ${srcs})
  get_filename_component(name ${source} NAME_WE)

  # caffe target already exits
  if(name MATCHES "caffe")
    set(name ${name}.bin)
  endif()

  # target
  add_executable(${name} ${source})
  target_link_libraries(${name} PUBLIC  caffe  glog::glog  gflags::gflags  leveldb::leveldb   ${Caffe_LINKER_LIBS})
  target_include_directories(${name} PUBLIC ${Caffe_INCLUDE_DIR})
  
  caffe_default_properties(${name})

  # set back RUNTIME_OUTPUT_DIRECTORY
  caffe_set_runtime_directory(${name} "${PROJECT_BINARY_DIR}/tools")
  caffe_set_solution_folder(${name} tools)

  # restore output name without suffix
  if(name MATCHES "caffe.bin")
    set_target_properties(${name} PROPERTIES OUTPUT_NAME caffe)
  endif()

  # Install
  install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})

endforeach(source)
