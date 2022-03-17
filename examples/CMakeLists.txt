file(GLOB_RECURSE examples_srcs "${PROJECT_SOURCE_DIR}/examples/*.cpp")

foreach(source_file ${examples_srcs})
  # get file name
  get_filename_component(name ${source_file} NAME_WE)

  # get folder name
  get_filename_component(path ${source_file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  message(STATUS "${CMAKE_CURRENT_BINARY_DIR}")
  message(STATUS "${CMAKE_BINARY_DIR}")
  
  add_library(caffestatic UNKNOWN IMPORTED)
  set_property(TARGET caffestatic PROPERTY IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/lib/Release/caffe.lib")
  
  add_executable(${name} ${source_file})
  target_link_libraries(${name} caffestatic )
  
  target_include_directories(${name} PUBLIC ${Caffe_INCLUDE_DIR})
  caffe_default_properties(${name})



  # set back RUNTIME_OUTPUT_DIRECTORY
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/examples/${folder}")

  caffe_set_solution_folder(${name} examples)

  # install
  install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})




  if(UNIX OR APPLE)
    # Funny command to make tutorials work
    # TODO: remove in future as soon as naming is standardized everywhere
    set(__outname ${PROJECT_BINARY_DIR}/examples/${folder}/${name}${Caffe_POSTFIX})
    add_custom_command(TARGET ${name} POST_BUILD
                       COMMAND ln -sf "${__outname}" "${__outname}.bin")
  endif()
endforeach()
