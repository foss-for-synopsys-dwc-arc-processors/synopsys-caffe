file(GLOB_RECURSE examples_srcs "${PROJECT_SOURCE_DIR}/examples/*.cpp")

foreach(source_file ${examples_srcs})
  # get file name
  get_filename_component(name ${source_file} NAME_WE)

  # get folder name
  get_filename_component(path ${source_file} PATH)
  get_filename_component(folder ${path} NAME_WE)


  message(STATUS "Justin hack for Linker Path issue --- START")
  
  message("CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
  message("Caffe_LINK: ${Caffe_LINK}")
  message("Caffe_LINKER_LIBS: ${Caffe_LINKER_LIBS}")
  message("CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
  message("CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
  message("CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES: ${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}")
  
  # reset
  set(CMAKE_EXE_LINKER_FLAGS "")
  set(CMAKE_EXE_LINKER_FLAGS "/machine:x64")
  # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  ${CMAKE_BINARY_DIR}/lib/Release  C:/vcpkg/installed/x64-windows-static/lib ")
  
  add_library(justinstatic_${name} UNKNOWN IMPORTED)
  set_property(TARGET justinstatic_${name} PROPERTY IMPORTED_LOCATION "D:/a/synopsys-caffe/synopsys-caffe/build/lib/Release/caffe.lib")
    
  # link_directories(${CMAKE_BINARY_DIR}/lib/Release)
  # set(HACKER_LINK_LIBS caffe.lib)
  
  add_executable(${name} ${source_file})
  target_link_libraries(${name} justinstatic_${name} )
  
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
