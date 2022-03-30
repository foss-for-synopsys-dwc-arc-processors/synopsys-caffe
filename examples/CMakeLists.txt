file(GLOB_RECURSE examples_srcs "${PROJECT_SOURCE_DIR}/examples/*.cpp")



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
  # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  ${CMAKE_BINARY_DIR}/lib/Release  C:/vcpkg/installed/x64-windows/lib ")
  
  add_library(justinstatic_${name} UNKNOWN IMPORTED)
  set_property(TARGET justinstatic_${name} PROPERTY IMPORTED_LOCATION "D:/a/synopsys-caffe/synopsys-caffe/build/lib/Release/caffe.lib")
 
  
  add_library(justin_glog_${name} UNKNOWN IMPORTED)
  set_property(TARGET justin_glog_${name} PROPERTY IMPORTED_LOCATION "C:/vcpkg/installed/x64-windows/lib/glog.lib")
  
  add_library(justin_leveldb_${name} UNKNOWN IMPORTED)
  set_property(TARGET justin_leveldb_${name} PROPERTY IMPORTED_LOCATION "C:/vcpkg/installed/x64-windows/lib/leveldb.lib")
  
  add_library(justin_protobuf_${name} UNKNOWN IMPORTED)
  set_property(TARGET justin_protobuf_${name} PROPERTY IMPORTED_LOCATION "C:/vcpkg/installed/x64-windows/lib/libprotobuf.lib")
    
  # link_directories(${CMAKE_BINARY_DIR}/lib/Release)
  # set(HACKER_LINK_LIBS caffe.lib)
  
  add_executable(${name} ${source_file})
  
  # this line is working
  # target_link_libraries(${name} PUBLIC justinstatic_${name} justin_protobuf_${name}  justin_leveldb_${name}  justin_glog_${name}  ${Caffe_LINKER_LIBS} )
  target_link_libraries(${name} PUBLIC  caffeproto  caffe   justin_protobuf_${name}   justin_leveldb_${name}  justin_glog_${name}  ${Caffe_LINKER_LIBS} ${Caffe_LINK})
  
  
  
  # add some preprocessor macros for linker bugs
  target_compile_definitions(${name} PRIVATE  _VARIADIC_MAX=10 GFLAGS_DLL_DECLARE_FLAG=  GFLAGS_DLL_DEFINE_FLAG=  GOOGLE_GLOG_DLL_DECL=  GLOG_NO_ABBREVIATED_SEVERITIES  )
  
  # boost linking issue
  # add_definitions("/EHsc")
  
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
