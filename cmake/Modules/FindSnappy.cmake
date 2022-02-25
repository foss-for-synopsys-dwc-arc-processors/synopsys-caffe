# Find the Snappy libraries
#
# The following variables are optionally searched for defaults
#  Snappy_ROOT_DIR:    Base directory where all Snappy components are found
#
# The following are set after configuration is done:
#  SNAPPY_FOUND
#  Snappy_INCLUDE_DIRS
#  Snappy_LIBRARIES

find_path(Snappy_INCLUDE_DIRS
  NAMES snappy.h
  HINTS ${snappy_ROOT_DIR}/include)

find_library(Snappy_LIBRARIES
  NAMES snappy
  HINTS ${snappy_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SNAPPY DEFAULT_MSG SNAPPY_LIBRARIES SNAPPY_INCLUDE_DIRS)

mark_as_advanced(
  SNAPPY_LIBRARIES
  SNAPPY_INCLUDE_DIRS)

if(SNAPPY_FOUND AND NOT (TARGET Snappy::snappy))

  add_library (Snappy::snappy UNKNOWN IMPORTED)
  set_target_properties(Snappy::snappy
    PROPERTIES
      IMPORTED_LOCATION ${SNAPPY_LIBRARIES}
      INTERFACE_INCLUDE_DIRECTORIES ${SNAPPY_INCLUDE_DIRS})
      
  message(STATUS "Found Snappy  (include: ${SNAPPY_INCLUDE_DIRS}, library: ${SNAPPY_LIBRARIES})")

  caffe_parse_header(${SNAPPY_INCLUDE_DIRS}/snappy-stubs-public.h
                     SNAPPY_VERION_LINES SNAPPY_MAJOR SNAPPY_MINOR SNAPPY_PATCHLEVEL)
  set(Snappy_VERSION "${SNAPPY_MAJOR}.${SNAPPY_MINOR}.${SNAPPY_PATCHLEVEL}")
endif()

