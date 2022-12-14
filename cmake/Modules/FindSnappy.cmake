# Find the Snappy libraries
#
# The following variables are optionally searched for defaults
#  Snappy_ROOT_DIR:    Base directory where all Snappy components are found
#
# The following are set after configuration is done:
#  SNAPPY_FOUND
#  Snappy_INCLUDE_DIR
#  Snappy_LIBRARIES
if(MSVC)
  # rely on snappy-config.cmake
  find_package(Snappy NO_MODULE)
  set(Snappy_INCLUDE_DIR "C:\/Users\/yche\/AppData\/Local\/Continuum\/miniconda3-4.5.4\/envs\/py38\/Library\/include")
  set(Snappy_LIBRARIES "C:\/Users\/yche\/AppData\/Local\/Continuum\/miniconda3-4.5.4\/envs\/py38\/Library\/bin\/snappy.dll")
else()
  find_path(Snappy_INCLUDE_DIR NAMES snappy.h
                              PATHS ${SNAPPY_ROOT_DIR} ${SNAPPY_ROOT_DIR}/include)

  find_library(Snappy_LIBRARIES NAMES snappy
                                PATHS ${SNAPPY_ROOT_DIR} ${SNAPPY_ROOT_DIR}/lib)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Snappy DEFAULT_MSG Snappy_INCLUDE_DIR Snappy_LIBRARIES)

if(SNAPPY_FOUND)
  message(STATUS "Found Snappy  (include: ${Snappy_INCLUDE_DIR}, library: ${Snappy_LIBRARIES})")
  mark_as_advanced(Snappy_INCLUDE_DIR Snappy_LIBRARIES)

  caffe_parse_header(${Snappy_INCLUDE_DIR}/snappy-stubs-public.h
                     SNAPPY_VERION_LINES SNAPPY_MAJOR SNAPPY_MINOR SNAPPY_PATCHLEVEL)
  set(Snappy_VERSION "${SNAPPY_MAJOR}.${SNAPPY_MINOR}.${SNAPPY_PATCHLEVEL}")
endif()

