# - Try to find Glog
#
# The following variables are optionally searched for defaults
#  GLOG_ROOT_DIR:            Base directory where all GLOG components are found
#
# The following are set after configuration is done:
#  GLOG_FOUND
#  GLOG_INCLUDE_DIRS
#  GLOG_LIBRARIES
#  GLOG_LIBRARYRARY_DIRS

include(FindPackageHandleStandardArgs)

set(GLOG_ROOT_DIR "" CACHE PATH "Folder contains Google glog")

find_path(GLOG_INCLUDE_DIR glog/logging.h
    PATHS ${GLOG_ROOT_DIR})

if(MSVC)
    # required for py38 when glog version updates (could also be added in libraries\lib\cmake\glog\glog-config.cmake instead)
    set (glog_LIBRARY ${CONDA_LIB_PATH}/glog.lib)
    set (glog_LIBRARIES ${glog_LIBRARY})
    set (glog_INCLUDE_DIRS ${glog_INCLUDE_DIR})

    # rely on glog-config.cmake
    find_package(glog NO_MODULE)
    set(GLOG_LIBRARY ${glog_LIBRARIES})
    set(GLOG_INCLUDE_DIR ${glog_INCLUDE_DIRS})
    add_compile_definitions(GLOG_NO_ABBREVIATED_SEVERITIES)
else()
    find_library(GLOG_LIBRARY glog
        PATHS ${GLOG_ROOT_DIR}
        PATH_SUFFIXES lib lib64)
endif()

find_package_handle_standard_args(Glog DEFAULT_MSG GLOG_INCLUDE_DIR GLOG_LIBRARY)

if(GLOG_FOUND)
  set(GLOG_INCLUDE_DIRS ${GLOG_INCLUDE_DIR})
  set(GLOG_LIBRARIES ${GLOG_LIBRARY})
  message(STATUS "Found glog    (include: ${GLOG_INCLUDE_DIR}, library: ${GLOG_LIBRARY})")
  mark_as_advanced(GLOG_ROOT_DIR GLOG_LIBRARY_RELEASE GLOG_LIBRARY_DEBUG
                                 GLOG_LIBRARY GLOG_INCLUDE_DIR)
endif()
