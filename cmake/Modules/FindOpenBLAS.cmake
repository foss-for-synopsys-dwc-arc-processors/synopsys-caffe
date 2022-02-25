


SET(Open_BLAS_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/openblas
  /usr/include/openblas-base
  /usr/local/include
  /usr/local/include/openblas
  /usr/local/include/openblas-base
  /opt/OpenBLAS/include
  $ENV{OpenBLAS_HOME}
  $ENV{OpenBLAS_HOME}/include
  C:/vcpkg/installed/x64-windows-static/include
  C:/vcpkg/installed/x64-windows/include
  C:/vcpkg/installed/x64-windows-static/openblas
  C:/vcpkg/installed/x64-windows-static/openblas/include
  C:/vcpkg/installed/x64-windows/openblas
  C:/vcpkg/installed/x64-windows/openblas/include
  C:/vcpkg/installed/x64-windows-static/OpenBLAS
  C:/vcpkg/installed/x64-windows-static/OpenBLAS/include
  C:/vcpkg/installed/x64-windows/OpenBLAS
  C:/vcpkg/installed/x64-windows/OpenBLAS/include
)

SET(Open_BLAS_LIB_SEARCH_PATHS
        /lib/
        /lib/openblas-base
        /lib64/
        /usr/lib
        /usr/lib/openblas-base
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/OpenBLAS/lib
        $ENV{OpenBLAS}cd
        $ENV{OpenBLAS}/lib
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/lib
        C:/vcpkg/installed/x64-windows-static/lib
        C:/vcpkg/installed/x64-windows-static/lib/openblas
        C:/vcpkg/installed/x64-windows-static/lib/OpenBLAS
        C:/vcpkg/installed/x64-windows/lib
        C:/vcpkg/installed/x64-windows/lib/openblas
        C:/vcpkg/installed/x64-windows/lib/OpenBLAS
        C:/vcpkg/installed/x64-windows-static/openblas
        C:/vcpkg/installed/x64-windows-static/openblas/lib
        C:/vcpkg/installed/x64-windows/openblas
        C:/vcpkg/installed/x64-windows/openblas/lib
        C:/vcpkg/installed/x64-windows-static/OpenBLAS
        C:/vcpkg/installed/x64-windows-static/OpenBLAS/lib
        C:/vcpkg/installed/x64-windows/OpenBLAS
        C:/vcpkg/installed/x64-windows/OpenBLAS/lib
 )

if(MSVC)
  set(OpenBLAS_LIB_NAMES 
    libopenblas.dll.a
    libopenblas.dll.so
    libopenblas.dll
    libopenblas.lib
    libopenblas.a
    libopenblas.so
    libopenblas
    libOpenBLAS.dll.a
    libOpenBLAS.dll.so
    libOpenBLAS.dll
    libOpenBLAS.lib
    libOpenBLAS.a
    libOpenBLAS.so
    libOpenBLAS
    )
else()
  set(OpenBLAS_LIB_NAMES openblas)
endif()

# FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS})
# FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS_SUFFIXES openblas)
FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS_SUFFIXES OpenBLAS)
MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
include_directories(${OpenBLAS_INCLUDE_DIR})

# FIND_LIBRARY(OpenBLAS_LIB NAMES ${OpenBLAS_LIB_NAMES} PATHS ${Open_BLAS_LIB_SEARCH_PATHS})
FIND_LIBRARY(OpenBLAS_LIB NAMES ${OpenBLAS_LIB_NAMES}  PATHS_SUFFIXES OpenBLAS)

SET(OpenBLAS_FOUND ON)

#    Check include files
IF(NOT OpenBLAS_INCLUDE_DIR)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS include. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT OpenBLAS_LIB)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off")
ENDIF()

IF (OpenBLAS_FOUND)
  IF (NOT OpenBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIB}")
    MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
  ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
  IF (OpenBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find OpenBLAS")
  ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

MARK_AS_ADVANCED(
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIB
    OpenBLAS
)

