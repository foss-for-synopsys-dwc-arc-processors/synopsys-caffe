set(DEPENDENCIES_VERSION 1.1.0)
caffe_option(USE_PREBUILT_DEPENDENCIES "Download and use the prebuilt dependencies" ON IF MSVC)

if(USE_PREBUILT_DEPENDENCIES)
    # Determine the python version
    if(BUILD_python)
        if(NOT PYTHONINTERP_FOUND)
            if(NOT "${python_version}" VERSION_LESS "3.8.0")
                find_package(PythonInterp 3.8)
            elseif(NOT "${python_version}" VERSION_LESS "3.6.0")
                find_package(PythonInterp 3.6)
            elseif(NOT "${python_version}" VERSION_LESS "3.5.0")
                find_package(PythonInterp 3.5)
            else()
                find_package(PythonInterp 2.7)
            endif()
        endif()
        set(_pyver ${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
    else()
        message(STATUS "Building without python. Prebuilt dependencies will default to Python 2.7")
        set(_pyver 27)
    endif()

    # prebuilt packages path
    set(CAFFE_DEPENDENCIES_DIR "C:\\Users\\yche\\Desktop\\Projects\\caffe-builder\\build_v140_x64")

    if(EXISTS ${CAFFE_DEPENDENCIES_DIR}/libraries/caffe-builder-config.cmake)
        include(${CAFFE_DEPENDENCIES_DIR}/libraries/caffe-builder-config.cmake)
    else()
        message(FATAL_ERROR "Something went wrong while dowloading dependencies could not open caffe-builder-config.cmake")
    endif()
endif()

