cmake_minimum_required(VERSION 3.22)

# Add current directory to module path for FindNvBuffer.cmake (standalone builds)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# Allow caller/toolchain to decide CUDA arch; provide a sane default if unset
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

project(camera_analysis LANGUAGES CXX CUDA)

find_package(vpi 4.1 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(NvBuffer REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED IMPORTED_TARGET
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
)

add_executable(${PROJECT_NAME}
    main.cpp
    line_analysis.cu
)

target_include_directories(camera_analysis PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${VPI_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GST_APP_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    vpi
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_videoio
    CUDA::cudart
    CUDA::nppif
    nvbuffer::surface
    ${CUDA_LIBRARIES}
    ${VPI_LIBRARIES}

    PkgConfig::GST
)
