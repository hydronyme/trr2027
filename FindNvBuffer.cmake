# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# FindNvBuffer.cmake
# Finds NvBufSurface library and creates nvbuffer::surface imported target
#
# This module is used for standalone sample builds on Tegra platforms.
# When building as part of the VPI release, the main cmake/FindNvBuffer.cmake
# is used instead (which relies on DefPackage.cmake).

# If the target already exists (e.g., from VPI build), skip everything
if(TARGET nvbuffer::surface)
    set(NvBuffer_FOUND TRUE)
    return()
endif()

include(FindPackageHandleStandardArgs)

# Find the include directory containing nvbufsurface.h
find_path(NvBuffer_INCLUDE_DIR
    NAMES nvbufsurface.h
    PATHS
        /usr/src/jetson_multimedia_api/include
        /usr/local/include
        /usr/include
    DOC "NvBufSurface include directory"
)

# Find the nvbufsurface library
find_library(NvBuffer_LIBRARY
    NAMES nvbufsurface
    PATHS
        /usr/src/jetson_multimedia_api/
        /usr/lib/aarch64-linux-gnu/nvidia
        /usr/lib/aarch64-linux-gnu
        /usr/local/lib
        /usr/lib
    DOC "NvBufSurface library"
)

# Handle standard find_package arguments
find_package_handle_standard_args(NvBuffer
    REQUIRED_VARS
        NvBuffer_LIBRARY
        NvBuffer_INCLUDE_DIR
)

# Create imported target if found
if(NvBuffer_FOUND AND NOT TARGET nvbuffer::surface)
    add_library(nvbuffer::surface SHARED IMPORTED)
    set_target_properties(nvbuffer::surface PROPERTIES
        IMPORTED_LOCATION "${NvBuffer_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NvBuffer_INCLUDE_DIR}"
    )
endif()

# Mark as advanced
mark_as_advanced(NvBuffer_INCLUDE_DIR NvBuffer_LIBRARY)

