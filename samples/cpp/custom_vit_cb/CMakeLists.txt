# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alongside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

add_executable(custom_vit_cb custom_vit_cb.cpp)

target_link_libraries(custom_vit_cb PRIVATE openvino::genai)

set_target_properties(custom_vit_cb PROPERTIES
    # Ensure out-of-box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS custom_vit_cb
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
