
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "load_image.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

std::vector<ov::Tensor> utils::load_images(const std::filesystem::path& input_path) {
    if (input_path.empty() || !fs::exists(input_path)) {
        throw std::runtime_error{"Path to images is empty or does not exist."};
    }
    if (fs::is_directory(input_path)) {
        std::set<fs::path> sorted_images{fs::directory_iterator(input_path), fs::directory_iterator()};
        std::vector<ov::Tensor> images;
        for (const fs::path& dir_entry : sorted_images) {
            images.push_back(utils::load_image(dir_entry));
        }
        return images;
    }
    return {utils::load_image(input_path)};
}

ov::Tensor utils::load_video(const std::filesystem::path& input_path) {
    auto rgbs = load_images(input_path);
    if (rgbs.size() == 0) {
        return {};
    }

    auto video = ov::Tensor(ov::element::u8,
                            ov::Shape{rgbs.size(), rgbs[0].get_shape()[1], rgbs[0].get_shape()[2], rgbs[0].get_shape()[3]});
    std::cout << "video.shape = " << video.get_shape() << std::endl;

    auto stride = rgbs[0].get_byte_size();
    std::cout << "stride = " << stride << std::endl;
    auto dst = reinterpret_cast<char*>(video.data());
    int b = 0;
    for (auto rgb : rgbs)
    {
        std::memcpy(dst + stride * b, rgb.data(), stride);
        b++;
    }
    return video;
}

// Return video with shape: [num_frames, height, width, 3]
ov::Tensor utils::create_countdown_frames()
{
    int frames_count = 5, height = 240, width = 360;
    auto video = ov::Tensor(ov::element::u8,
                            ov::Shape{(size_t)frames_count, (size_t)height, (size_t)width, 3});

    // Fill each frame with a distinct grayscale intensity to preserve deterministic
    // per-frame content without relying on external image processing libraries.
    auto* dst = reinterpret_cast<uint8_t*>(video.data());
    const size_t frame_size = static_cast<size_t>(height) * static_cast<size_t>(width) * 3;
    for (int i = frames_count; i > 0; i--) {
        const int idx = frames_count - i;
        const uint8_t value = static_cast<uint8_t>(40 * i);
        std::memset(dst + static_cast<size_t>(idx) * frame_size, value, frame_size);
    }
    return video;
}

ov::Tensor utils::load_image(const std::filesystem::path &image_path)
{
    int x = 0, y = 0, channels_in_file = 0;
    constexpr int desired_channels = 3;
    unsigned char *image = stbi_load(
        image_path.string().c_str(),
        &x, &y, &channels_in_file, desired_channels);
    if (!image)
    {
        std::stringstream error_message;
        error_message << "Failed to load the image '" << image_path << "'";
        throw std::runtime_error{error_message.str()};
    }
    struct SharedImageAllocator
    {
        unsigned char *image;
        int channels, height, width;
        void *allocate(size_t bytes, size_t) const
        {
            if (image && channels * height * width == bytes)
            {
                return image;
            }
            throw std::runtime_error{"Unexpected number of bytes was requested to allocate."};
        }
        void deallocate(void *, size_t, size_t) noexcept
        {
            stbi_image_free(image);
            image = nullptr;
        }
        bool is_equal(const SharedImageAllocator &other) const noexcept { return this == &other; }
    };
    return ov::Tensor(
        ov::element::u8,
        ov::Shape{1, size_t(y), size_t(x), size_t(desired_channels)},
        SharedImageAllocator{image, desired_channels, y, x});
}

namespace TEST_DATA {

std::string img_cat_120_100() {
    std::string full_path = get_data_path() + "/cat_120_100.png";
    OPENVINO_ASSERT(check_file_exists(full_path), "File does not exist: " + full_path);
    return full_path;
}

std::string img_dog_120_120() {
    std::string full_path = get_data_path() + "/dog_120_120.png";
    OPENVINO_ASSERT(check_file_exists(full_path), "File does not exist: " + full_path);
    return full_path;
}
}  // namespace TEST_DATA