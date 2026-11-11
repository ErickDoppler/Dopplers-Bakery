// Minimal PNG writer (8-bit RGB, stored deflate blocks, optional tEXt chunk). No dependencies.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

bool writePngRgb8(const std::string& utf8Path, const uint8_t* rgb, int width, int height,
                  const std::string& textKeyword = "", const std::string& text = "");
