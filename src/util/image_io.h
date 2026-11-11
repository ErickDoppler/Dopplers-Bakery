// Reading images of any common format (PNG, JPEG, BMP, GIF, TIFF, WebP when Windows has the codec) through WIC.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct ImageRgb { int w = 0, h = 0; std::vector<uint8_t> px; };      // 8-bit RGB, row-major interleaved

bool loadImage(const std::string& utf8Path, ImageRgb& out, std::string& err);

// Bilinear/area resize to exactly w x h (stretches); output floats in [0,1], interleaved RGB.
void resizeToFloat(const ImageRgb& in, int w, int h, std::vector<float>& out);

// Largest size with the aspect ratio of `in` that fits `maxSide`, both multiples of 8.
void fitSize8(int inW, int inH, int maxSide, int& w, int& h);

// Thumbnail as a 32-bit premultiplied-alpha DIB section (HBITMAP, caller deletes it); longest side <= maxSide.
void* loadThumbnail(const std::string& utf8Path, int maxSide, int& outW, int& outH);

// Picture currently on the clipboard (bitmap, or a copied image file) saved as a PNG into `dir`.
// Returns false (err empty) when the clipboard holds no image.
bool clipboardImageToFile(void* hwndOwner, const std::string& dir, std::string& outPath, std::string& err);
