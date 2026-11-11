#include "util/png.h"
#include <algorithm>
#include <cstring>
#include "util/sys.h"

namespace {
uint32_t crcTable[256];
bool crcInit = false;
uint32_t crc32(uint32_t crc, const uint8_t* p, size_t n) {
    if (!crcInit) {
        for (uint32_t i = 0; i < 256; i++) { uint32_t c = i; for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; crcTable[i] = c; }
        crcInit = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; i++) crc = crcTable[(crc ^ p[i]) & 255] ^ (crc >> 8);
    return ~crc;
}
void be32(std::string& s, uint32_t v) { for (int i = 3; i >= 0; i--) s += (char)((v >> (i * 8)) & 255); }
void chunk(std::string& out, const char* type, const std::string& data) {
    be32(out, (uint32_t)data.size());
    std::string td(type, 4); td += data;
    out += td;
    be32(out, crc32(0, (const uint8_t*)td.data(), td.size()));
}
}  // namespace

bool writePngRgb8(const std::string& path, const uint8_t* rgb, int w, int h, const std::string& key, const std::string& text) {
    std::string out("\x89PNG\r\n\x1a\n", 8);
    std::string ihdr; be32(ihdr, w); be32(ihdr, h); ihdr += (char)8; ihdr += (char)2; ihdr += (char)0; ihdr += (char)0; ihdr += (char)0;
    chunk(out, "IHDR", ihdr);
    if (!key.empty()) chunk(out, "tEXt", key + std::string(1, '\0') + text);

    // raw scanlines (filter 0), wrapped in zlib stored blocks
    const size_t rowBytes = (size_t)w * 3, rawSize = (rowBytes + 1) * h;
    std::string z; z.reserve(rawSize + rawSize / 65535 * 5 + 16);
    z += (char)0x78; z += (char)0x01;
    uint32_t a = 1, b = 0;
    size_t pos = 0;
    std::string block;
    auto rawByte = [&](size_t i) -> uint8_t { size_t r = i / (rowBytes + 1), c = i % (rowBytes + 1); return c == 0 ? 0 : rgb[r * rowBytes + c - 1]; };
    while (pos < rawSize) {
        size_t n = std::min<size_t>(65535, rawSize - pos);
        z += (char)(pos + n >= rawSize ? 1 : 0);
        z += (char)(n & 255); z += (char)(n >> 8); z += (char)(~n & 255); z += (char)((~n >> 8) & 255);
        size_t base = z.size();
        z.resize(base + n);
        for (size_t i = 0; i < n; i++) { uint8_t v = rawByte(pos + i); z[base + i] = (char)v; a = (a + v) % 65521; b = (b + a) % 65521; }
        pos += n;
    }
    be32(z, (b << 16) | a);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", "");
    return sys::writeFile(path, out);
}
