// MP3 export through the encoder that ships with Windows (Media Foundation). No third-party code.
#pragma once
#include <string>
#include <vector>

// interleaved float samples in [-1, 1]; kbps: constant bit rate (128, 160, 192, 224, 256 or 320 for 44.1 kHz stereo)
bool writeMp3(const std::string& utf8Path, const std::vector<float>& interleaved, int channels, int sampleRate, int kbps, std::string& err);
