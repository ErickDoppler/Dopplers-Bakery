// MP4 (H.264 video + optional AAC audio) written with the encoders that ship with Windows (Media Foundation). No third-party code.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

// frames: RGB8 interleaved, width * height * 3 bytes each; audio: interleaved float [-1, 1] (may be empty)
bool writeMp4(const std::string& utf8Path, const std::vector<const uint8_t*>& frames, int width, int height, int fps,
              const std::vector<float>& audio, int channels, int sampleRate, std::string& err);
// Decodes one video frame of an MP4 (for tests): RGB8 top-down. Also reports the duration and whether there is an audio stream.
bool readMp4Frame(const std::string& utf8Path, int frameIndex, std::vector<uint8_t>& rgb, int& width, int& height, double& seconds, bool& hasAudio, std::string& err);
