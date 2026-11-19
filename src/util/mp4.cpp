#include "util/mp4.h"
#include <windows.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "util/sys.h"

namespace {
template <class T> struct Rel { T* p = nullptr; ~Rel() { if (p) p->Release(); } T** operator&() { return &p; } T* operator->() { return p; } operator bool() const { return p != nullptr; } };
std::string hr(const char* what, HRESULT h) { char b[110]; snprintf(b, sizeof b, "%s failed (0x%08lx)", what, (unsigned long)h); return b; }
struct Mf { bool co = false, mf = false; Mf() { co = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED)); mf = SUCCEEDED(MFStartup(MF_VERSION, MFSTARTUP_LITE)); } ~Mf() { if (mf) MFShutdown(); if (co) CoUninitialize(); } };
}

bool writeMp4(const std::string& path, const std::vector<const uint8_t*>& frames, int W, int H, int fps, const std::vector<float>& audio, int channels, int rate, std::string& err) {
    if (frames.empty()) { err = "no frames"; return false; }
    sys::makeDirs(path.substr(0, path.find_last_of("\\/")));
    Mf mf; if (!mf.mf) { err = "Media Foundation is not available"; return false; }
    bool ok = false; HRESULT h;
    Rel<IMFSinkWriter> writer;
    DWORD vs = 0, as = 0; const bool withAudio = !audio.empty();
    h = MFCreateSinkWriterFromURL(sys::widen(path).c_str(), nullptr, nullptr, &writer);
    if (FAILED(h)) { err = hr("creating the MP4 writer", h); return false; }
    {   Rel<IMFMediaType> out; MFCreateMediaType(&out);
        out->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video); out->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
        out->SetUINT32(MF_MT_AVG_BITRATE, (UINT32)std::max(4000000.0, (double)W * H * fps * 0.25));
        out->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        MFSetAttributeSize(out.p, MF_MT_FRAME_SIZE, W, H); MFSetAttributeRatio(out.p, MF_MT_FRAME_RATE, fps, 1); MFSetAttributeRatio(out.p, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
        out->SetUINT32(MF_MT_MPEG2_PROFILE, 100);                                                       // H.264 High
        h = writer->AddStream(out.p, &vs); if (FAILED(h)) { err = hr("adding the H.264 stream (is the H.264 encoder installed?)", h); goto done; }
        Rel<IMFMediaType> in; MFCreateMediaType(&in);
        in->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video); in->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
        in->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        MFSetAttributeSize(in.p, MF_MT_FRAME_SIZE, W, H); MFSetAttributeRatio(in.p, MF_MT_FRAME_RATE, fps, 1); MFSetAttributeRatio(in.p, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
        in->SetUINT32(MF_MT_DEFAULT_STRIDE, (UINT32)(W * 4));                                            // positive stride = top-down rows
        h = writer->SetInputMediaType(vs, in.p, nullptr); if (FAILED(h)) { err = hr("H.264 encoder input type", h); goto done; }
    }
    if (withAudio) {
        Rel<IMFMediaType> out; MFCreateMediaType(&out);
        out->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio); out->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_AAC);
        out->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16); out->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, rate); out->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels);
        out->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 24000);                                         // 192 kbit/s
        out->SetUINT32(MF_MT_AAC_PAYLOAD_TYPE, 0); out->SetUINT32(MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, 0x29);
        h = writer->AddStream(out.p, &as); if (FAILED(h)) { err = hr("adding the AAC stream", h); goto done; }
        Rel<IMFMediaType> in; MFCreateMediaType(&in);
        in->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio); in->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM);
        in->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels); in->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, rate); in->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
        in->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, channels * 2); in->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, rate * channels * 2);
        h = writer->SetInputMediaType(as, in.p, nullptr); if (FAILED(h)) { err = hr("AAC encoder input type", h); goto done; }
    }
    h = writer->BeginWriting(); if (FAILED(h)) { err = hr("starting the MP4 writer", h); goto done; }
    {   const size_t aFrames = withAudio ? audio.size() / channels : 0; size_t aDone = 0;
        auto writeAudio = [&](size_t upto) -> bool {
            while (aDone < upto) {
                const size_t n = std::min<size_t>(upto - aDone, 4096), bytes = n * channels * 2;
                Rel<IMFMediaBuffer> buf; if (FAILED(MFCreateMemoryBuffer((DWORD)bytes, &buf))) return false;
                BYTE* dst = nullptr; buf->Lock(&dst, nullptr, nullptr);
                for (size_t i = 0; i < n * channels; i++) { float v = std::max(-1.f, std::min(1.f, audio[aDone * channels + i])); int16_t s = (int16_t)(v * 32767.0f + (v >= 0 ? 0.5f : -0.5f)); memcpy(dst + i * 2, &s, 2); }
                buf->Unlock(); buf->SetCurrentLength((DWORD)bytes);
                Rel<IMFSample> smp; MFCreateSample(&smp); smp->AddBuffer(buf.p);
                smp->SetSampleTime((LONGLONG)((double)aDone / rate * 1e7)); smp->SetSampleDuration((LONGLONG)((double)n / rate * 1e7));
                if (FAILED(writer->WriteSample(as, smp.p))) return false;
                aDone += n;
            }
            return true;
        };
        const LONGLONG dur = (LONGLONG)(1e7 / fps);
        for (size_t i = 0; i < frames.size(); i++) {
            const size_t bytes = (size_t)W * H * 4;
            Rel<IMFMediaBuffer> buf; if (FAILED(MFCreateMemoryBuffer((DWORD)bytes, &buf))) { err = "out of memory"; goto done; }
            BYTE* dst = nullptr; buf->Lock(&dst, nullptr, nullptr);
            const uint8_t* src = frames[i];
            for (size_t p = 0; p < (size_t)W * H; p++) { dst[p * 4] = src[p * 3 + 2]; dst[p * 4 + 1] = src[p * 3 + 1]; dst[p * 4 + 2] = src[p * 3]; dst[p * 4 + 3] = 255; }      // RGB -> BGRA
            buf->Unlock(); buf->SetCurrentLength((DWORD)bytes);
            Rel<IMFSample> smp; MFCreateSample(&smp); smp->AddBuffer(buf.p);
            smp->SetSampleTime((LONGLONG)i * dur); smp->SetSampleDuration(dur);
            h = writer->WriteSample(vs, smp.p); if (FAILED(h)) { err = hr("writing a video frame", h); goto done; }
            if (withAudio && !writeAudio(std::min(aFrames, (size_t)((double)(i + 1) / fps * rate)))) { err = "writing audio failed"; goto done; }
        }
        if (withAudio && !writeAudio(aFrames)) { err = "writing audio failed"; goto done; }
    }
    h = writer->Finalize(); if (FAILED(h)) { err = hr("finishing the MP4 file", h); goto done; }
    ok = true;
done:
    if (!ok) { writer.p->Release(); writer.p = nullptr; DeleteFileW(sys::widen(path).c_str()); }
    return ok;
}

bool readMp4Frame(const std::string& path, int frameIndex, std::vector<uint8_t>& rgb, int& W, int& H, double& seconds, bool& hasAudio, std::string& err) {
    Mf mf; if (!mf.mf) { err = "Media Foundation is not available"; return false; }
    Rel<IMFAttributes> at; MFCreateAttributes(&at, 1); at->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
    Rel<IMFSourceReader> rd; HRESULT h = MFCreateSourceReaderFromURL(sys::widen(path).c_str(), at.p, &rd);
    if (FAILED(h)) { err = hr("opening the MP4", h); return false; }
    Rel<IMFMediaType> want; MFCreateMediaType(&want); want->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video); want->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    h = rd->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, want.p); if (FAILED(h)) { err = hr("decoder output type", h); return false; }
    Rel<IMFMediaType> cur; rd->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &cur);
    UINT32 w = 0, hh = 0; MFGetAttributeSize(cur.p, MF_MT_FRAME_SIZE, &w, &hh); W = (int)w; H = (int)hh;
    const LONG stride = (LONG)MFGetAttributeUINT32(cur.p, MF_MT_DEFAULT_STRIDE, (UINT32)(W * 4)); const bool bottomUp = stride < 0;
    PROPVARIANT pv; PropVariantInit(&pv);
    seconds = SUCCEEDED(rd->GetPresentationAttribute((DWORD)MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &pv)) ? (double)pv.uhVal.QuadPart / 1e7 : 0; PropVariantClear(&pv);
    Rel<IMFMediaType> an; hasAudio = SUCCEEDED(rd->GetNativeMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, &an));
    for (int i = 0;; i++) {
        DWORD idx = 0, flags = 0; LONGLONG ts = 0; Rel<IMFSample> smp;
        h = rd->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &idx, &flags, &ts, &smp);
        if (FAILED(h) || (flags & MF_SOURCE_READERF_ENDOFSTREAM)) { err = "the MP4 has fewer frames than requested"; return false; }
        if (!smp || i < frameIndex) continue;
        Rel<IMFMediaBuffer> buf; smp->ConvertToContiguousBuffer(&buf); BYTE* p = nullptr; DWORD len = 0; buf->Lock(&p, nullptr, &len);
        rgb.resize((size_t)W * H * 3);
        for (int y = 0; y < H; y++) { const BYTE* row = p + (size_t)(bottomUp ? H - 1 - y : y) * W * 4; for (int x = 0; x < W; x++) { rgb[((size_t)y * W + x) * 3] = row[x * 4 + 2]; rgb[((size_t)y * W + x) * 3 + 1] = row[x * 4 + 1]; rgb[((size_t)y * W + x) * 3 + 2] = row[x * 4]; } }
        buf->Unlock();
        return true;
    }
}

bool readMp4All(const std::string& path, std::vector<std::vector<uint8_t>>& frames, int& W, int& H, double& fps, std::vector<float>& audio, std::string& err) {
    Mf mf; if (!mf.mf) { err = "Media Foundation is not available"; return false; }
    Rel<IMFAttributes> at; MFCreateAttributes(&at, 1); at->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
    Rel<IMFSourceReader> rd; HRESULT h = MFCreateSourceReaderFromURL(sys::widen(path).c_str(), at.p, &rd);
    if (FAILED(h)) { err = hr("opening the MP4", h); return false; }
    Rel<IMFMediaType> want; MFCreateMediaType(&want); want->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video); want->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    h = rd->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, want.p); if (FAILED(h)) { err = hr("decoder output type", h); return false; }
    Rel<IMFMediaType> cur; rd->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &cur);
    UINT32 w = 0, hh = 0, fn = 0, fd = 1; MFGetAttributeSize(cur.p, MF_MT_FRAME_SIZE, &w, &hh); W = (int)w; H = (int)hh;
    MFGetAttributeRatio(cur.p, MF_MT_FRAME_RATE, &fn, &fd); fps = fd ? (double)fn / fd : 24.0;
    const LONG stride = (LONG)MFGetAttributeUINT32(cur.p, MF_MT_DEFAULT_STRIDE, (UINT32)(W * 4)); const bool bottomUp = stride < 0;
    // the sound track as 48 kHz stereo float, when there is one
    Rel<IMFMediaType> an; bool hasAudio = SUCCEEDED(rd->GetNativeMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, &an));
    if (hasAudio) {
        Rel<IMFMediaType> wa; MFCreateMediaType(&wa); wa->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio); wa->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float);
        wa->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, 2); wa->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, 48000); wa->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32);
        wa->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, 8); wa->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 48000 * 8);
        if (FAILED(rd->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, wa.p))) hasAudio = false;
    }
    if (!hasAudio) rd->SetStreamSelection((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, FALSE);
    audio.clear(); frames.clear();
    bool videoDone = false, audioDone = !hasAudio;
    while (!videoDone || !audioDone) {
        const DWORD stream = !videoDone ? (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM : (DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM;
        DWORD idx = 0, flags = 0; LONGLONG ts = 0; Rel<IMFSample> smp;
        h = rd->ReadSample(stream, 0, &idx, &flags, &ts, &smp);
        if (FAILED(h)) { err = hr("reading the MP4", h); return false; }
        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) { if (!videoDone) videoDone = true; else audioDone = true; continue; }
        if (!smp) continue;
        Rel<IMFMediaBuffer> buf; smp->ConvertToContiguousBuffer(&buf); BYTE* p = nullptr; DWORD len = 0; buf->Lock(&p, nullptr, &len);
        if (!videoDone) {
            std::vector<uint8_t> rgb((size_t)W * H * 3);
            for (int y = 0; y < H; y++) { const BYTE* row = p + (size_t)(bottomUp ? H - 1 - y : y) * W * 4; for (int x = 0; x < W; x++) { rgb[((size_t)y * W + x) * 3] = row[x * 4 + 2]; rgb[((size_t)y * W + x) * 3 + 1] = row[x * 4 + 1]; rgb[((size_t)y * W + x) * 3 + 2] = row[x * 4]; } }
            frames.push_back(std::move(rgb));
        } else {
            const size_t n = len / 4; const size_t at0 = audio.size(); audio.resize(at0 + n); memcpy(&audio[at0], p, n * 4);
        }
        buf->Unlock();
    }
    if (frames.empty()) { err = "the MP4 has no video frames"; return false; }
    return true;
}
