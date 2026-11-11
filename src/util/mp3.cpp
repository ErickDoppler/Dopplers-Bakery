#include "util/mp3.h"
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
std::string hr(const char* what, HRESULT h) { char b[96]; snprintf(b, sizeof b, "%s failed (0x%08lx)", what, (unsigned long)h); return b; }
}

bool writeMp3(const std::string& path, const std::vector<float>& pcmF, int channels, int rate, int kbps, std::string& err) {
    sys::makeDirs(path.substr(0, path.find_last_of("\\/")));
    HRESULT h = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool coInit = SUCCEEDED(h);
    h = MFStartup(MF_VERSION, MFSTARTUP_LITE);
    if (FAILED(h)) { err = hr("MFStartup", h); if (coInit) CoUninitialize(); return false; }
    bool ok = false;
    {
        Rel<IMFSinkWriter> writer;
        std::wstring wp = sys::widen(path);
        h = MFCreateSinkWriterFromURL(wp.c_str(), nullptr, nullptr, &writer);
        if (FAILED(h)) { err = hr("creating the MP3 writer", h); goto done; }

        Rel<IMFMediaType> out;
        MFCreateMediaType(&out);
        out->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
        out->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_MP3);
        out->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels);
        out->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, rate);
        out->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, kbps * 125);
        out->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, 1);
        DWORD stream = 0;
        h = writer->AddStream(out.p, &stream);
        if (FAILED(h)) { err = hr("this Windows has no MP3 encoder for the requested format", h); goto done; }

        Rel<IMFMediaType> in;
        MFCreateMediaType(&in);
        in->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
        in->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM);
        in->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels);
        in->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, rate);
        in->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
        in->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, channels * 2);
        in->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, rate * channels * 2);
        h = writer->SetInputMediaType(stream, in.p, nullptr);
        if (FAILED(h)) { err = hr("MP3 encoder input type", h); goto done; }
        h = writer->BeginWriting();
        if (FAILED(h)) { err = hr("starting the MP3 writer", h); goto done; }

        const size_t frames = pcmF.size() / channels, chunk = (size_t)rate;      // one second per sample
        for (size_t f0 = 0; f0 < frames; f0 += chunk) {
            const size_t n = std::min(chunk, frames - f0), bytes = n * channels * 2;
            Rel<IMFMediaBuffer> buf;
            if (FAILED(MFCreateMemoryBuffer((DWORD)bytes, &buf))) { err = "out of memory"; goto done; }
            BYTE* dst = nullptr;
            buf->Lock(&dst, nullptr, nullptr);
            for (size_t i = 0; i < n * channels; i++) {
                float v = pcmF[f0 * channels + i]; v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
                int16_t s = (int16_t)(v * 32767.0f + (v >= 0 ? 0.5f : -0.5f));
                memcpy(dst + i * 2, &s, 2);
            }
            buf->Unlock();
            buf->SetCurrentLength((DWORD)bytes);
            Rel<IMFSample> smp;
            MFCreateSample(&smp);
            smp->AddBuffer(buf.p);
            smp->SetSampleTime((LONGLONG)((double)f0 / rate * 1e7));
            smp->SetSampleDuration((LONGLONG)((double)n / rate * 1e7));
            h = writer->WriteSample(stream, smp.p);
            if (FAILED(h)) { err = hr("writing audio to the MP3 encoder", h); goto done; }
        }
        h = writer->Finalize();
        if (FAILED(h)) { err = hr("finishing the MP3 file", h); goto done; }
        ok = true;
    }
done:
    MFShutdown();
    if (coInit) CoUninitialize();
    if (!ok) DeleteFileW(sys::widen(path).c_str());          // no half-written file
    return ok;
}
