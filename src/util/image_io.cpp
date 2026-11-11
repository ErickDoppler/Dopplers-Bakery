#include "util/image_io.h"
#include <windows.h>
#include <wincodec.h>
#include <shellapi.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include "util/png.h"
#include "util/sys.h"

namespace {
template <class T> struct Rel { T* p = nullptr; ~Rel() { if (p) p->Release(); } T** operator&() { return &p; } T* operator->() { return p; } };

struct ComScope {
    bool init;
    ComScope() { init = SUCCEEDED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED)); }
    ~ComScope() { if (init) CoUninitialize(); }
};

bool factory(Rel<IWICImagingFactory>& f, std::string& err) {
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f)))) { err = "Windows imaging component is not available"; return false; }
    return true;
}

// any WIC source -> 24-bit RGB (alpha composited on white)
bool toRgb(IWICImagingFactory* f, IWICBitmapSource* src, ImageRgb& out, std::string& err) {
    UINT w = 0, h = 0;
    src->GetSize(&w, &h);
    if (!w || !h) { err = "empty image"; return false; }
    Rel<IWICFormatConverter> conv;
    if (FAILED(f->CreateFormatConverter(&conv)) || FAILED(conv->Initialize(src, GUID_WICPixelFormat32bppBGRA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom))) { err = "cannot convert the image"; return false; }
    std::vector<uint8_t> bgra((size_t)w * h * 4);
    if (FAILED(conv->CopyPixels(nullptr, w * 4, (UINT)bgra.size(), bgra.data()))) { err = "cannot read the pixels"; return false; }
    out.w = (int)w; out.h = (int)h; out.px.resize((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; i++) {
        float a = bgra[i * 4 + 3] / 255.0f;
        for (int c = 0; c < 3; c++) out.px[i * 3 + c] = (uint8_t)(bgra[i * 4 + (2 - c)] * a + 255.0f * (1 - a) + 0.5f);
    }
    return true;
}
}  // namespace

bool loadImage(const std::string& path, ImageRgb& out, std::string& err) {
    ComScope com;
    Rel<IWICImagingFactory> f;
    if (!factory(f, err)) return false;
    Rel<IWICBitmapDecoder> dec;
    if (FAILED(f->CreateDecoderFromFilename(sys::widen(path).c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec))) { err = "cannot open image: " + path; return false; }
    Rel<IWICBitmapFrameDecode> frame;
    if (FAILED(dec->GetFrame(0, &frame))) { err = "cannot decode image: " + path; return false; }
    return toRgb(f.p, frame.p, out, err);
}

void* loadThumbnail(const std::string& path, int maxSide, int& outW, int& outH) {
    ComScope com;
    std::string err;
    Rel<IWICImagingFactory> f;
    if (!factory(f, err)) return nullptr;
    Rel<IWICBitmapDecoder> dec;
    if (FAILED(f->CreateDecoderFromFilename(sys::widen(path).c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec))) return nullptr;
    Rel<IWICBitmapFrameDecode> frame;
    if (FAILED(dec->GetFrame(0, &frame))) return nullptr;
    UINT w = 0, h = 0; frame->GetSize(&w, &h);
    if (!w || !h) return nullptr;
    double s = std::min(1.0, (double)maxSide / std::max(w, h));
    UINT tw = std::max<UINT>(1, (UINT)std::lround(w * s)), th = std::max<UINT>(1, (UINT)std::lround(h * s));
    Rel<IWICBitmapScaler> scaler;
    if (FAILED(f->CreateBitmapScaler(&scaler)) || FAILED(scaler->Initialize(frame.p, tw, th, WICBitmapInterpolationModeFant))) return nullptr;
    Rel<IWICFormatConverter> conv;
    if (FAILED(f->CreateFormatConverter(&conv)) || FAILED(conv->Initialize(scaler.p, GUID_WICPixelFormat32bppPBGRA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom))) return nullptr;
    BITMAPINFO bi = {};
    bi.bmiHeader.biSize = sizeof bi.bmiHeader; bi.bmiHeader.biWidth = (LONG)tw; bi.bmiHeader.biHeight = -(LONG)th;
    bi.bmiHeader.biPlanes = 1; bi.bmiHeader.biBitCount = 32; bi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP hb = CreateDIBSection(nullptr, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!hb) return nullptr;
    if (FAILED(conv->CopyPixels(nullptr, tw * 4, tw * th * 4, (BYTE*)bits))) { DeleteObject(hb); return nullptr; }
    outW = (int)tw; outH = (int)th;
    return hb;
}

void resizeToFloat(const ImageRgb& in, int w, int h, std::vector<float>& out) {
    out.assign((size_t)w * h * 3, 0.0f);
    const double sx = (double)in.w / w, sy = (double)in.h / h;
    for (int y = 0; y < h; y++) {
        // box filter over the source footprint (area average when shrinking, bilinear-like when enlarging)
        double y0 = y * sy, y1 = (y + 1) * sy;
        if (y1 - y0 < 1.0) { double m = (y0 + y1) / 2; y0 = m - 0.5; y1 = m + 0.5; }
        for (int x = 0; x < w; x++) {
            double x0 = x * sx, x1 = (x + 1) * sx;
            if (x1 - x0 < 1.0) { double m = (x0 + x1) / 2; x0 = m - 0.5; x1 = m + 0.5; }
            double acc[3] = {0, 0, 0}, wsum = 0;
            for (int yy = (int)std::floor(y0); yy < (int)std::ceil(y1); yy++) {
                double wy = std::min<double>(yy + 1, y1) - std::max<double>(yy, y0);
                int cy = std::min(std::max(yy, 0), in.h - 1);
                for (int xx = (int)std::floor(x0); xx < (int)std::ceil(x1); xx++) {
                    double wx = std::min<double>(xx + 1, x1) - std::max<double>(xx, x0);
                    int cx = std::min(std::max(xx, 0), in.w - 1);
                    double wgt = wx * wy; const uint8_t* p = &in.px[((size_t)cy * in.w + cx) * 3];
                    acc[0] += p[0] * wgt; acc[1] += p[1] * wgt; acc[2] += p[2] * wgt; wsum += wgt;
                }
            }
            for (int c = 0; c < 3; c++) out[((size_t)y * w + x) * 3 + c] = (float)(acc[c] / wsum / 255.0);
        }
    }
}

void fitSize8(int inW, int inH, int maxSide, int& w, int& h) {
    double s = std::min(1.0, (double)maxSide / std::max(inW, inH));
    w = std::max(64, (int)std::lround(inW * s / 8.0) * 8);
    h = std::max(64, (int)std::lround(inH * s / 8.0) * 8);
}

bool clipboardImageToFile(void* owner, const std::string& dir, std::string& outPath, std::string& err) {
    err.clear();
    ComScope com;
    if (!OpenClipboard((HWND)owner)) { err = "cannot open the clipboard"; return false; }
    bool ok = false;
    // a file copied in Explorer
    if (HANDLE hd = GetClipboardData(CF_HDROP)) {
        wchar_t buf[MAX_PATH * 2] = {};
        if (DragQueryFileW((HDROP)hd, 0, buf, MAX_PATH * 2)) {
            std::string p = sys::narrow(buf), l = p; for (auto& c : l) c = (char)tolower((unsigned char)c);
            for (const char* e : {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"})
                if (l.size() > strlen(e) && l.compare(l.size() - strlen(e), strlen(e), e) == 0) { outPath = p; ok = true; }
        }
    }
    if (!ok && IsClipboardFormatAvailable(CF_BITMAP)) {
        HBITMAP hb = (HBITMAP)GetClipboardData(CF_BITMAP);
        Rel<IWICImagingFactory> f;
        Rel<IWICBitmap> bmp;
        ImageRgb img;
        if (hb && factory(f, err) && SUCCEEDED(f->CreateBitmapFromHBITMAP(hb, nullptr, WICBitmapUseAlpha, &bmp)) && toRgb(f.p, bmp.p, img, err)) {
            sys::makeDirs(dir);
            char name[64]; time_t t = time(nullptr); tm tmv; localtime_s(&tmv, &t); strftime(name, sizeof name, "pasted-%Y%m%d-%H%M%S.png", &tmv);
            outPath = sys::join(dir, name);
            ok = writePngRgb8(outPath, img.px.data(), img.w, img.h);
            if (!ok) err = "cannot save the pasted image";
        }
    }
    CloseClipboard();
    return ok;
}
