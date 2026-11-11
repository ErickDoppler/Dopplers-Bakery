#include "audio/spiece.h"
#include <cstdio>
#include <cstring>

namespace {
struct Reader {
    const unsigned char* p; const unsigned char* e;
    bool more() const { return p < e; }
    bool varint(unsigned long long& v) {
        v = 0; int shift = 0;
        while (p < e) { unsigned char b = *p++; v |= (unsigned long long)(b & 0x7f) << shift; if (!(b & 0x80)) return true; shift += 7; if (shift > 63) return false; }
        return false;
    }
    // reads a field header and, for length-delimited fields, the payload
    bool field(int& num, int& wire, unsigned long long& val, const unsigned char*& data, size_t& len) {
        unsigned long long k; if (!varint(k)) return false;
        num = (int)(k >> 3); wire = (int)(k & 7); data = nullptr; len = 0; val = 0;
        switch (wire) {
        case 0: return varint(val);
        case 1: if (e - p < 8) return false; memcpy(&val, p, 8); p += 8; return true;
        case 2: { unsigned long long l; if (!varint(l) || (unsigned long long)(e - p) < l) return false; data = p; len = (size_t)l; p += l; return true; }
        case 5: { if (e - p < 4) return false; unsigned int v32; memcpy(&v32, p, 4); val = v32; p += 4; return true; }
        default: return false;
        }
    }
};
}  // namespace

bool SentencePiece::load(const std::string& proto, std::string& err) {
    pieces_.clear(); index_.clear();
    Reader r{(const unsigned char*)proto.data(), (const unsigned char*)proto.data() + proto.size()};
    int num, wire; unsigned long long val; const unsigned char* d; size_t n;
    while (r.more()) {
        if (!r.field(num, wire, val, d, n)) { err = "corrupt sentencepiece model"; return false; }
        if (num == 1 && wire == 2) {
            Piece pc{"", 0.0f, 1};
            Reader pr{d, d + n};
            int pn, pw; unsigned long long pv; const unsigned char* pd; size_t pl;
            while (pr.more()) {
                if (!pr.field(pn, pw, pv, pd, pl)) break;
                if (pn == 1 && pw == 2) pc.s.assign((const char*)pd, pl);
                else if (pn == 2 && pw == 5) { unsigned int b = (unsigned int)pv; memcpy(&pc.score, &b, 4); }
                else if (pn == 3 && pw == 0) pc.type = (int)pv;
            }
            pieces_.push_back(pc);
        } else if (num == 2 && wire == 2) {            // TrainerSpec
            Reader tr{d, d + n};
            int tn, tw; unsigned long long tv; const unsigned char* td; size_t tl;
            while (tr.more()) {
                if (!tr.field(tn, tw, tv, td, tl)) break;
                if (tn == 3 && tw == 0) modelType = (int)tv;
                if (tn == 35 && tw == 0) byteFallback = tv != 0;
            }
        } else if (num == 3 && wire == 2) {            // NormalizerSpec
            Reader nr{d, d + n};
            int nn, nw; unsigned long long nv; const unsigned char* nd; size_t nl;
            while (nr.more()) {
                if (!nr.field(nn, nw, nv, nd, nl)) break;
                if (nn == 2 && nw == 2) charsmapBytes = nl;
                if (nn == 3 && nw == 0) addDummyPrefix = nv != 0;
                if (nn == 4 && nw == 0) removeExtraWhitespaces = nv != 0;
                if (nn == 5 && nw == 0) escapeWhitespaces = nv != 0;
            }
        }
    }
    if (pieces_.empty()) { err = "sentencepiece model has no pieces"; return false; }
    for (int i = 0; i < 256; i++) byteId_[i] = -1;
    for (size_t i = 0; i < pieces_.size(); i++) {
        const Piece& p = pieces_[i];
        if (p.type == 2) unk_ = (int)i;
        if (p.type == 6 || p.type == 5) {              // byte pieces look like <0xNN>
            unsigned int b;
            if (p.s.size() == 6 && p.s[0] == '<' && p.s[1] == '0' && p.s[2] == 'x' && sscanf(p.s.c_str() + 3, "%2x", &b) == 1) byteId_[b] = (int)i;
        }
        if (p.type == 1 || p.type == 4 || p.type == 5 || p.type == 6) index_.emplace(p.s, (int)i);
    }
    return true;
}

std::vector<int> SentencePiece::encode(const std::string& text) const {
    // normalization: escape spaces as U+2581, optional dummy prefix (collapsing whitespace is not applied: identity normalizer)
    std::string s;
    if (addDummyPrefix) s = "\xE2\x96\x81";
    for (char c : text) { if (c == ' ' && escapeWhitespaces) s += "\xE2\x96\x81"; else s += c; }

    // split into segments that start at each U+2581
    std::vector<std::string> segs;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = i + 1;
        while (j < s.size() && !(s.compare(j, 3, "\xE2\x96\x81") == 0)) j++;
        segs.push_back(s.substr(i, j - i)); i = j;
    }
    std::vector<int> out;
    for (auto& seg : segs) {
        std::vector<std::string> sym;
        for (size_t k = 0; k < seg.size();) {
            size_t l = 1; unsigned char c = seg[k];
            if (c >= 0xF0) l = 4; else if (c >= 0xE0) l = 3; else if (c >= 0xC0) l = 2;
            if (k + l > seg.size()) l = seg.size() - k;
            sym.push_back(seg.substr(k, l)); k += l;
        }
        while (sym.size() > 1) {
            float best = -1e30f; int bi = -1;
            for (size_t k = 0; k + 1 < sym.size(); k++) {
                auto it = index_.find(sym[k] + sym[k + 1]);
                if (it == index_.end()) continue;
                float sc = pieces_[it->second].score;
                if (sc > best) { best = sc; bi = (int)k; }
            }
            if (bi < 0) break;
            sym[bi] += sym[bi + 1]; sym.erase(sym.begin() + bi + 1);
        }
        for (auto& t : sym) {
            auto it = index_.find(t);
            if (it != index_.end()) { out.push_back(it->second); continue; }
            for (unsigned char b : t) out.push_back(byteId_[b] >= 0 ? byteId_[b] : unk_);   // byte fallback
        }
    }
    return out;
}
