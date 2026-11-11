#include "krea2/bpe.h"
#include <windows.h>
#include <algorithm>
#include <climits>
#include "util/json.h"
#include "util/sys.h"

std::string krea2Template(const std::string& text) {
    return "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
           "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n" + text +
           "<|im_end|>\n<|im_start|>assistant\n";
}

static bool resource(int id, std::string& out) {
    HRSRC r = FindResourceW(nullptr, MAKEINTRESOURCEW(id), RT_RCDATA);
    if (!r) return false;
    HGLOBAL g = LoadResource(nullptr, r);
    out.assign((const char*)LockResource(g), SizeofResource(nullptr, r));
    return true;
}

static void appendUtf8(std::string& o, unsigned cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 63)); }
    else { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
}

bool QwenTokenizer::load(const std::string& folder, std::string& err) {
    std::string vocab, merges, cfg;
    if (!folder.empty()) {
        if (!sys::readFile(sys::join(folder, "vocab.json"), vocab) || !sys::readFile(sys::join(folder, "merges.txt"), merges)) { err = "tokenizer files not found in " + folder; return false; }
        sys::readFile(sys::join(folder, "tokenizer_config.json"), cfg);
    } else if (!resource(101, vocab) || !resource(102, merges)) { err = "tokenizer resources missing from the exe"; return false; }
    else resource(103, cfg);

    bool ok; Json v = Json::parse(vocab, &ok);
    if (!ok) { err = "bad vocab.json"; return false; }
    vocab_.reserve(v.o.size() * 2);
    for (auto& kv : v.o) vocab_[kv.first] = (int)kv.second.num();

    size_t p = 0; int rank = 0; bool first = true;
    while (p < merges.size()) {
        size_t e = merges.find('\n', p); if (e == std::string::npos) e = merges.size();
        std::string line = merges.substr(p, e - p); p = e + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (first) { first = false; if (line.rfind("#version", 0) == 0) continue; }
        if (line.empty()) continue;
        rank_[line] = rank++;
    }
    if (!cfg.empty()) {
        Json c = Json::parse(cfg, &ok);
        if (ok) for (auto& kv : c.get("added_tokens_decoder").o) added_.push_back({kv.second.str("content"), atoi(kv.first.c_str())});
        std::sort(added_.begin(), added_.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
    }
    // GPT-2 byte <-> unicode table
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; i++) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);
    std::vector<int> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; b++) if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
    for (size_t i = 0; i < bs.size(); i++) { std::string s; appendUtf8(s, cs[i]); b2u_[bs[i]] = s; }
    return true;
}

std::vector<int> QwenTokenizer::bpe(const std::string& mapped) const {
    // split into utf8 characters
    std::vector<std::string> parts;
    for (size_t i = 0; i < mapped.size();) {
        size_t l = 1; unsigned char c = mapped[i];
        if (c >= 0xF0) l = 4; else if (c >= 0xE0) l = 3; else if (c >= 0xC0) l = 2;
        parts.push_back(mapped.substr(i, l)); i += l;
    }
    while (parts.size() > 1) {
        int best = INT_MAX; size_t bi = 0;
        for (size_t i = 0; i + 1 < parts.size(); i++) {
            auto it = rank_.find(parts[i] + " " + parts[i + 1]);
            if (it != rank_.end() && it->second < best) { best = it->second; bi = i; }
        }
        if (best == INT_MAX) break;
        parts[bi] += parts[bi + 1];
        parts.erase(parts.begin() + bi + 1);
    }
    std::vector<int> ids;
    for (auto& s : parts) { auto it = vocab_.find(s); if (it != vocab_.end()) ids.push_back(it->second); }
    return ids;
}

// ---- pre-tokenizer: the Qwen2 regex, implemented by hand over code points
namespace {
enum Cls { L, N, S, NL, O };      // letter, number, space (non-newline), newline, other
struct CP { unsigned cp; size_t pos, len; };

Cls classify(unsigned cp) {
    if (cp == '\n' || cp == '\r') return NL;
    if (cp < 128) {
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return L;
        if (cp >= '0' && cp <= '9') return N;
        if (cp == ' ' || cp == '\t' || cp == '\f' || cp == '\v') return S;
        return O;
    }
    wchar_t w[3]; int n = 0;
    if (cp >= 0x10000) { unsigned c = cp - 0x10000; w[0] = (wchar_t)(0xD800 + (c >> 10)); w[1] = (wchar_t)(0xDC00 + (c & 1023)); n = 2; }
    else { w[0] = (wchar_t)cp; n = 1; }
    WORD t[2] = {0, 0};
    if (n == 1) GetStringTypeW(CT_CTYPE1, w, 1, t);
    else { if (cp >= 0x1F000 && cp < 0x20000) return O; return L; }   // astral: emoji = other, rest = letters (CJK ext)
    if (t[0] & C1_DIGIT) return N;
    if (cp == 0x85 || cp == 0x2028 || cp == 0x2029) return NL;
    if (t[0] & C1_SPACE) return S;
    if (t[0] & C1_ALPHA) return L;
    if (cp >= 0x2000 && cp <= 0x206F) return O;               // general punctuation
    if (cp >= 0x3000 && cp <= 0x303F) return O;               // CJK punctuation
    if (cp >= 0xFF00 && cp <= 0xFF0F) return O;
    if (cp >= 0x2190 && cp < 0x2C00) return O;                // arrows, symbols, dingbats
    if (cp >= 0x4E00 && cp <= 0x9FFF) return L;
    if (t[0] & C1_PUNCT) return O;
    if (t[0] == 0) return L;
    return O;
}
}  // namespace

std::vector<int> QwenTokenizer::encodeChunk(const std::string& text) const {
    std::vector<CP> cps;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = text[i]; size_t l = 1; unsigned cp = c;
        if (c >= 0xF0 && i + 3 < text.size()) { l = 4; cp = ((c & 7) << 18) | ((text[i + 1] & 63) << 12) | ((text[i + 2] & 63) << 6) | (text[i + 3] & 63); }
        else if (c >= 0xE0 && i + 2 < text.size()) { l = 3; cp = ((c & 15) << 12) | ((text[i + 1] & 63) << 6) | (text[i + 2] & 63); }
        else if (c >= 0xC0 && i + 1 < text.size()) { l = 2; cp = ((c & 31) << 6) | (text[i + 1] & 63); }
        cps.push_back({cp, i, l}); i += l;
    }
    const size_t n = cps.size();
    std::vector<Cls> cl(n);
    for (size_t i = 0; i < n; i++) cl[i] = classify(cps[i].cp);
    auto lower = [&](size_t i) { unsigned c = cps[i].cp; return c < 128 ? (unsigned)tolower((int)c) : c; };

    std::vector<std::pair<size_t, size_t>> pieces;   // [start, end) in cps
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        // 1. contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cps[i].cp == '\'' && i + 1 < n) {
            unsigned a = lower(i + 1);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') j = i + 2;
            else if (i + 2 < n) {
                unsigned b = lower(i + 2);
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) j = i + 3;
            }
            if (j > i) { pieces.push_back({i, j}); i = j; continue; }
        }
        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        {
            size_t k = i;
            if (cl[k] != NL && cl[k] != L && cl[k] != N && k + 1 < n && cl[k + 1] == L) k++;
            if (cl[k] == L) { while (k < n && cl[k] == L) k++; pieces.push_back({i, k}); i = k; continue; }
        }
        // 3. \p{N}
        if (cl[i] == N) { pieces.push_back({i, i + 1}); i++; continue; }
        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t k = i;
            if (cps[k].cp == ' ' && k + 1 < n && cl[k + 1] == O) k++;
            if (cl[k] == O) {
                while (k < n && cl[k] == O) k++;
                while (k < n && cl[k] == NL) k++;
                pieces.push_back({i, k}); i = k; continue;
            }
        }
        // 5. \s*[\r\n]+
        {
            size_t k = i;
            while (k < n && (cl[k] == S || cl[k] == NL)) k++;
            size_t last = i; bool nl = false;
            for (size_t q = i; q < k; q++) if (cl[q] == NL) { last = q + 1; nl = true; }
            if (nl) { pieces.push_back({i, last}); i = last; continue; }
        }
        // 6. \s+(?!\S)   7. \s+
        {
            size_t k = i;
            while (k < n && (cl[k] == S || cl[k] == NL)) k++;
            if (k < n && k - i > 1) k--;          // leave the last space to attach to the next word
            if (k == i) k = i + 1;
            pieces.push_back({i, k}); i = k; continue;
        }
    }
    std::vector<int> ids;
    for (auto& pc : pieces) {
        std::string mapped;
        size_t b = cps[pc.first].pos, e = pc.second < n ? cps[pc.second].pos : text.size();
        for (size_t q = b; q < e; q++) mapped += b2u_[(unsigned char)text[q]];
        auto t = bpe(mapped);
        ids.insert(ids.end(), t.begin(), t.end());
    }
    return ids;
}

std::vector<int> QwenTokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    size_t pos = 0, chunkStart = 0;
    while (pos < text.size()) {
        bool hit = false;
        if (text[pos] == '<') for (auto& a : added_) {
            if (text.compare(pos, a.first.size(), a.first) == 0) {
                if (pos > chunkStart) { auto t = encodeChunk(text.substr(chunkStart, pos - chunkStart)); out.insert(out.end(), t.begin(), t.end()); }
                out.push_back(a.second); pos += a.first.size(); chunkStart = pos; hit = true; break;
            }
        }
        if (!hit) pos++;
    }
    if (chunkStart < text.size()) { auto t = encodeChunk(text.substr(chunkStart)); out.insert(out.end(), t.begin(), t.end()); }
    return out;
}
