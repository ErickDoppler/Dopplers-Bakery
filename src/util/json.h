// Minimal JSON value, parser and writer. No dependencies.
#pragma once
#include <string>
#include <vector>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj } t = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::vector<std::pair<std::string, Json>> o;

    Json() {}
    Json(bool v) : t(Bool), b(v) {}
    Json(int v) : t(Num), n(v) {}
    Json(long long v) : t(Num), n((double)v) {}
    Json(double v) : t(Num), n(v) {}
    Json(const char* v) : t(Str), s(v) {}
    Json(const std::string& v) : t(Str), s(v) {}
    static Json array() { Json j; j.t = Arr; return j; }
    static Json object() { Json j; j.t = Obj; return j; }

    bool isNull() const { return t == Null; }
    bool has(const std::string& k) const { for (auto& kv : o) if (kv.first == k) return true; return false; }
    const Json& get(const std::string& k) const {
        static const Json nul;
        for (auto& kv : o) if (kv.first == k) return kv.second;
        return nul;
    }
    Json& operator[](const std::string& k) {
        if (t != Obj) { *this = object(); }
        for (auto& kv : o) if (kv.first == k) return kv.second;
        o.emplace_back(k, Json());
        return o.back().second;
    }
    void push(const Json& v) { if (t != Arr) *this = array(); a.push_back(v); }
    void erase(const std::string& k) { for (size_t i = 0; i < o.size(); i++) if (o[i].first == k) { o.erase(o.begin() + i); return; } }

    double num(double def = 0) const { return t == Num ? n : (t == Bool ? (b ? 1 : 0) : (t == Str && !s.empty() ? atof(s.c_str()) : def)); }
    int integer(int def = 0) const { return t == Null ? def : (int)std::llround(num(def)); }
    bool boolean(bool def = false) const { return t == Bool ? b : (t == Num ? n != 0 : def); }
    std::string asStr(const std::string& def = "") const { return t == Str ? s : def; }
    double num(const std::string& k, double def) const { const Json& j = get(k); return j.isNull() ? def : j.num(def); }
    int integer(const std::string& k, int def) const { const Json& j = get(k); return j.isNull() ? def : j.integer(def); }
    bool boolean(const std::string& k, bool def) const { const Json& j = get(k); return j.isNull() ? def : j.boolean(def); }
    std::string str(const std::string& k, const std::string& def = "") const { const Json& j = get(k); return j.t == Str ? j.s : def; }

    // ---------- writer ----------
    static void esc(std::string& out, const std::string& v) {
        out += '"';
        for (unsigned char c : v) {
            switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
                else out += (char)c;
            }
        }
        out += '"';
    }
    void write(std::string& out, int indent, int depth) const {
        auto nl = [&](int d) { if (indent >= 0) { out += '\n'; out.append((size_t)d * indent, ' '); } };
        switch (t) {
        case Null: out += "null"; break;
        case Bool: out += b ? "true" : "false"; break;
        case Num: {
            char buf[40];
            if (n == std::floor(n) && std::fabs(n) < 1e15) snprintf(buf, sizeof buf, "%lld", (long long)n);
            else snprintf(buf, sizeof buf, "%.9g", n);
            out += buf; break;
        }
        case Str: esc(out, s); break;
        case Arr:
            out += '[';
            for (size_t i = 0; i < a.size(); i++) { if (i) out += ','; nl(depth + 1); a[i].write(out, indent, depth + 1); }
            if (!a.empty()) nl(depth);
            out += ']'; break;
        case Obj:
            out += '{';
            for (size_t i = 0; i < o.size(); i++) {
                if (i) out += ',';
                nl(depth + 1); esc(out, o[i].first); out += indent >= 0 ? ": " : ":";
                o[i].second.write(out, indent, depth + 1);
            }
            if (!o.empty()) nl(depth);
            out += '}'; break;
        }
    }
    std::string dump(int indent = -1) const { std::string r; write(r, indent, 0); return r; }

    // ---------- parser ----------
    static Json parse(const std::string& src, bool* ok = nullptr) {
        P p{src.c_str(), src.c_str() + src.size(), true};
        Json j = p.value();
        p.ws();
        if (p.p != p.e) p.good = false;
        if (ok) *ok = p.good;
        return p.good ? j : Json();
    }

private:
    struct P {
        const char* p; const char* e; bool good;
        void ws() { while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++; }
        static void utf8(std::string& o, unsigned cp) {
            if (cp < 0x80) o += (char)cp;
            else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 63)); }
            else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
            else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 63)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
        }
        unsigned hex4() {
            unsigned v = 0;
            for (int i = 0; i < 4; i++) {
                if (p >= e) { good = false; return 0; }
                char c = *p++; v <<= 4;
                if (c >= '0' && c <= '9') v |= c - '0'; else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
                else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10; else good = false;
            }
            return v;
        }
        std::string str() {
            std::string r; p++;
            while (p < e && *p != '"') {
                if (*p == '\\') {
                    p++; if (p >= e) break;
                    char c = *p++;
                    switch (c) {
                    case 'n': r += '\n'; break; case 't': r += '\t'; break; case 'r': r += '\r'; break;
                    case 'b': r += '\b'; break; case 'f': r += '\f'; break;
                    case 'u': {
                        unsigned cp = hex4();
                        if (cp >= 0xD800 && cp < 0xDC00 && p + 1 < e && p[0] == '\\' && p[1] == 'u') { p += 2; unsigned lo = hex4(); cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); }
                        utf8(r, cp); break;
                    }
                    default: r += c;
                    }
                } else r += *p++;
            }
            if (p >= e) { good = false; return r; }
            p++; return r;
        }
        Json value() {
            ws();
            if (p >= e) { good = false; return Json(); }
            char c = *p;
            if (c == '{') {
                Json j = Json::object(); p++; ws();
                if (p < e && *p == '}') { p++; return j; }
                while (good) {
                    ws(); if (p >= e || *p != '"') { good = false; break; }
                    std::string k = str(); ws();
                    if (p >= e || *p != ':') { good = false; break; }
                    p++; j.o.emplace_back(k, value()); ws();
                    if (p < e && *p == ',') { p++; continue; }
                    if (p < e && *p == '}') { p++; break; }
                    good = false;
                }
                return j;
            }
            if (c == '[') {
                Json j = Json::array(); p++; ws();
                if (p < e && *p == ']') { p++; return j; }
                while (good) {
                    j.a.push_back(value()); ws();
                    if (p < e && *p == ',') { p++; continue; }
                    if (p < e && *p == ']') { p++; break; }
                    good = false;
                }
                return j;
            }
            if (c == '"') return Json(str());
            if (p + 4 <= e && !strncmp(p, "true", 4)) { p += 4; return Json(true); }
            if (p + 5 <= e && !strncmp(p, "false", 5)) { p += 5; return Json(false); }
            if (p + 4 <= e && !strncmp(p, "null", 4)) { p += 4; return Json(); }
            char* end = nullptr;
            double d = strtod(p, &end);
            if (end == p) { good = false; return Json(); }
            p = end; return Json(d);
        }
    };
};
