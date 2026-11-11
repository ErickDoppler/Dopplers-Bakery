#include "web/server.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <algorithm>
#include <map>
#include <random>
#include <thread>
#include "chat/agent.h"
#include "util/image_io.h"
#include "util/json.h"
#include "util/sys.h"
#include "web/page.h"

namespace {
struct Req {
    std::string method, path, query, body;
    std::map<std::string, std::string> h;       // lower-case header names
};

std::string lower(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }
std::string trim(const std::string& s) { size_t a = s.find_first_not_of(" \r\n\t"), b = s.find_last_not_of(" \r\n\t"); return a == std::string::npos ? "" : s.substr(a, b - a + 1); }

std::string urlDecode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size() && isxdigit((unsigned char)s[i + 1]) && isxdigit((unsigned char)s[i + 2])) { o += (char)strtol(s.substr(i + 1, 2).c_str(), nullptr, 16); i += 2; }
        else if (s[i] == '+') o += ' ';
        else o += s[i];
    }
    return o;
}
std::string queryParam(const std::string& q, const std::string& k) {
    size_t p = 0;
    while (p < q.size()) {
        size_t e = q.find('&', p); if (e == std::string::npos) e = q.size();
        size_t eq = q.find('=', p);
        if (eq != std::string::npos && eq < e && q.substr(p, eq - p) == k) return urlDecode(q.substr(eq + 1, e - eq - 1));
        p = e + 1;
    }
    return "";
}

const char* mimeOf(const std::string& path) {
    std::string l = lower(path);
    auto ends = [&](const char* e) { size_t n = strlen(e); return l.size() >= n && l.compare(l.size() - n, n, e) == 0; };
    if (ends(".png")) return "image/png"; if (ends(".jpg") || ends(".jpeg")) return "image/jpeg"; if (ends(".gif")) return "image/gif";
    if (ends(".webp")) return "image/webp"; if (ends(".bmp")) return "image/bmp";
    if (ends(".mp3")) return "audio/mpeg"; if (ends(".wav")) return "audio/wav"; if (ends(".txt")) return "text/plain; charset=utf-8";
    return "application/octet-stream";
}
bool isImageName(const std::string& p) { std::string m = mimeOf(p); return m.rfind("image/", 0) == 0; }

bool sendAll(SOCKET s, const char* d, size_t n) {
    while (n) { int k = send(s, d, (int)std::min<size_t>(n, 1 << 20), 0); if (k <= 0) return false; d += k; n -= (size_t)k; }
    return true;
}
void reply(SOCKET s, int code, const char* type, const std::string& body, const std::string& extra = "") {
    const char* text = code == 200 ? "OK" : code == 206 ? "Partial Content" : code == 400 ? "Bad Request" : code == 401 ? "Unauthorized" : code == 403 ? "Forbidden" : code == 404 ? "Not Found" : "Error";
    char head[512];
    snprintf(head, sizeof head, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n%s\r\n", code, text, type, body.size(), extra.c_str());
    sendAll(s, head, strlen(head)); sendAll(s, body.data(), body.size());
}
void replyJson(SOCKET s, const Json& j, int code = 200) { reply(s, code, "application/json; charset=utf-8", j.dump()); }

bool readRequest(SOCKET s, Req& r) {
    std::string buf; char tmp[8192];
    size_t hend = std::string::npos;
    while (hend == std::string::npos) {
        int k = recv(s, tmp, sizeof tmp, 0);
        if (k <= 0) return false;
        buf.append(tmp, k);
        hend = buf.find("\r\n\r\n");
        if (buf.size() > 64 * 1024 && hend == std::string::npos) return false;
    }
    std::string head = buf.substr(0, hend);
    size_t le = head.find("\r\n");
    std::string line = head.substr(0, le);
    size_t a = line.find(' '), b = line.rfind(' ');
    if (a == std::string::npos || b <= a) return false;
    r.method = line.substr(0, a);
    std::string target = line.substr(a + 1, b - a - 1);
    size_t q = target.find('?');
    r.path = urlDecode(target.substr(0, q)); r.query = q == std::string::npos ? "" : target.substr(q + 1);
    size_t pos = le;
    while (pos != std::string::npos && pos < head.size()) {
        size_t nx = head.find("\r\n", pos + 2);
        std::string h = head.substr(pos + 2, nx == std::string::npos ? std::string::npos : nx - pos - 2);
        size_t c = h.find(':');
        if (c != std::string::npos) r.h[lower(trim(h.substr(0, c)))] = trim(h.substr(c + 1));
        pos = nx;
    }
    size_t want = r.h.count("content-length") ? (size_t)strtoull(r.h["content-length"].c_str(), nullptr, 10) : 0;
    if (want > 96u * 1024 * 1024) return false;
    r.body = buf.substr(hend + 4);
    while (r.body.size() < want) {
        int k = recv(s, tmp, sizeof tmp, 0);
        if (k <= 0) return false;
        r.body.append(tmp, k);
    }
    r.body.resize(want);
    return true;
}

std::string outRoot() { std::string o = sys::outputDir(); for (auto& c : o) if (c == '/') c = '\\'; return o; }

// absolute path of `rel` (relative to output\) when it really lies inside output\, otherwise ""
std::string safeOutputPath(const std::string& rel) {
    std::string r = rel; for (auto& c : r) if (c == '/') c = '\\';
    if (r.empty() || r.find("..") != std::string::npos || r.find(':') != std::string::npos || r[0] == '\\') return "";
    std::wstring full(MAX_PATH * 2, 0);
    DWORD n = GetFullPathNameW(sys::widen(sys::join(outRoot(), r)).c_str(), (DWORD)full.size(), &full[0], nullptr);
    if (!n || n >= full.size()) return "";
    full.resize(n);
    std::string f = sys::narrow(full), root = outRoot();
    if (f.size() <= root.size() + 1 || _strnicmp(f.c_str(), root.c_str(), root.size()) != 0 || f[root.size()] != '\\') return "";
    return f;
}

// path of a file under output\ as a '/' separated relative name, "" if it is somewhere else
std::string relToOutput(const std::string& abs) {
    std::string root = outRoot(), a = abs; for (auto& c : a) if (c == '/') c = '\\';
    if (a.size() > root.size() + 1 && _strnicmp(a.c_str(), root.c_str(), root.size()) == 0 && a[root.size()] == '\\') {
        std::string r = a.substr(root.size() + 1); for (auto& c : r) if (c == '\\') c = '/';
        return r;
    }
    return "";
}

Json stateJson() {
    ChatAgent& ag = ChatAgent::get();
    ag.poll();
    Json j = Json::object();
    j["busy"] = ag.busy(); j["status"] = ag.status(); j["live"] = ag.live();
    j["voice"] = ag.settings.voice; j["speaker"] = ag.settings.speaker;
    Json msgs = Json::array();
    for (auto& m : ag.transcript()) {
        Json o = Json::object(); o["role"] = m.role; o["text"] = m.text;
        Json fl = Json::array();
        for (auto& f : m.files) {
            std::string rel = relToOutput(f);
            if (rel.empty()) continue;
            Json e = Json::object(); e["url"] = "/file/" + rel; e["name"] = rel.substr(rel.find_last_of('/') + 1);
            std::string mt = mimeOf(f); e["kind"] = mt.rfind("image/", 0) == 0 ? "image" : mt.rfind("audio/", 0) == 0 ? "audio" : "file";
            fl.push(e);
        }
        o["files"] = fl; msgs.push(o);
    }
    j["messages"] = msgs;
    return j;
}

void serveFile(SOCKET s, const Req& r, const std::string& path) {
    std::string data;
    if (!sys::readFile(path, data)) { reply(s, 404, "text/plain", "not found"); return; }
    size_t from = 0, to = data.size() ? data.size() - 1 : 0; bool partial = false;
    auto it = r.h.find("range");
    if (it != r.h.end() && it->second.rfind("bytes=", 0) == 0 && !data.empty()) {
        std::string spec = it->second.substr(6);
        size_t dash = spec.find('-');
        if (dash != std::string::npos) {
            if (dash > 0) from = (size_t)strtoull(spec.c_str(), nullptr, 10);
            if (dash + 1 < spec.size()) to = (size_t)strtoull(spec.c_str() + dash + 1, nullptr, 10);
            else to = data.size() - 1;
            if (dash == 0) { size_t n = (size_t)strtoull(spec.c_str() + 1, nullptr, 10); from = n >= data.size() ? 0 : data.size() - n; to = data.size() - 1; }
            if (to >= data.size()) to = data.size() - 1;
            if (from <= to) partial = true; else from = 0, to = data.size() - 1;
        }
    }
    std::string body = data.empty() ? "" : data.substr(from, to - from + 1);
    char extra[160];
    snprintf(extra, sizeof extra, "Accept-Ranges: bytes\r\n%s", "");
    std::string ex = extra;
    if (partial) { char cr[96]; snprintf(cr, sizeof cr, "Content-Range: bytes %zu-%zu/%zu\r\n", from, to, data.size()); ex += cr; }
    reply(s, partial ? 206 : 200, mimeOf(path), body, ex);
}

std::string safeName(const std::string& n) {
    std::string o;
    for (char c : n) o += (isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_') ? c : '_';
    if (o.size() > 60) o = o.substr(o.size() - 60);
    return o.empty() ? "image.png" : o;
}
}  // namespace

WebServer& WebServer::get() { static WebServer w; return w; }

bool WebServer::clientActive() const { return sys::now() - lastPoll_ < 4.0; }

std::string WebServer::url() const {
    std::string u = "http://127.0.0.1:" + std::to_string(port_) + "/";
    if (lan_) u += "?key=" + key_;
    return u;
}

void WebServer::handleConnection(unsigned long long sk) {
    SOCKET s = (SOCKET)sk;
    DWORD to = 20000; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&to, sizeof to);
    Req r;
    if (!readRequest(s, r)) { closesocket(s); return; }
    ChatAgent& ag = ChatAgent::get();
    std::string setCookie;
    do {
        // DNS-rebinding guard on this-PC-only servers; access key on LAN servers
        std::string host = lower(r.h["host"]);
        if (!lan_) {
            std::string a = "127.0.0.1:" + std::to_string(port_), b = "localhost:" + std::to_string(port_);
            if (host != a && host != b) { reply(s, 403, "text/plain", "forbidden"); break; }
        } else {
            std::string cookie = r.h["cookie"], k = queryParam(r.query, "key");
            bool ok = k == key_ || cookie.find("bk=" + key_) != std::string::npos;
            if (!ok) { reply(s, 401, "text/plain", "Open the address shown in Dopplers Bakery (it contains ?key=...)."); break; }
            if (k == key_) setCookie = "Set-Cookie: bk=" + key_ + "; Path=/; HttpOnly; SameSite=Strict\r\n";
        }
        if (r.method == "POST" && r.h["x-bakery"] != "1") { reply(s, 403, "text/plain", "forbidden"); break; }

        if (r.method == "GET" && r.path == "/") { reply(s, 200, "text/html; charset=utf-8", kWebPage, setCookie); break; }
        if (r.method == "GET" && r.path == "/api/state") { lastPoll_ = sys::now(); replyJson(s, stateJson()); break; }
        if (r.method == "GET" && r.path.rfind("/file/", 0) == 0) {
            std::string p = safeOutputPath(r.path.substr(6));
            if (p.empty()) reply(s, 403, "text/plain", "forbidden"); else serveFile(s, r, p);
            break;
        }
        if (r.method == "POST" && r.path == "/api/upload") {
            std::string name = safeName(queryParam(r.query, "name"));
            if (!isImageName(name)) { replyJson(s, Json::object(), 400); break; }
            std::string dir = sys::join(outRoot(), "chat");
            sys::makeDirs(dir);
            std::string rel = "chat/web-" + sys::timestamp() + "-" + std::to_string(GetTickCount() % 10000) + "-" + name;
            std::string full = safeOutputPath(rel);
            Json j = Json::object();
            if (full.empty() || r.body.empty() || !sys::writeFile(full, r.body)) { j["error"] = "cannot save the file"; replyJson(s, j, 400); break; }
            j["path"] = rel; j["url"] = "/file/" + rel; replyJson(s, j); break;
        }
        if (r.method == "POST" && r.path == "/api/send") {
            bool ok; Json b = Json::parse(r.body, &ok);
            Json j = Json::object();
            if (!ok) { j["error"] = "bad request"; replyJson(s, j, 400); break; }
            std::vector<std::string> imgs;
            for (auto& x : b.get("images").a) { std::string p = safeOutputPath(x.asStr()); if (!p.empty() && isImageName(p)) imgs.push_back(p); }
            std::string err;
            if (b.has("voice")) ag.settings.voice = b.get("voice").boolean(false);
            if (b.has("speaker")) ag.settings.speaker = b.str("speaker", "Ryan");
            if (!ag.send(b.str("text"), imgs, err)) { j["error"] = err; replyJson(s, j, 400); break; }
            j["ok"] = true; replyJson(s, j); break;
        }
        if (r.method == "POST" && r.path == "/api/settings") {
            bool ok; Json b = Json::parse(r.body, &ok);
            if (ok) { if (b.has("voice")) ag.settings.voice = b.get("voice").boolean(false); if (b.has("speaker")) ag.settings.speaker = b.str("speaker", "Ryan"); }
            replyJson(s, stateJson()); break;
        }
        if (r.method == "POST" && r.path == "/api/stop") { ag.cancel(); replyJson(s, stateJson()); break; }
        if (r.method == "POST" && r.path == "/api/reset") { ag.reset(); replyJson(s, stateJson()); break; }
        reply(s, 404, "text/plain", "not found");
    } while (false);
    shutdown(s, SD_SEND);
    closesocket(s);
}

bool WebServer::start(int port, bool lan, std::string& err) {
    if (running_) return true;
    WSADATA wd;
    if (WSAStartup(MAKEWORD(2, 2), &wd) != 0) { err = "winsock is not available"; return false; }
    SOCKET ls = INVALID_SOCKET;
    for (int p = port; p < port + 20; p++) {
        ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (ls == INVALID_SOCKET) { err = "cannot create a socket"; return false; }
        sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons((u_short)p);
        a.sin_addr.s_addr = lan ? htonl(INADDR_ANY) : htonl(INADDR_LOOPBACK);
        if (bind(ls, (sockaddr*)&a, sizeof a) == 0 && listen(ls, 16) == 0) { port_ = p; break; }
        closesocket(ls); ls = INVALID_SOCKET;
    }
    if (ls == INVALID_SOCKET) { err = "no free port near " + std::to_string(port); return false; }
    lan_ = lan;
    std::random_device rd; char kb[40]; snprintf(kb, sizeof kb, "%08x%08x", rd(), rd()); key_ = kb;
    listen_ = (unsigned long long)ls; running_ = true;
    std::thread([this] {
        while (running_) {
            SOCKET c = accept((SOCKET)listen_, nullptr, nullptr);
            if (c == INVALID_SOCKET) { if (!running_) break; continue; }
            std::thread([this, c] { handleConnection((unsigned long long)c); }).detach();
        }
    }).detach();
    return true;
}

void WebServer::stop() {
    if (!running_) return;
    running_ = false;
    closesocket((SOCKET)listen_);
}
