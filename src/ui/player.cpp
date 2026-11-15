#include "ui/player.h"
#include <windows.h>
#include <mmsystem.h>
#include <algorithm>
#include "util/sys.h"

namespace {
const wchar_t* kAlias = L"bkplayer";
bool ends(const std::string& s, const char* e) {
    std::string l = s; for (auto& c : l) c = (char)tolower((unsigned char)c);
    size_t n = strlen(e); return l.size() >= n && l.compare(l.size() - n, n, e) == 0;
}
bool mci(const std::wstring& cmd, std::wstring* out = nullptr) {
    wchar_t buf[128] = {};
    bool ok = mciSendStringW(cmd.c_str(), buf, 128, nullptr) == 0;
    if (out) *out = buf;
    return ok;
}
std::wstring status(const wchar_t* what) {
    std::wstring o; mci(std::wstring(L"status ") + kAlias + L" " + what, &o); return o;
}
}  // namespace

Player& Player::get() { static Player p; return p; }

void Player::closeFile() {
    mci(std::wstring(L"close ") + kAlias);
    path_.clear(); paused_ = false; length_ = 0;
}

bool Player::play(const std::string& path) {
    if (path == path_ && loaded()) return playCurrent();
    closeFile();
    std::wstring open = L"open \"" + sys::widen(path) + L"\" type " + (ends(path, ".wav") ? L"waveaudio" : L"mpegvideo") + L" alias " + kAlias;
    if (!mci(open)) return false;
    mci(std::wstring(L"set ") + kAlias + L" time format milliseconds");
    path_ = path;
    length_ = _wtoi64(status(L"length").c_str());
    paused_ = false;
    return mci(std::wstring(L"play ") + kAlias);
}

bool Player::playCurrent() {
    if (!loaded()) return false;
    paused_ = false;
    if (length_ > 0 && positionMs() >= length_ - 150) mci(std::wstring(L"seek ") + kAlias + L" to start");      // ended: from the top, otherwise from where it is
    return mci(std::wstring(L"play ") + kAlias);
}

void Player::pause() { if (loaded() && playing()) { mci(std::wstring(L"stop ") + kAlias); paused_ = true; } }      // stop keeps the position
void Player::resume() { if (loaded()) playCurrent(); }

void Player::stop() {
    if (!loaded()) return;
    mci(std::wstring(L"stop ") + kAlias);
    mci(std::wstring(L"seek ") + kAlias + L" to start");
    paused_ = false;
}

void Player::toggle() {
    if (!loaded()) return;
    if (playing()) pause(); else playCurrent();
}

bool Player::playing() { return loaded() && status(L"mode") == L"playing"; }

long long Player::positionMs() { return loaded() ? _wtoi64(status(L"position").c_str()) : 0; }
long long Player::lengthMs() { return length_; }

void Player::seek(double f) {
    if (!loaded() || length_ <= 0) return;
    f = std::min(1.0, std::max(0.0, f));
    long long ms = std::min<long long>((long long)(f * length_), std::max<long long>(0, length_ - 50));
    bool wasPlaying = playing();
    mci(std::wstring(L"seek ") + kAlias + L" to " + std::to_wstring(ms));
    paused_ = false;
    if (wasPlaying) mci(std::wstring(L"play ") + kAlias);          // seeking stops MCI playback: carry on from the new position
}

std::string Player::timeText(long long ms) {
    long long s = std::max<long long>(0, ms / 1000);
    char b[24]; snprintf(b, sizeof b, "%lld:%02lld", s / 60, s % 60);
    return b;
}
