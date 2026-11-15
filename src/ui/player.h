// One audio player for the whole app (MP3 / WAV through Windows MCI): play, pause, stop (rewinds), seek, position.
// Only one thing plays at a time, whichever tab or the chat started it. UI thread only.
#pragma once
#include <string>

class Player {
public:
    static Player& get();
    bool play(const std::string& path);      // start `path` from the beginning (a different file replaces the current one)
    bool playCurrent();                      // start the loaded file again from the beginning if it ended or was stopped
    void pause();
    void resume();
    void stop();                             // stop and rewind to 0; the file stays loaded so Play starts it again
    void release() { closeFile(); }          // stop and unload (needed before the file is deleted)
    void toggle();                           // playing -> pause, paused -> resume, stopped -> play again
    void seek(double fraction);              // 0..1 of the length; keeps playing or pausing as before
    std::string current() const { return path_; }
    bool loaded() const { return !path_.empty(); }
    bool playing();                          // really producing sound right now
    bool paused() const { return paused_; }
    long long positionMs();
    long long lengthMs();
    static std::string timeText(long long ms);   // "1:05"

private:
    void closeFile();
    std::string path_;
    bool paused_ = false;
    long long length_ = 0;
};
