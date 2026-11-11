// Small built-in web server: the chat in a browser, pictures and sound included. Plain HTTP over winsock, no dependencies.
// By default it only listens on this PC (127.0.0.1). With `lan` it listens on every interface and asks for an access key.
#pragma once
#include <atomic>
#include <string>

class WebServer {
public:
    static WebServer& get();
    bool start(int port, bool lan, std::string& err);       // tries port, port+1 ... when it is busy
    void stop();
    bool running() const { return running_; }
    int port() const { return port_; }
    bool lan() const { return lan_; }
    std::string url() const;                                 // address to open (includes the key in LAN mode)
    bool clientActive() const;                               // a browser polled in the last few seconds (it plays the voice replies then)
    void handleConnection(unsigned long long sock);          // internal

private:
    std::atomic<bool> running_{false};
    std::atomic<double> lastPoll_{-100};
    unsigned long long listen_ = ~0ull;
    int port_ = 0;
    bool lan_ = false;
    std::string key_;
    friend struct ServerImpl;
};
