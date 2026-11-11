// Paths, UTF-8/UTF-16 helpers, small file utilities.
#pragma once
#include <string>
#include <vector>
#include <windows.h>

namespace sys {
std::wstring widen(const std::string& s);
std::string narrow(const std::wstring& s);
std::string exeDir();                       // folder holding the exe, no trailing slash
std::string modelsDir();                    // <exe>\models
std::string outputDir();                    // <exe>\output
std::string join(const std::string& a, const std::string& b);
bool exists(const std::string& p);
bool isDir(const std::string& p);
bool makeDirs(const std::string& p);
bool readFile(const std::string& p, std::string& out);
bool writeFile(const std::string& p, const std::string& data);
unsigned long long fileSize(const std::string& p);
std::string timestamp();                    // yyyymmdd-hhmmss
double now();                               // seconds, monotonic
std::string humanSize(unsigned long long n);
// recursive listing of files (relative paths with '/' separators) under root
void listFiles(const std::string& root, std::vector<std::string>& out, int maxDepth = 6);
void listDirs(const std::string& root, std::vector<std::string>& out, int maxDepth = 3);
}  // namespace sys
