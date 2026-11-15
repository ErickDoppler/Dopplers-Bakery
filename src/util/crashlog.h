// bakery.log next to the exe (devices, engine warnings, crashes), a readable message instead of a silent exit when the AI engine
// aborts, and a marker file that tells the next start that the app died inside a GPU job.
#pragma once
#include <string>

void crashLogInstall();                          // log file + ggml log/abort hooks + unhandled-exception report; call once at start
void logLine(const std::string& text);           // append a timestamped line to bakery.log (thread safe, the file is kept small)
void crashContext(const std::string& what);      // what is running right now; part of the crash report
void crashGuardBegin(const std::string& what);   // write the marker before a risky GPU job ...
void crashGuardEnd();                            // ... and remove it when the job ended normally (success or error)
bool crashGuardRecover(std::string& what);       // true (and the marker is removed) when the previous run died inside a guarded job
