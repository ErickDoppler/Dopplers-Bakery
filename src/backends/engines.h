#pragma once
#include <string>
#include "core/queue.h"

bool runLLM(const Job& job, JobCtl& ctl, std::string& err);     // llm_engine.cpp   (llama.cpp / ggml)
bool runTTS(const Job& job, JobCtl& ctl, std::string& err);     // pending_engines.cpp (Qwen3-TTS port)
bool runImage(const Job& job, JobCtl& ctl, std::string& err);   // pending_engines.cpp (Krea2 port)
bool runAudio(const Job& job, JobCtl& ctl, std::string& err);   // pending_engines.cpp (Stable Audio 3 port)

// Output file name helper: <exe>\output\<sub>\<timestamp>_<id>.<ext>
std::string outputPath(const char* sub, int jobId, const char* ext);
