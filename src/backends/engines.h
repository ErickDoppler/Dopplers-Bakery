#pragma once
#include <string>
#include "core/queue.h"

bool runLLM(const Job& job, JobCtl& ctl, std::string& err);
void releaseLLM();                                                  // llm_engine.cpp: free the cached chat model     // llm_engine.cpp   (llama.cpp / ggml)
bool runTTS(const Job& job, JobCtl& ctl, std::string& err);     // pending_engines.cpp (Qwen3-TTS port)
bool runImage(const Job& job, JobCtl& ctl, std::string& err);   // pending_engines.cpp (Krea2 port)
bool runAudio(const Job& job, JobCtl& ctl, std::string& err);   // pending_engines.cpp (Stable Audio 3 port)

bool expandVideoPromptForTest(const std::string& ggufModel, const std::string& prompt, std::string& out);   // video_engine.cpp: the chat-model prompt expansion alone
bool runVideo(const Job& job, JobCtl& ctl, std::string& err);   // video_engine.cpp (LTX-2.3 port, in progress)

// Output file name helper: <exe>\output\<sub>\<timestamp>_<id>.<ext>
std::string outputPath(const char* sub, int jobId, const char* ext);


