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

// "Improve" button: the chat-model request (an internal LLM job) that rewrites the prompt of a tab; false + err when it cannot be made
bool buildImproveJob(Kind k, const Json& form, Json& llmParams, std::string& err);          // backends/improve.cpp
const char* videoImproveSystemPrompt();                                                      // video_engine.cpp
std::string audioImproveSystemPrompt(const std::string& mode);                               // audio_engine.cpp

// Output file name helper: <exe>\output\<sub>\<timestamp>_<id>.<ext>
std::string outputPath(const char* sub, int jobId, const char* ext);


