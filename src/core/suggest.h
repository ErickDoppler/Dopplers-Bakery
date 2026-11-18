#pragma once
// Example text for a blank prompt field (what the chat does with its input box): a suggestion shown in grey, taken with Tab, and used when
// Create is pressed on a blank main prompt. A new random one each time a page is built, so the examples vary.
#include <string>
#include "core/types.h"

// "" when the field has no suggestion. `key` is the field's key (prompt, negative, pre_prompt, text, instruct, system).
// `pairWith` ties the TTS text and its style instruction to one mood: pass the same seed for both fields of one page.
std::string suggestionFor(Kind k, const char* key, unsigned seed);
