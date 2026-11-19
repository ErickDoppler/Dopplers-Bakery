#pragma once
#include <windows.h>
#include <string>
#include "core/types.h"

// [Settings] panel of a tab: pick model / clip / vae / lora files. Returns true if changed.
bool showModelSettings(HWND parent, Kind k);
// Catalog download window with progress bar. `onlyMissing` pre-checks what is not installed.
// Returns true if anything was downloaded.
bool showDownloadDialog(HWND parent, bool firstLaunch);
// Small window with a number field. Returns true (and `out`) when the user confirmed a value from lo to hi.
bool askNumber(HWND parent, const std::string& title, const std::string& prompt, int initial, int lo, int hi, int& out);
// "Extend" panel of a finished video: how many seconds to add and which prompt to use (the same one, or a new one typed in the box that then appears).
bool askExtendClip(HWND parent, const std::string& clipName, double clipSeconds, int proposed, int maxSeconds, const std::string& samePrompt, int& seconds, bool& useNew, std::string& newPrompt);
