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
