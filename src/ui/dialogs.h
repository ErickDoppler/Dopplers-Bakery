#pragma once
#include <windows.h>
#include "core/types.h"

// [Settings] panel of a tab: pick model / clip / vae / lora files. Returns true if changed.
bool showModelSettings(HWND parent, Kind k);
// Catalog download window with progress bar. `onlyMissing` pre-checks what is not installed.
// Returns true if anything was downloaded.
bool showDownloadDialog(HWND parent, bool firstLaunch);
