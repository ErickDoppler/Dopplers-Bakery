// The Chat tab: messages with picture thumbnails and audio players, and a composer with a "+" menu,
// attachment thumbnails, drag-and-drop and Ctrl+V for pictures. One custom-drawn child window.
#pragma once
#include <windows.h>
#include <string>

HWND chatViewCreate(HWND parent);
HWND chatViewHwnd();
void chatViewShow(bool on);
void chatViewTick();                                 // call a few times per second: drives the agent and refreshes the view
void chatViewAddFile(const std::string& path);       // attach a picture to the next message (dropped, picked or pasted)
void chatViewStopSound();
