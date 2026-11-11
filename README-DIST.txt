Dopplers Bakery
===============
Offline studio on your own PC: chat that makes pictures, music, sound effects and speech; plus text, image, audio and
voice tools. Nothing leaves the computer after the models are downloaded.

Start:    double-click DopplersBakery.exe
          Keep this whole folder together - the .dll files must stay next to the exe.

First launch: a dialog offers to download the models (or "Search in folder" to point at models you already have).
Models go to   models\      Results go to   output\      (both created next to the exe)

CHAT (first tab)
  Type what you want: "make a picture of a lighthouse at sunset", "20 seconds of calm piano", "a door creaking".
  + button      upload a picture, paste one, turn voice replies on and pick the voice, start a new chat
  Pictures      drag a picture into the window, press Ctrl+V on a copied picture, or use +. Then say how to change it.
  Voice         the speaker button reads the replies aloud.

CHAT IN THE BROWSER
  The same chat runs at  http://127.0.0.1:8787  (menu Web > Open the chat in the browser). Pictures and sound play there too.
  Web > Share on the local network lets a phone or another PC on your Wi-Fi use it (needs the access key in the link).
  Without the window:  DopplersBakery.exe serve        (add --lan to share)

Needs: Windows 10/11 64-bit, a CPU with AVX2 (2013 or newer). A GPU with up-to-date Vulkan drivers
(NVIDIA, AMD or Intel) is used automatically; without one it runs on the CPU, much slower.
Rough VRAM: images 8 GB+, audio 8 GB+, speech 6 GB+; the chat model is small.

Command line:  DopplersBakery.exe chat --say "make me a picture of a fox"
               DopplersBakery.exe image --prompt "..."    (see: DopplersBakery.exe image --help)

Uses llama.cpp / ggml (MIT license, see LICENSE-llama.cpp.txt).
Krea 2 models are under the Krea 2 license; check it before sharing generated images or the model files.

License: free for personal use; enterprise use needs a paid license (US$100 per workstation). See LICENSE.txt.

Images above about 1264 px: the built-in Vulkan backend cannot render them correctly (they are reduced). For larger images open a tab's
"Model settings" and press "Download CUDA libraries" (NVIDIA GPU, about 0.45 GB), restart, then pick Backend = CUDA on the Text to image tab.
