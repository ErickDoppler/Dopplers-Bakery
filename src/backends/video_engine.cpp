// Text/Image to video: LTX-2.3 (audio + video DiT, Gemma 3 text encoder, video / audio VAE, vocoder, x2 latent upscaler) on ggml.
// The port is staged, see docs/LTX-PORT.md. Until the sampler exists this engine only validates the inputs and says what is missing.
#include "backends/engines.h"
#include "util/sys.h"

bool runVideo(const Job& job, JobCtl& ctl, std::string& err) {
    ctl.progress(0.0f, "checking the model files");
    for (const char* role : {"model", "clip", "upscaler", "lora"}) {
        const std::string path = job.params.str(role);
        if (path.empty() || !sys::exists(path)) { err = std::string("the LTX-2.3 ") + role + " file is not set or missing (Model settings of this tab)"; return false; }
    }
    err = "video generation is not available yet: the LTX-2.3 port is in progress (see docs/LTX-PORT.md). The tab, settings and queue are in place.";
    return false;
}
