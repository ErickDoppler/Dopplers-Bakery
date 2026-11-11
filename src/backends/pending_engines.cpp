// Shared helper for the engines (every engine is implemented in its own file now).
#include "backends/engines.h"
#include "util/sys.h"

std::string outputPath(const char* sub, int jobId, const char* ext) {
    return sys::join(sys::join(sys::outputDir(), sub), sys::timestamp() + "_" + std::to_string(jobId) + ext);
}
