// Declarative description of each tab's generation settings. The GUI builds its controls
// from it and the CLI uses it for flags/defaults, so both stay in sync.
#pragma once
#include <string>
#include <vector>
#include "core/types.h"
#include "util/json.h"

enum class FT { Text, Multi, Int, Float, Slider, Combo, Check, File };

struct Field {
    const char* key;
    const char* label;
    FT type;
    double def;          // numbers / checkbox
    double mn, mx;       // Int / Float / Slider range
    const char* sdef;    // text default or combo default
    const char* items;   // combo items, '|' separated
    int lines;           // Multi height in text lines
};

const std::vector<Field>& fieldsOf(Kind k);
Json defaultsOf(Kind k);                       // every field with its default value
std::string jobTitle(Kind k, const Json& params);

// Form values + the tab's selected model files (resolved to absolute paths) = job params.
Json makeJobParams(Kind k, const Json& form);
