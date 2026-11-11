// Dopplers Bakery: one exe, window UI when started plain, command line when given arguments.
#include <windows.h>
#include <shellapi.h>
#include <cstdio>
#include "cli/cli.h"
#include "ui/ui.h"
#include "util/sys.h"

int WINAPI WinMain(HINSTANCE inst, HINSTANCE, LPSTR, int) {
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argc <= 1) {
        LocalFree(argv);
        return runGui(inst);
    }
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) args.push_back(sys::narrow(argv[i]));
    LocalFree(argv);

    // Attach to the launching console so output shows up in cmd / PowerShell.
    // Keep stdout/stderr if they were redirected to a file or pipe.
    bool outRedir = GetFileType(GetStdHandle(STD_OUTPUT_HANDLE)) != FILE_TYPE_UNKNOWN;
    bool errRedir = GetFileType(GetStdHandle(STD_ERROR_HANDLE)) != FILE_TYPE_UNKNOWN;
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        if (!outRedir) freopen("CONOUT$", "w", stdout);
        if (!errRedir) freopen("CONOUT$", "w", stderr);
        SetConsoleOutputCP(CP_UTF8);
    }
    int rc = runCli(args);
    fflush(stdout); fflush(stderr);
    return rc;
}
