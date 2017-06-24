#include <engine/Engine.h>
#include <memory>
#include <engine/util/Logger.h>
#include "application/CUDAPathTracer/PathTracingApp.h"

int main(int, char**)
{
    std::unique_ptr<Engine> engine = std::make_unique<Engine>();
    std::unique_ptr<PathTracingApp> game = std::make_unique<PathTracingApp>();

    engine->init(game.get());
	
    while (engine->running())
    {
        engine->update();
    }
	
    engine->shutdown();

    return 0;
}
