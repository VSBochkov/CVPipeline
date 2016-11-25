#include "cvpipeline.h"

int main(int argc, char** argv) {
    std::string config_file = argc > 1 ? argv[1] : "default_conf.json";
    cvpipeline pipeline(config_file);
    pipeline.process();
}
