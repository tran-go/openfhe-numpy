#pragma once
#include <iostream>

#define INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define WARN(msg) std::cout << "[WARNING] " << msg << std::endl
#define ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl


#ifdef DEBUG
#ifndef DEBUG
#define DEBUG(msg) std::cout << "[DEBUG] " << msg << std::endl
#endif
#else
#ifndef DEBUG
#define DEBUG(msg)
#endif
#endif