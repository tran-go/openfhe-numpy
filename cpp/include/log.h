#pragma once
#include <iostream>

#define OPENFHE_INFO(msg) std::cout << "[OPENFHE_INFO] " << msg << std::endl
#define OPENFHE_WARN(msg) std::cout << "[OPENFHE_WARNING] " << msg << std::endl
#define OPENFHE_ERROR(msg) std::cerr << "[OPENFHE_ERROR] " << msg << std::endl

#ifdef DEBUG
#define OPENFHE_DEBUG(msg) std::cout << "[OPENFHE_DEBUG] " << msg << std::endl
#else
#define OPENFHE_DEBUG(msg)
#endif
