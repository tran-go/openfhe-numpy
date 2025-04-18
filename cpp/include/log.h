#pragma once
#include <iostream>

#define LOG_INFO(msg) std::cout << "[OPENFHE-NUMPY INFO] " << msg << std::endl
#define LOG_WARN(msg) std::cout << "[OPENFHE-NUMPY WARNING] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[OPENFHE-NUMPY ERROR] " << msg << std::endl

#ifdef DEBUG
#define LOG_DEBUG(msg) std::cout << "[OPENFHE-NUMPY - DEBUG] " << msg << std::endl
#else
#define LOG_DEBUG(msg)
#endif