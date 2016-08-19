#pragma once

#include "SonsodeCommon.h"

namespace Sonsode {
	namespace Utils {
		inline size_t GetBlockCount(size_t dataSize) {
			return (dataSize / BLOCK_SIZE) + ((dataSize % BLOCK_SIZE) > 0 ? 1 : 0);
		}

		inline size_t GetBlockCount(size_t dataSize, size_t overlay) {
			return (dataSize - 2 * overlay) / (BLOCK_SIZE - overlay) + (((dataSize - 2 * overlay) % (BLOCK_SIZE - overlay)) > 0 ? 1 : 0);
		}
	}
}