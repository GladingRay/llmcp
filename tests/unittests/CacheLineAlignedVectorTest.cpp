#include "llmcp/Support/CacheLineAlignedVector.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace llmcp;

TEST(CacheLineAlignedVector, Alignment) {
  CacheLineAlignedVector<float> values(1024, 1.0f);
  ASSERT_EQ(reinterpret_cast<std::uintptr_t>(values.data()) % kCacheLineSize,
            0u);
  EXPECT_EQ(values[0], 1.0f);
  EXPECT_EQ(values[1023], 1.0f);
}

TEST(CacheLineAlignedVector, Growth) {
  CacheLineAlignedVector<float> values;
  for (int i = 0; i < 10000; ++i)
    values.push_back(static_cast<float>(i));
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(values.data()) % kCacheLineSize,
            0u);
  EXPECT_EQ(values.front(), 0.0f);
  EXPECT_EQ(values.back(), 9999.0f);
}

TEST(CacheLineAlignedVector, Empty) {
  CacheLineAlignedVector<float> values;
  EXPECT_TRUE(values.empty());
  EXPECT_EQ(values.data(), nullptr);
}
