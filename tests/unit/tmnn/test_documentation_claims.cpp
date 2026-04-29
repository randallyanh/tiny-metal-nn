/**
 * @file test_documentation_claims.cpp
 * @brief Regression tests for public benchmark/SOTA positioning claims.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

std::string read_repo_file(const std::filesystem::path &relative_path) {
  const auto path = std::filesystem::path(TMNN_REPO_ROOT) / relative_path;
  std::ifstream in(path);
  if (!in) {
    ADD_FAILURE() << "Unable to open " << path;
    return {};
  }
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

} // namespace

TEST(DocumentationClaims, DoesNotPublishUnsupportedMlxBaselineClaim) {
  const std::string status = read_repo_file("STATUS.md");
  const std::string comparison =
      read_repo_file("docs/VS-MLX-AND-TCNN.md");

  ASSERT_NE(status.find("Comparison benchmark vs MLX"), std::string::npos);
  ASSERT_NE(status.find("Not yet measured / published"), std::string::npos);

  EXPECT_EQ(comparison.find("now includes a first same-machine `MLX` "
                            "baseline slice"),
            std::string::npos);
}

TEST(DocumentationClaims, MicrobenchReferencesAreNotClaimedAsCtestGates) {
  const std::string phase2 =
      read_repo_file("docs/know-how/_baselines/010-phase-2-acceptance.md");

  ASSERT_NE(phase2.find("G5b Transient allocate"), std::string::npos);
  EXPECT_EQ(phase2.find("so the gate is checked on every\nctest invocation"),
            std::string::npos);
}
