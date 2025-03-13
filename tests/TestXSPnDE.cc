// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include "bayesnet/classifiers/XSPnDE.h"  // <-- your new 2-superparent classifier
#include "TestUtils.h"                   // for RawDatasets, etc.

// Helper function to handle each (sp1, sp2) pair in tests
static void check_spnde_pair(
    int sp1, 
    int sp2, 
    RawDatasets &raw, 
    bool fitVector, 
    bool fitTensor)
{
  // Create our classifier
  bayesnet::XSpnde clf(sp1, sp2);

  // Option A: fit with vector-based data
  if (fitVector) {
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
  }
  // Option B: fit with the whole dataset in torch::Tensor form
  else if (fitTensor) {
    // your “tensor” version of fit
    clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
  }
  // Option C: or you might do the “dataset” version:
  else {
    clf.fit(raw.dataset, raw.features, raw.className, raw.states, raw.smoothing);
  }

  // Basic checks
  REQUIRE(clf.getNumberOfNodes() == 5);  // for iris: 4 features + 1 class
  // For XSpnde, edges are often computed as 3*nFeatures - 4. For iris nFeatures=4 => 3*4 -4 = 8
  REQUIRE(clf.getNumberOfEdges() == 8);
  REQUIRE(clf.getNotes().size() == 0);

  // Evaluate on test set
  float sc = clf.score(raw.X_test, raw.y_test);
  // If you know the exact expected accuracy for each pair, use:
  // REQUIRE(sc == Catch::Approx(someValue));
  // Otherwise, just check it's > some threshold:
  REQUIRE(sc >= 0.90f);  // placeholder; you can pick your own threshold
}

// ------------------------------------------------------------
// 1) Fit vector test
// ------------------------------------------------------------
TEST_CASE("fit vector test (XSPNDE)", "[XSPNDE]") {
  auto raw = RawDatasets("iris", true);

  // We’ll test a couple of two-superparent pairs, e.g. (0,1) and (2,3).
  // You can add more if you like, e.g. (0,2), (1,3), etc.
  std::vector<std::pair<int,int>> parentPairs = {
    {0,1}, {2,3}
  };
  for (auto &p : parentPairs) {
    // We’re doing the “vector” version
    check_spnde_pair(p.first, p.second, raw, /*fitVector=*/true, /*fitTensor=*/false);
  }
}

// ------------------------------------------------------------
// 2) Fit dataset test
// ------------------------------------------------------------
TEST_CASE("fit dataset test (XSPNDE)", "[XSPNDE]") {
  auto raw = RawDatasets("iris", true);

  // Again test multiple pairs:
  std::vector<std::pair<int,int>> parentPairs = {
    {0,2}, {1,3}
  };
  for (auto &p : parentPairs) {
    // Now do the “dataset” version
    check_spnde_pair(p.first, p.second, raw, /*fitVector=*/false, /*fitTensor=*/false);
  }
}

// ------------------------------------------------------------
// 3) Tensors dataset predict & predict_proba
// ------------------------------------------------------------
TEST_CASE("tensors dataset predict & predict_proba (XSPNDE)", "[XSPNDE]") {
  auto raw = RawDatasets("iris", true);

  // Let’s test a single pair or multiple pairs. For brevity:
  std::vector<std::pair<int,int>> parentPairs = {
    {0,3}
  };

  for (auto &p : parentPairs) {
    bayesnet::XSpnde clf(p.first, p.second);
    // Fit using the “tensor” approach
    clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);

    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 8);
    REQUIRE(clf.getNotes().size() == 0);

    // Check the score
    float sc = clf.score(raw.X_test, raw.y_test);
    REQUIRE(sc >= 0.90f);

    // You can also test predict_proba on a small slice:
    // e.g. the first 3 samples in X_test
    auto X_reduced = raw.X_test.slice(1, 0, 3); 
    auto proba = clf.predict_proba(X_reduced);

    // If you know exact probabilities, compare them with Catch::Approx.
    // For example:
    // REQUIRE(proba[0][0].item<double>() == Catch::Approx(0.98));
    // etc.
  }
}

