//===- SimplifyCFG.h - Simplify and canonicalize the CFG --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for the pass responsible for both
/// simplifying and canonicalizing the CFG.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SIMPLIFYCFG_H
#define LLVM_TRANSFORMS_SCALAR_SIMPLIFYCFG_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Tapir/TapirTypes.h"

namespace llvm {

/// \brief A pass to simplify and canonicalize the CFG of a function.
///
/// This pass iteratively simplifies the entire CFG of a function, removing
/// unnecessary control flows and bringing it into the canonical form expected
/// by the rest of the mid-level optimizer.
class SimplifyCFGPass : public PassInfoMixin<SimplifyCFGPass> {
  int BonusInstThreshold;
  bool LateSimplifyCFG;
  tapir::TapirTarget* tapirTarget;

public:
  /// \brief Construct a pass with the default thresholds
  /// and switch optimizations.
  SimplifyCFGPass(tapir::TapirTarget* tapirTarget=nullptr);

  /// \brief Construct a pass with a specific bonus threshold
  /// and optional switch optimizations.
  SimplifyCFGPass(int BonusInstThreshold, bool LateSimplifyCFG, tapir::TapirTarget* tapirTarget);

  /// \brief Run the pass over the function.
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

}

#endif
