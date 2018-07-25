//===---- LoopSpawning.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass modifies Tapir loops to spawn their iterations efficiently.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H
#define LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H

#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Tapir/TapirUtils.h"

#define LS_NAME "loop-spawning"

namespace llvm {

/// LoopOutline serves as a base class for different variants of LoopSpawning.
/// LoopOutline implements common parts of LoopSpawning transformations, namely,
/// lifting a Tapir loop into a separate helper function.
class LoopOutline {
public:
   inline LoopOutline(Loop *OrigLoop, ScalarEvolution &SE,
              LoopInfo *LI, DominatorTree *DT,
              AssumptionCache *AC,
              OptimizationRemarkEmitter &ORE)
      : OrigLoop(OrigLoop), OrigFunction(OrigLoop->getHeader()->getParent()), SE(SE), LI(LI), DT(DT), AC(AC), ORE(ORE),
        ExitBlock(nullptr)
  {
    // Use the loop latch to determine the canonical exit block for this loop.
    TerminatorInst *TI = OrigLoop->getLoopLatch()->getTerminator();
    if (2 != TI->getNumSuccessors())
      return;
    ExitBlock = TI->getSuccessor(0);
    if (ExitBlock == OrigLoop->getHeader())
      ExitBlock = TI->getSuccessor(1);
  }

  virtual bool processLoop() = 0;

  virtual ~LoopOutline() {}

protected:
  PHINode* canonicalizeIVs(Type *Ty);
  Value* canonicalizeLoopLatch(PHINode *IV, Value *Limit);
  bool removeNonCanonicalIVs(BasicBlock* Header, BasicBlock* Preheader, PHINode* CanonicalIV, SmallVector<PHINode*, 8> &IVs, SCEVExpander &Exp);
  //bool setIVStartingValues();

  void unlinkLoop();

  /// The original loop.
  Loop * const OrigLoop;

  // Function containing original loop
  Function * const OrigFunction;

  /// A wrapper around ScalarEvolution used to add runtime SCEV checks. Applies
  /// dynamic knowledge to simplify SCEV expressions and converts them to a
  /// more usable form.
  // PredicatedScalarEvolution &PSE;
  ScalarEvolution &SE;
  /// Loop info.
  LoopInfo *LI;
  /// Dominator tree.
  DominatorTree *DT;
  /// Assumption cache.
  AssumptionCache *AC;
  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter &ORE;

  /// The exit block of this loop.  We compute our own exit block, based on the
  /// latch, and handle other exit blocks (i.e., for exception handling) in a
  /// special manner.
  BasicBlock *ExitBlock;
};

/// The LoopSpawning Pass.
struct LoopSpawningPass : public PassInfoMixin<LoopSpawningPass> {
  TapirTarget* tapirTarget;
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
}

#endif // LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H
