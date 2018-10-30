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
              LoopInfo &LI, DominatorTree &DT,
              AssumptionCache &AC,
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
  const SCEV* getLimit();

    /// \brief Compute the grainsize of the loop, based on the limit.
    ///
    /// The grainsize is computed by the following equation:
    ///
    ///     Grainsize = min(2048, ceil(Limit / (8 * workers)))
    ///
    /// This computation is inserted into the preheader of the loop.
    ///
    /// TODO: This method is the only method that depends on the CilkABI.
    /// Generalize this method for other grainsize calculations and to query TLI.
  Value* computeGrainsize(Value *Limit, TapirTarget* tapirTarget, Type* T=nullptr);

  Value* canonicalizeLoopLatch(PHINode *IV, Value *Limit);

  bool getHandledExits(BasicBlock* Header, SmallPtrSetImpl<BasicBlock *> &HandledExits);

  bool removeNonCanonicalIVs(BasicBlock* Header, BasicBlock* Preheader, PHINode* CanonicalIV, SmallVectorImpl<PHINode*> &IVs);
  bool setIVStartingValues(Value* newStart, Value* CanonicalIV, const SmallVectorImpl<PHINode*> &IVs, BasicBlock* NewPreheader, ValueToValueMapTy &VMap);

    // In the general case, var is the result of some computation
    // in the loop's preheader. The pass wants to prevent outlining from passing
    // var as an arbitrary argument to the outlined function, but one that is
    // potentially in a specific place for ABI reasons.
    // Hence, this pass adds the loop-limit variable as an argument
    // manually.
    //
    // There are two special cases to consider: the var is a constant, or
    // the var is used elsewhere within the loop.  To handle these two
    // cases, this pass adds an explict argument for var, to ensure it isn't
    // clobberred by the other use or not passed because it is constant.
  Value* ensureDistinctArgument(const std::vector<BasicBlock *> &LoopBlocks, Value* var, const Twine &name="");

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
  LoopInfo &LI;
  /// Dominator tree.
  DominatorTree &DT;
  /// Assumption cache.
  AssumptionCache &AC;
  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter &ORE;

  /// The exit block of this loop.  We compute our own exit block, based on the
  /// latch, and handle other exit blocks (i.e., for exception handling) in a
  /// special manner.
  BasicBlock *ExitBlock;
};

}

#endif // LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H
