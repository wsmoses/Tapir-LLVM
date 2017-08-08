//===------------- LoopFuse.h - Loop Fusion Utility -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// Fuse two adjacent loops to improve cache locality.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include <list>

namespace llvm {
/// \brief The pass class.
class LoopFuse : public FunctionPass {

public:
  // Kind of fusion made.
  enum Kind {
    NO_FUSION = 0,   // Fusion was not made even to check dependence legality.
                     // This is when loops had failed basic structure checks.
    REVERTED_FUSION, // Fusion was reverted due to failed dependence legality.
    PURE_FUSION,     // Fusion succeeded with removal of original loops.
    VERSIONED_FUSION // Fusion succeeded with versioning due to runtime checks.
  };

private:
  // Analyses used.
  LoopInfo *LI;
  LoopAccessLegacyAnalysis *LAA;
  DominatorTree *DT;
  ScalarEvolution *SE;

  // FusionSwitcher - Branch instruction that controls switching between
  // original and fused versions. This gets initialized to true when loops are
  // multiversioned to check fusion legality. By default, it points to original
  // version.
  BranchInst *FusionSwitcher;

  Loop *FusedLoop;

  // LAI for FusedLoop.
  const LoopAccessInfo *LAI;

  // Kind of fusion that happened.
  Kind FusionKind = NO_FUSION;

  // CustomVMap: VMap of BBs for fused loop. The problem about having
  // ValueToValueMapTy passed from a client is that it gets updated when the
  // loops are removed based on fusion success and this is undesirable. Also
  // a ValueToValueMapTy is used when both Values are present. So, only a
  // normal llvm::Value* is maintained as map's value in contrast with
  // ValueToValueMapTy's WeakVH. Clients can use this mapping as a VMap.
  typedef std::map<const Value *, Value *> CustomVMap;
  CustomVMap VMap;

  // Rewrite IncomingBlocks in PHIs of @Br's successor blocks from Br's parent
  // to @To.
  void RewritePHI(BranchInst *Br, BasicBlock *To);

  // Fuse loops - @L1 and @L2 and return the fused loop.
  Loop *FuseLoops(Loop &L1, Loop &L2);

  // Legality and profitability checks.
  bool DependenceLegal(Loop &L1, Loop &L2);
  bool DefsUsedAcrossLoops(Loop &L1, Loop &L2);
  bool IsLegalAndProfitable(Loop &L1, Loop &L2);

  // Removal routines based on fusion success.
  void RemoveLoopCompletelyWithPreheader(Loop &L);
  void RemoveFusionSwitcher(Loop &L);

  // Outside use updates.
  void UpdateUsesOutsideLoop(Loop &L);
  void AddPHIsOutsideLoop(Loop &L, BasicBlock *OrigIncomingBlock);

public:
  LoopFuse() : FunctionPass(ID) {
    initializeLoopFusePass(*PassRegistry::getPassRegistry());
  }

  // Initialization interface when this pass is used as a utility.
  LoopFuse(LoopInfo *_LI, LoopAccessLegacyAnalysis *_LAA, DominatorTree *_DT,
           ScalarEvolution *_SE)
      : FunctionPass(ID), LI(_LI), LAA(_LAA), DT(_DT), SE(_SE) {}

  Loop *getFusedLoop() { return FusedLoop; }

  const CustomVMap &getVMap() { return VMap; }

  unsigned getFusionKind() { return FusionKind; }

  // Interface; when this pass is used as a utility.
  bool run(Loop &L1, Loop &L2);

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<LoopAccessLegacyAnalysis>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();

    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<ScalarEvolutionWrapperPass>();
  }

  static char ID;
};
} // anonymous namespace
