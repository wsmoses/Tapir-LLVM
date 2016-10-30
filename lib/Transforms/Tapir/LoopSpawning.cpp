//===- LoopSpawning.cpp - Spawn loop iterations efficiently ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Modify Tapir loops to spawn their iterations efficiently.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/LoopSpawning.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Tapir/CilkABI.h"
#include "llvm/Transforms/Tapir/TapirOutline.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <utility>

using std::make_pair;

using namespace llvm;

#define LS_NAME "loop-spawning"
#define DEBUG_TYPE LS_NAME

STATISTIC(LoopsAnalyzed, "Number of Tapir loops analyzed");
STATISTIC(LoopsConvertedToDAC,
          "Number of Tapir loops converted to divide-and-conquer iteration spawning");

namespace {
// Forward declarations.
class LoopSpawningHints;

/// \brief This modifies LoopAccessReport to initialize message with
/// tapir-loop-specific part.
class TapirLoopReport : public LoopAccessReport {
public:
  TapirLoopReport(Instruction *I = nullptr)
      : LoopAccessReport("loop not transformed: ", I) {}

  /// \brief This allows promotion of the loop-access analysis report into the
  /// tapir-loop report.  It modifies the message to add the tapir-loop-specific
  /// part of the message.
  explicit TapirLoopReport(const LoopAccessReport &R)
      : LoopAccessReport(Twine("loop not transformed: ") + R.str(),
                         R.getInstr()) {}
};


/// Utility class for getting and setting loop spawning hints in the form
/// of loop metadata.
/// This class keeps a number of loop annotations locally (as member variables)
/// and can, upon request, write them back as metadata on the loop. It will
/// initially scan the loop for existing metadata, and will update the local
/// values based on information in the loop.
class LoopSpawningHints {
  enum HintKind { HK_STRATEGY };

  /// Hint - associates name and validation with the hint value.
  struct Hint {
    const char *Name;
    unsigned Value; // This may have to change for non-numeric values.
    HintKind Kind;

    Hint(const char *Name, unsigned Value, HintKind Kind)
        : Name(Name), Value(Value), Kind(Kind) {}

    bool validate(unsigned Val) {
      switch (Kind) {
      case HK_STRATEGY:
        return (Val < ST_END);
      }
      return false;
    }
  };

  /// Spawning strategy
  Hint Strategy;

  /// Return the loop metadata prefix.
  static StringRef Prefix() { return "tapir.loop."; }

public:
  enum SpawningStrategy {
    ST_NONE,
    ST_DAC,
    ST_END,
  };

  static std::string printStrategy(enum SpawningStrategy Strat) {
    switch(Strat) {
    case LoopSpawningHints::ST_NONE:
      return "Do not convert";
    case LoopSpawningHints::ST_DAC:
      return "Use divide-and-conquer";
    case LoopSpawningHints::ST_END:
      return "Unspecified";
    default:
      return "Unknown";
    }
  }

  LoopSpawningHints(const Loop *L, OptimizationRemarkEmitter &ORE)
      : Strategy("spawn.strategy", ST_NONE, HK_STRATEGY),
        TheLoop(L), ORE(ORE) {
    // Populate values with existing loop metadata.
    getHintsFromMetadata();
  }

  /// Dumps all the hint information.
  std::string emitRemark() const {
    TapirLoopReport R;
    R << "Strategy = " << printStrategy(getStrategy());

    return R.str();
  }

  enum SpawningStrategy getStrategy() const {
    return (SpawningStrategy)Strategy.Value;
  }

private:
  /// Find hints specified in the loop metadata and update local values.
  void getHintsFromMetadata() {
    MDNode *LoopID = TheLoop->getLoopID();
    if (!LoopID)
      return;

    // First operand should refer to the loop id itself.
    assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
    assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      const MDString *S = nullptr;
      SmallVector<Metadata *, 4> Args;

      // The expected hint is either a MDString or a MDNode with the first
      // operand a MDString.
      if (const MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i))) {
        if (!MD || MD->getNumOperands() == 0)
          continue;
        S = dyn_cast<MDString>(MD->getOperand(0));
        for (unsigned i = 1, ie = MD->getNumOperands(); i < ie; ++i)
          Args.push_back(MD->getOperand(i));
      } else {
        S = dyn_cast<MDString>(LoopID->getOperand(i));
        assert(Args.size() == 0 && "too many arguments for MDString");
      }

      if (!S)
        continue;

      // Check if the hint starts with the loop metadata prefix.
      StringRef Name = S->getString();
      if (Args.size() == 1)
        setHint(Name, Args[0]);
    }
  }

  /// Checks string hint with one operand and set value if valid.
  void setHint(StringRef Name, Metadata *Arg) {
    if (!Name.startswith(Prefix()))
      return;
    Name = Name.substr(Prefix().size(), StringRef::npos);

    const ConstantInt *C = mdconst::dyn_extract<ConstantInt>(Arg);
    if (!C)
      return;
    unsigned Val = C->getZExtValue();

    Hint *Hints[] = {&Strategy};
    for (auto H : Hints) {
      if (Name == H->Name) {
        if (H->validate(Val))
          H->Value = Val;
        else
          DEBUG(dbgs() << LS_NAME << " ignoring invalid hint '" <<
                Name << "'\n");
        break;
      }
    }
  }

  /// Create a new hint from name / value pair.
  MDNode *createHintMetadata(StringRef Name, unsigned V) const {
    LLVMContext &Context = TheLoop->getHeader()->getContext();
    Metadata *MDs[] = {MDString::get(Context, Name),
                       ConstantAsMetadata::get(
                           ConstantInt::get(Type::getInt32Ty(Context), V))};
    return MDNode::get(Context, MDs);
  }

  /// Matches metadata with hint name.
  bool matchesHintMetadataName(MDNode *Node, ArrayRef<Hint> HintTypes) {
    MDString *Name = dyn_cast<MDString>(Node->getOperand(0));
    if (!Name)
      return false;

    for (auto H : HintTypes)
      if (Name->getString().endswith(H.Name))
        return true;
    return false;
  }

  /// Sets current hints into loop metadata, keeping other values intact.
  void writeHintsToMetadata(ArrayRef<Hint> HintTypes) {
    if (HintTypes.size() == 0)
      return;

    // Reserve the first element to LoopID (see below).
    SmallVector<Metadata *, 4> MDs(1);
    // If the loop already has metadata, then ignore the existing operands.
    MDNode *LoopID = TheLoop->getLoopID();
    if (LoopID) {
      for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
        MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
        // If node in update list, ignore old value.
        if (!matchesHintMetadataName(Node, HintTypes))
          MDs.push_back(Node);
      }
    }

    // Now, add the missing hints.
    for (auto H : HintTypes)
      MDs.push_back(createHintMetadata(Twine(Prefix(), H.Name).str(), H.Value));

    // Replace current metadata node with new one.
    LLVMContext &Context = TheLoop->getHeader()->getContext();
    MDNode *NewLoopID = MDNode::get(Context, MDs);
    // Set operand 0 to refer to the loop id itself.
    NewLoopID->replaceOperandWith(0, NewLoopID);

    TheLoop->setLoopID(NewLoopID);
  }

  /// The loop these hints belong to.
  const Loop *TheLoop;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter &ORE;
};

static void emitAnalysisDiag(const Loop *TheLoop,
                             OptimizationRemarkEmitter &ORE,
                             const LoopAccessReport &Message) {
  const char *Name = LS_NAME;
  LoopAccessReport::emitAnalysis(Message, TheLoop, Name, ORE);
}

static void emitMissedWarning(Function *F, Loop *L,
                              const LoopSpawningHints &LH,
                              OptimizationRemarkEmitter *ORE) {
  ORE->emitOptimizationRemarkMissed(LS_NAME, L, LH.emitRemark());
  switch (LH.getStrategy()) {
  case LoopSpawningHints::ST_DAC:
    emitLoopSpawningWarning(
        F->getContext(), *F, L->getStartLoc(),
        "failed to use divide-and-conquer loop spawning");
    break;
  case LoopSpawningHints::ST_NONE:
    emitLoopSpawningWarning(
        F->getContext(), *F, L->getStartLoc(),
        "spawning conversion disabled");
    break;
  case LoopSpawningHints::ST_END:
    emitLoopSpawningWarning(
        F->getContext(), *F, L->getStartLoc(),
        "unknown spawning strategy");
    break;
  }
}

class DACLoopSpawning {
public:
  DACLoopSpawning(Loop *OrigLoop, ScalarEvolution &SE,
                  LoopInfo *LI, DominatorTree *DT,
                  const TargetLibraryInfo *TLI,
                  const TargetTransformInfo *TTI,
                  OptimizationRemarkEmitter *ORE)
      : OrigLoop(OrigLoop), SE(SE), LI(LI), DT(DT),
        TLI(TLI), TTI(TTI), ORE(ORE) {}

  bool convertLoop();

  virtual ~DACLoopSpawning() {}

protected:
    bool verifyLoopExit(const Loop *L);
    // bool verifyLoopStructureForConversion(const Loop *L);
    PHINode* canonicalizeIVs(Loop *L, Type *Ty, ScalarEvolution &SE, DominatorTree &DT);
    Value* canonicalizeLoopLatch(Loop *L, PHINode *IV, Value *Limit);
    Value* computeGrainsize(Loop *L, Value *Limit);
    void implementDACIterSpawnOnHelper(Function *Helper,
                                       BasicBlock *Preheader,
                                       BasicBlock *Header,
                                       PHINode *CanonicalIV,
                                       Argument *Limit,
                                       Argument *Grainsize,
                                       DominatorTree *DT,
                                       LoopInfo *LI,
                                       bool CanonicalIVFlagNUW = false,
                                       bool CanonicalIVFlagNSW = false);
    void eraseLoop(Loop *L, ScalarEvolution &SE, DominatorTree &DT, LoopInfo &LoopInfo);
    // Function* convertLoopToDACIterSpawn(Loop *L, ScalarEvolution &SE,
    //                                     DominatorTree &DT, LoopInfo &LI);

  /// The original loop.
  Loop *OrigLoop;
  /// A wrapper around ScalarEvolution used to add runtime SCEV checks. Applies
  /// dynamic knowledge to simplify SCEV expressions and converts them to a
  /// more usable form.
  // PredicatedScalarEvolution &PSE;
  ScalarEvolution &SE;
  /// Loop Info.
  LoopInfo *LI;
  /// Dominator Tree.
  DominatorTree *DT;
  /// Target Library Info.
  const TargetLibraryInfo *TLI;
  /// Target Transform Info.
  const TargetTransformInfo *TTI;
  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

};

/// The LoopSpawning Pass.
struct LoopSpawning : public FunctionPass {
  /// Pass identification, replacement for typeid
  static char ID;

  explicit LoopSpawning()
      : FunctionPass(ID) {
    initializeLoopSpawningPass(*PassRegistry::getPassRegistry());
  }
  
  LoopSpawningPass Impl;

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    auto *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
    auto *TLI = TLIP ? &TLIP->getTLI() : nullptr;
    auto *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto *AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto *ORE = &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

    return Impl.runImpl(F, *SE, *LI, *TTI, *DT, TLI, *AA, *AC, *ORE);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  }
};

} // end anonymous namespace


/// Verify that the CFG between the loop exit and the subsequent sync is
/// appropriately simple -- essentially, a straight sequence of basic blocks
/// that unconditionally branch to the sync.  Return false if this process
/// fails, indicating that this loop is not a simple loop for conversion.
///
/// This method only checks the structure of the loop exit, rather than modify
/// it, to preserve analyses.
bool DACLoopSpawning::verifyLoopExit(const Loop *L) {
  BasicBlock *LoopExit = L->getExitBlock();
  SmallPtrSet<BasicBlock *, 8> Visited;
  Visited.insert(LoopExit);
  // Attempt to remove empty blocks terminated by unconditional
  // branches that follow the loop exit until the loop exit is empty
  // aside from a sync.
  do {
    // LoopExit = L->getExitBlock();
    // dbgs() << "verifyLoopExit: Checking block " << *LoopExit;
    // // Fail if the loop exit has multiple predecessors.
    // if (!LoopExit->getUniquePredecessor()) {
    //   dbgs() << "LS loop exit does not have a unique predecessor.\n";
    //   return false;
    // }

    // Fail if the loop exit contains non-terminator instructions
    // other than PHI nodes or debug instructions.
    if (LoopExit->getFirstNonPHIOrDbg() != LoopExit->getTerminator()) {
      dbgs() << "LS loop exit contains non-terminator instructions other than PHI nodes and debug instructions.\n";
      return false;
    }

    // If the loop exit is terminated by a sync, we're done.
    if (isa<SyncInst>(LoopExit->getTerminator()))
      return true;

    // Check if the loop exit is terminated by an unconditional branch.
    if (BranchInst *Br = dyn_cast_or_null<BranchInst>(LoopExit->getTerminator())) {
      if (Br->isConditional()) {
        dbgs() << "LS loop exit is terminated by a conditional branch.\n";
        return false;
      }
    } else {
      dbgs() << "LS loop exit is not terminated by a branch.\n";
      return false;
    }
    LoopExit = LoopExit->getTerminator()->getSuccessor(0);
  } while (Visited.insert(LoopExit).second);
  // } while (TryToSimplifyUncondBranchFromEmptyBlock(LoopExit));

  dbgs() << "LS could not confirm simple exit to sync.\n";
  return false;
}

// /// Verify assumptions on the structure of the Loop:
// /// 1) Loop has a single preheader, latch, and exit.
// /// 2) Loop header is terminated by a detach.
// /// 3) Continuation of detach in header is the latch.
// /// 4) All other predecessors of the latch are terminated by reattach
// /// instructions.
// bool DACLoopSpawning::verifyLoopStructureForConversion(const Loop *L) {
//   const BasicBlock *Header = L->getHeader();
//   const BasicBlock *Preheader = L->getLoopPreheader();
//   const BasicBlock *Latch = L->getLoopLatch();
//   const BasicBlock *Exit = L->getExitBlock();

//   // dbgs() << "LS checking structure of " << *L;

//   // Header must be terminated by a detach.
//   if (!isa<DetachInst>(Header->getTerminator())) {
//     DEBUG(dbgs() << "LS Loop header is not terminated by a detach.\n");
//     return false;
//   }

//   // Loop must have a single preheader.
//   if (nullptr == Preheader) {
//     DEBUG(dbgs() << "LS Loop preheader not found.\n");
//     return false;
//   }

//   // Loop must have a unique latch.
//   if (nullptr == Latch) {
//     DEBUG(dbgs() << "LS Loop does not have a unique latch.\n");
//     return false;
//   }

//   // Loop must have a unique exit block.
//   if (nullptr == Exit) {
//     DEBUG(dbgs() << "LS Loop does not have a unique exit block.\n");
//     return false;
//   }

//   // Continuation of header terminator must be the latch.
//   const DetachInst *HeaderDetach = cast<DetachInst>(Header->getTerminator());
//   const BasicBlock *Continuation = HeaderDetach->getContinue();
//   if (Continuation != Latch) {
//     DEBUG(dbgs() << "LS Continuation of detach in header is not the latch.\n");
//     return false;
//   }

//   // All other predecessors of Latch are terminated by reattach instructions.
//   for (auto PI = pred_begin(Latch), PE = pred_end(Latch);  PI != PE; ++PI) {
//     const BasicBlock *Pred = *PI;
//     if (Header == Pred) continue;
//     if (!isa<ReattachInst>(Pred->getTerminator())) {
//       DEBUG(dbgs() << "LS Latch has a predecessor that is not terminated by a reattach.\n");
//       return false;
//     }
//   }

//   return verifyLoopExit(L);
// }

/// Canonicalize the induction variables in the loop L.  Return the canonical
/// induction variable created or inserted by the scalar evolution expander.
PHINode* DACLoopSpawning::canonicalizeIVs(Loop *L, Type *Ty, ScalarEvolution &SE, DominatorTree &DT) {
  BasicBlock* Header = L->getHeader();
  Module* M = Header->getParent()->getParent();

  DEBUG(dbgs() << "LS Header:" << *Header);
  BasicBlock *Latch = L->getLoopLatch();
  DEBUG(dbgs() << "LS Latch:" << *Latch);

  DEBUG(dbgs() << "LS exiting block:" << *(L->getExitingBlock()));

  // dbgs() << "LS SE trip count: " << SE->getSmallConstantTripCount(L, L->getExitingBlock()) << "\n";
  // dbgs() << "LS SE trip multiple: " << SE->getSmallConstantTripMultiple(L, L->getExitingBlock()) << "\n";
  DEBUG(dbgs() << "LS SE backedge taken count: " << *(SE.getBackedgeTakenCount(L)) << "\n");
  DEBUG(dbgs() << "LS SE max backedge taken count: " << *(SE.getMaxBackedgeTakenCount(L)) << "\n");
  DEBUG(dbgs() << "LS SE exit count: " << *(SE.getExitCount(L, L->getExitingBlock())) << "\n");

  SCEVExpander Exp(SE, M->getDataLayout(), "l2c");

  PHINode *CanonicalIV = Exp.getOrInsertCanonicalInductionVariable(L, Ty);
  DEBUG(dbgs() << "LS Canonical induction variable " << *CanonicalIV << "\n");

  SmallVector<WeakVH, 16> DeadInsts;
  Exp.replaceCongruentIVs(L, &DT, DeadInsts);
  // dbgs() << "Updated header:" << *(L->getHeader());
  // dbgs() << "Updated exiting block:" << *(L->getExitingBlock());
  for (WeakVH V : DeadInsts) {
    DEBUG(dbgs() << "LS erasing dead inst " << *V << "\n");
    Instruction *I = cast<Instruction>(V);
    I->eraseFromParent();
  }

  return CanonicalIV;
}

/// \brief Replace the Latch of Loop L to check that IV is always less
/// than or equal to Limit.
///
/// This method assumes that L has a single loop latch and a single
/// exit block.
Value* DACLoopSpawning::canonicalizeLoopLatch(Loop *L, PHINode *IV, Value *Limit) {
  Value *NewCondition;
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  assert(Latch && "No single loop latch found for loop.");

  IRBuilder<> Builder(&*Latch->getFirstInsertionPt());

  // This process assumes that IV's increment is in Latch.

  // Create comparison between IV and Limit at top of Latch.
  NewCondition = Builder.CreateICmpULT(IV, Limit);

  // Replace the conditional branch at the end of Latch.
  BranchInst *LatchBr = dyn_cast_or_null<BranchInst>(Latch->getTerminator());
  assert(LatchBr && LatchBr->isConditional() &&
         "Latch does not terminate with a conditional branch.");
  Builder.SetInsertPoint(Latch->getTerminator());
  assert(L->getExitBlock() &&
         "Loop does not have a single exit block.");
  Builder.CreateCondBr(NewCondition, Header, L->getExitBlock());

  // Erase the old conditional branch.
  LatchBr->eraseFromParent();

  return NewCondition;
}

/// \brief Compute the grainsize of Loop L, based on Limit.
///
/// The grainsize is computed by the following equation:
///
///     Grainsize = min(2048, ceil(Limit / (8 * workers)))
///
/// This computation is inserted into the Preheader of the Loop L.
///
/// TODO: This method is the only method that depends on the CilkABI.
/// Generalize this method for other grainsize calculations and to query TLI.
Value* DACLoopSpawning::computeGrainsize(Loop *L, Value *Limit) {
  Value *Grainsize;
  BasicBlock *Preheader = L->getLoopPreheader();
  assert(Preheader && "No Preheader found for loop.");

  IRBuilder<> Builder(Preheader->getTerminator());

  // Get 8 * workers
  Value *Workers8 = Builder.CreateIntCast(cilk::GetOrCreateWorker8(*Preheader->getParent()),
                                          Limit->getType(), false);
  // Compute ceil(limit / 8 * workers) = (limit + 8 * workers - 1) / (8 * workers)
  Value *SmallLoopVal =
    Builder.CreateUDiv(Builder.CreateSub(Builder.CreateAdd(Limit, Workers8),
                                         ConstantInt::get(Limit->getType(), 1)),
                       Workers8);
  // Compute min
  Value *LargeLoopVal = ConstantInt::get(Limit->getType(), 2048);
  Value *Cmp = Builder.CreateICmpULT(LargeLoopVal, SmallLoopVal);
  Grainsize = Builder.CreateSelect(Cmp, LargeLoopVal, SmallLoopVal);

  return Grainsize;
}

/// \brief Method to help convertLoopToDACIterSpawn convert the Tapir
/// loop cloned into function Helper to spawn its iterations in a
/// parallel divide-and-conquer fashion.
///
/// Example: Suppose that Helper contains the following Tapir loop:
///
/// Helper(iter_t start, iter_t end, iter_t grain, ...) {
///   iter_t i = start;
///   ... Other loop setup ...
///   do {
///     spawn { ... loop body ... };
///   } while (i++ < end);
///   sync;
/// }
///
/// Then this method transforms Helper into the following form:
///
/// Helper(iter_t start, iter_t end, iter_t grain, ...) {
/// recur:
///   iter_t itercount = end - start;
///   if (itercount > grain) {
///     // Invariant: itercount >= 2
///     count_t miditer = start + itercount / 2;
///     spawn Helper(start, miditer, grain, ...);
///     start = miditer + 1;
///     goto recur;
///   }
///
///   iter_t i = start;
///   ... Other loop setup ...
///   do {
///     ... Loop Body ...
///   } while (i++ < end);
///   sync;
/// }
///
void DACLoopSpawning::implementDACIterSpawnOnHelper(Function *Helper,
                                                    BasicBlock *Preheader,
                                                    BasicBlock *Header,
                                                    PHINode *CanonicalIV,
                                                    Argument *Limit,
                                                    Argument *Grainsize,
                                                    DominatorTree *DT,
                                                    LoopInfo *LI,
                                                    bool CanonicalIVFlagNUW,
                                                    bool CanonicalIVFlagNSW) {
  // Serialize the cloned copy of the loop.
  assert(Preheader->getParent() == Helper &&
         "Preheader does not belong to helper function.");
  assert(Header->getParent() == Helper &&
         "Header does not belong to helper function.");
  assert(CanonicalIV->getParent() == Header &&
         "CanonicalIV does not belong to header");
  assert(isa<DetachInst>(Header->getTerminator()) &&
         "Cloned header is not terminated by a detach.");
  DetachInst *DI = dyn_cast<DetachInst>(Header->getTerminator());
  SerializeDetachedCFG(DI, DT);

  // Convert the cloned loop into the strip-mined loop body.

  BasicBlock *DACHead = Preheader;
  if (&(Helper->getEntryBlock()) == Preheader)
    // Split the entry block.  We'll want to create a backedge into
    // the split block later.
    DACHead = SplitBlock(Preheader, &(Preheader->front()), DT, LI);

  BasicBlock *RecurHead, *RecurDet, *RecurCont;
  Value *IterCount;
  Value *CanonicalIVInput;
  PHINode *CanonicalIVStart;
  {
    Instruction *PreheaderOrigFront = &(DACHead->front());
    IRBuilder<> Builder(PreheaderOrigFront);
    // Create branch based on grainsize.
    DEBUG(dbgs() << "LS CanonicalIV: " << *CanonicalIV << "\n");
    CanonicalIVInput = CanonicalIV->getIncomingValueForBlock(DACHead);
    CanonicalIVStart = Builder.CreatePHI(CanonicalIV->getType(), 2,
                                         CanonicalIV->getName()+".dac");
    CanonicalIVInput->replaceAllUsesWith(CanonicalIVStart);
    IterCount = Builder.CreateSub(Limit, CanonicalIVStart,
                                  "itercount");
    Value *IterCountCmp = Builder.CreateICmpUGT(IterCount, Grainsize);
    // dbgs() << "DAC head before split:" << *DACHead;
    TerminatorInst *RecurTerm =
      SplitBlockAndInsertIfThen(IterCountCmp, PreheaderOrigFront,
                                /*Unreachable=*/false,
                                /*BranchWeights=*/nullptr,
                                DT);
    RecurHead = RecurTerm->getParent();
    // Create skeleton of divide-and-conquer recursion:
    // DACHead -> RecurHead -> RecurDet -> RecurCont -> DACHead
    RecurDet = SplitBlock(RecurHead, RecurHead->getTerminator(),
                          DT, LI);
    RecurCont = SplitBlock(RecurDet, RecurDet->getTerminator(),
                           DT, LI);
    RecurCont->getTerminator()->replaceUsesOfWith(RecurTerm->getSuccessor(0),
                                                  DACHead);
    // Builder.SetInsertPoint(&(RecurCont->front()));
    // Builder.CreateBr(DACHead);
    // RecurCont->getTerminator()->eraseFromParent();
  }

  // Compute mid iteration in RecurHead.
  Value *MidIter, *MidIterPlusOne;
  {
    IRBuilder<> Builder(&(RecurHead->front()));
    MidIter = Builder.CreateAdd(CanonicalIVStart,
                                Builder.CreateLShr(IterCount, 1,
                                                   "halfcount"),
                                "miditer",
                                CanonicalIVFlagNUW, CanonicalIVFlagNSW);
  }

  // Create recursive call in RecurDet.
  {
    // Create input array for recursive call.
    IRBuilder<> Builder(&(RecurDet->front()));
    SetVector<Value*> RecurInputs;
    Function::arg_iterator AI = Helper->arg_begin();
    assert(cast<Argument>(CanonicalIVInput) == &*AI &&
           "First argument does not match original input to canonical IV.");
    RecurInputs.insert(CanonicalIVStart);
    ++AI;
    assert(Limit == &*AI &&
           "Second argument does not match original input to the loop limit.");
    RecurInputs.insert(MidIter);
    ++AI;
    for (Function::arg_iterator AE = Helper->arg_end();
         AI != AE;  ++AI)
        RecurInputs.insert(&*AI);
    // RecurInputs.insert(CanonicalIVStart);
    // // for (PHINode *IV : IVs)
    // //   RecurInputs.insert(DACStart[IV]);
    // RecurInputs.insert(Limit);
    // RecurInputs.insert(Grainsize);
    // for (Value *V : BodyInputs)
    //   RecurInputs.insert(VMap[V]);
    DEBUG({
        dbgs() << "RecurInputs: ";
        for (Value *Input : RecurInputs)
          dbgs() << *Input << ", ";
        dbgs() << "\n";
      });

    // Create call instruction.
    CallInst *RecurCall = Builder.CreateCall(Helper, RecurInputs.getArrayRef());
    RecurCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
    // Use a fast calling convention for the helper.
    RecurCall->setCallingConv(CallingConv::Fast);
    // RecurCall->setCallingConv(Helper->getCallingConv());
  }

  // Set up continuation of detached recursive call.  We effectively
  // inline this tail call automatically.
  {
    IRBuilder<> Builder(&(RecurCont->front()));
    MidIterPlusOne = Builder.CreateAdd(MidIter,
                                       ConstantInt::get(Limit->getType(), 1),
                                       "miditerplusone",
                                       CanonicalIVFlagNUW,
                                       CanonicalIVFlagNSW);
  }

  // Finish setup of new phi node for canonical IV.
  {
    CanonicalIVStart->addIncoming(CanonicalIVInput, Preheader);
    CanonicalIVStart->addIncoming(MidIterPlusOne, RecurCont);
  }

  /// Make the recursive DAC parallel.
  {
    IRBuilder<> Builder(RecurHead->getTerminator());
    // Create the detach.
    Builder.CreateDetach(RecurDet, RecurCont);
    RecurHead->getTerminator()->eraseFromParent();
    // Create the reattach.
    Builder.SetInsertPoint(RecurDet->getTerminator());
    Builder.CreateReattach(RecurCont);
    RecurDet->getTerminator()->eraseFromParent();
  }
}

/// Recursive routine to remove a loop and all of its subloops.
static void removeLoopAndAllSubloops(Loop *L, LoopInfo &LoopInfo) {
  for (Loop *SL : L->getSubLoops())
    removeLoopAndAllSubloops(SL, LoopInfo);

  dbgs() << "Removing " << *L << "\n";

  SmallPtrSet<BasicBlock *, 8> Blocks;
  Blocks.insert(L->block_begin(), L->block_end());
  for (BasicBlock *BB : Blocks)
    LoopInfo.removeBlock(BB);

  LoopInfo.markAsRemoved(L);
}

// Recursive routine to traverse the subloops of a loop and push all 
static void collectLoopAndAllSubLoops(Loop *L,
                                      SetVector<Loop*> &SubLoops) {
  for (Loop *SL : L->getSubLoops())
    collectLoopAndAllSubLoops(SL, SubLoops);
  SubLoops.insert(L);
}

/// Recursive routine to mark a loop and all of its subloops as removed.
static void markLoopAndAllSubloopsAsRemoved(Loop *L, LoopInfo &LoopInfo) {
  for (Loop *SL : L->getSubLoops())
    markLoopAndAllSubloopsAsRemoved(SL, LoopInfo);

  dbgs() << "Marking as removed:" << *L << "\n";
  LoopInfo.markAsRemoved(L);
}

/// Erase the specified loop, and update analysis accordingly.
///
/// TODO: Depracate this method in favor of using LoopDeletion pass.
void DACLoopSpawning::eraseLoop(Loop *L, ScalarEvolution &SE,
                          DominatorTree &DT, LoopInfo &LoopInfo) {
    // Get components of the old loop.
    BasicBlock *Preheader = L->getLoopPreheader();
    assert(Preheader && "Loop does not have a unique preheader.");
    BasicBlock *ExitBlock = L->getExitBlock();
    assert(ExitBlock && "Loop does not have a unique exit block.");
    BasicBlock *ExitingBlock = L->getExitingBlock();
    assert(ExitingBlock && "Loop does not have a unique exiting block.");

    // Invalidate the analysis of the old loop.
    SE.forgetLoop(L);

    // Redirect the preheader to branch directly to loop exit.
    assert(1 == Preheader->getTerminator()->getNumSuccessors() &&
           "Preheader does not have a unique successor.");
    Preheader->getTerminator()->replaceUsesOfWith(L->getHeader(),
                                                  ExitBlock);
    
    // Rewrite phis in the exit block cto get their inputs from
    // the preheader instead of the exiting block.
    BasicBlock::iterator BI = ExitBlock->begin();
    while (PHINode *P = dyn_cast<PHINode>(BI)) {
      int j = P->getBasicBlockIndex(ExitingBlock);
      assert(j >= 0 && "Can't find exiting block in exit block's phi node!");
      P->setIncomingBlock(j, Preheader);
      P->removeIncomingValue(ExitingBlock);
      ++BI;
    }

  // Update the dominator tree and remove the instructions and blocks that will
  // be deleted from the reference counting scheme.
  SmallVector<DomTreeNode*, 8> ChildNodes;
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI) {
    // Move all of the block's children to be children of the preheader, which
    // allows us to remove the domtree entry for the block.
    ChildNodes.insert(ChildNodes.begin(), DT[*LI]->begin(), DT[*LI]->end());
    for (DomTreeNode *ChildNode : ChildNodes) {
      DT.changeImmediateDominator(ChildNode, DT[Preheader]);
    }

    ChildNodes.clear();
    DT.eraseNode(*LI);

    // Remove the block from the reference counting scheme, so that we can
    // delete it freely later.
    (*LI)->dropAllReferences();
  }

  // Erase the instructions and the blocks without having to worry
  // about ordering because we already dropped the references.
  // NOTE: This iteration is safe because erasing the block does not remove its
  // entry from the loop's block list.  We do that in the next section.
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI)
    (*LI)->eraseFromParent();

  // Finally, the blocks from loopinfo.  This has to happen late because
  // otherwise our loop iterators won't work.

  SetVector<Loop *> SubLoops;
  collectLoopAndAllSubLoops(L, SubLoops);
  
  SmallPtrSet<BasicBlock *, 8> Blocks;
  Blocks.insert(L->block_begin(), L->block_end());
  for (BasicBlock *BB : Blocks)
    LoopInfo.removeBlock(BB);

  // The last step is to update LoopInfo now that we've eliminated this loop.
  // for (Loop *SL : L->getSubLoops())
  //   LoopInfo.markAsRemoved(SL);
  // LoopInfo.markAsRemoved(L);
  // markLoopAndAllSubloopsAsRemoved(L, LoopInfo);
  // removeLoopAndAllSubloops(L, LoopInfo);
  for (Loop *SL : SubLoops)
    LoopInfo.markAsRemoved(SL);
}

/// Top-level call to convert loop to spawn its iterations in a
/// divide-and-conquer fashion.
bool DACLoopSpawning::convertLoop() {

  // Verify the exit of this loop.
  if (!verifyLoopExit(OrigLoop))
    return false;
  
  Loop *L = OrigLoop;

  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();
  Module* M = Header->getParent()->getParent();

  DEBUG(dbgs() << "LS loop header:" << *Header);
  DEBUG(dbgs() << "LS loop latch:" << *Latch);
  DEBUG(dbgs() << "LS exiting block:" << *(L->getExitingBlock()));

  // dbgs() << "LS SE trip count: " << SE->getSmallConstantTripCount(L, L->getExitingBlock()) << "\n";
  // dbgs() << "LS SE trip multiple: " << SE->getSmallConstantTripMultiple(L, L->getExitingBlock()) << "\n";
  DEBUG(dbgs() << "LS SE backedge taken count: " << *(SE.getBackedgeTakenCount(L)) << "\n");
  DEBUG(dbgs() << "LS SE max backedge taken count: " << *(SE.getMaxBackedgeTakenCount(L)) << "\n");
  DEBUG(dbgs() << "LS SE exit count: " << *(SE.getExitCount(L, L->getExitingBlock())) << "\n");

  /// Get loop limit.
  const SCEV *Limit = SE.getBackedgeTakenCount(L);
  // const SCEV *Limit = SE.getAddExpr(BETC, SE.getOne(BETC->getType()));
  DEBUG(dbgs() << "LS Loop limit: " << *Limit << "\n");
  // PredicatedScalarEvolution PSE(SE, *L);
  // const SCEV *PLimit = PSE.getBackedgeTakenCount();
  // DEBUG(dbgs() << "LS predicated loop limit: " << *PLimit << "\n");
  if (SE.getCouldNotCompute() == Limit) {
    DEBUG(dbgs() << "SE could not compute loop limit.  Quitting extractLoop.\n");
    return false;
  }

  /// Clean up the loop's induction variables.
  PHINode *CanonicalIV = canonicalizeIVs(L, Limit->getType(), SE, *DT);
  const SCEVAddRecExpr *CanonicalSCEV =
    cast<const SCEVAddRecExpr>(SE.getSCEV(CanonicalIV));
  // dbgs() << "[Loop2Cilk] Current loop preheader";
  // dbgs() << *Preheader;
  // dbgs() << "[Loop2Cilk] Current loop:";
  // for (BasicBlock *BB : L->getBlocks()) {
  //   dbgs() << *BB;
  // }
  if (!CanonicalIV) {
    DEBUG(dbgs() << "Could not get canonical IV.  Quitting extractLoop.\n");
    return false;
  }

  // Remove all IV's other can CanonicalIV.
  // First, check that we can do this.
  bool CanRemoveIVs = true;
  for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
    PHINode *PN = cast<PHINode>(II);
    if (CanonicalIV == PN) continue;
    dbgs() << "IV " << *PN;
    const SCEV *S = SE.getSCEV(PN);
    dbgs() << " SCEV " << *S << "\n";
    if (SE.getCouldNotCompute() == S)
      CanRemoveIVs = false;
  }

  if (!CanRemoveIVs) {
    DEBUG(dbgs() << "Could not compute SCEV for all IV's.  Quitting extractLoop.\n");
    return false;
  }

  ////////////////////////////////////////////////////////////////////////
  // We now have everything we need to extract the loop.  It's time to
  // do some surgery.

  SCEVExpander Exp(SE, M->getDataLayout(), "l2c");

  // Remove the IV's (other than CanonicalIV) and replace them with
  // their stronger forms.
  //
  // TODO?: We can probably adapt this loop->DAC process such that we
  // don't require all IV's to be canonical.
  {
    SmallVector<PHINode*, 8> IVsToRemove;
    for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
      PHINode *PN = cast<PHINode>(II);
      if (PN == CanonicalIV) continue;
      const SCEV *S = SE.getSCEV(PN);
      Value *NewIV = Exp.expandCodeFor(S, S->getType(), CanonicalIV);
      PN->replaceAllUsesWith(NewIV);
      IVsToRemove.push_back(PN);
    }
    for (PHINode *PN : IVsToRemove)
      PN->eraseFromParent();
  }
  // dbgs() << "EL Preheader after IV removal:" << *Preheader;
  // dbgs() << "EL Header after IV removal:" << *Header;
  // dbgs() << "EL Latch after IV removal:" << *Latch;

  // All remaining IV's should be canonical.  Collect them.
  //
  // TODO?: We can probably adapt this loop->DAC process such that we
  // don't require all IV's to be canonical.
  SmallVector<PHINode*, 8> IVs;
  bool AllCanonical = true;
  for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
    PHINode *PN = cast<PHINode>(II);
    const SCEVAddRecExpr *PNSCEV = dyn_cast<const SCEVAddRecExpr>(SE.getSCEV(PN));
    assert(PNSCEV && "PHINode did not have corresponding SCEVAddRecExpr.");
    assert(PNSCEV->getStart()->isZero() && "PHINode SCEV does not start at 0.");
    DEBUG(dbgs() << "LS step recurrence for SCEV " << *PNSCEV << " is " << *(PNSCEV->getStepRecurrence(SE)) << "\n");
    assert(PNSCEV->getStepRecurrence(SE)->isOne() && "PHINode SCEV step is not 1.");
    if (ConstantInt *C = dyn_cast<ConstantInt>(PN->getIncomingValueForBlock(Preheader))) {
      if (C->isZero())
        IVs.push_back(PN);
    } else {
      AllCanonical = false;
      DEBUG(dbgs() << "Remaining non-canonical PHI Node found: " << *PN << "\n");
    }
  }
  if (!AllCanonical)
    return false;

  // Insert the computation for the loop limit into the Preheader.
  Value *LimitVar = Exp.expandCodeFor(Limit, Limit->getType(),
                                      &(Preheader->front()));
  DEBUG(dbgs() << "LimitVar: " << *LimitVar << "\n");
  // dbgs() << "EL Preheader after adding Limit:" << *Preheader;
  // dbgs() << "EL Header after adding Limit:" << *Header;
  // dbgs() << "EL Latch after adding Limit:" << *Latch;

  // Canonicalize the loop latch.
  // dbgs() << "Loop backedge guarded by " << *(SE.getSCEV(CanonicalIV)) << " < " << *Limit <<
  //    ": " << SE.isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT, SE.getSCEV(CanonicalIV), Limit) << "\n";
  assert(SE.isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT, SE.getSCEV(CanonicalIV), Limit) &&
         "Loop backedge is not guarded by canonical comparison with limit.");
  Value *NewCond = canonicalizeLoopLatch(L, CanonicalIV, LimitVar);

  // dbgs() << "EL Preheader after new Latch:" << *Preheader;
  // dbgs() << "EL Header after new Latch:" << *Header;
  // dbgs() << "EL Latch after new Latch:" << *Latch;

  // Insert computation of grainsize into the Preheader.
  // For debugging.
  Value *GrainVar = ConstantInt::get(Limit->getType(), 2);
  // Value *GrainVar = computeGrainsize(L, LimitVar);
  DEBUG(dbgs() << "GrainVar: " << *GrainVar << "\n");
  // dbgs() << "EL Preheader after adding grainsize computation:" << *Preheader;
  // dbgs() << "EL Header after adding grainsize computation:" << *Header;
  // dbgs() << "EL Latch after adding grainsize computation:" << *Latch;

  /// Clone the loop into a new function.

  // Get the inputs and outputs for the Loop blocks.
  SetVector<Value*> Inputs, Outputs;
  SetVector<Value*> BodyInputs, BodyOutputs;
  ValueToValueMapTy VMap, InputMap;
  // Add start iteration, end iteration, and grainsize to inputs.
  {
    // Get the inputs and outputs for the loop body.
    CodeExtractor Ext(L->getBlocks(), DT);
    Ext.findInputsOutputs(BodyInputs, BodyOutputs);

    // Add argument for start of CanonicalIV.
    Value *CanonicalIVInput = CanonicalIV->getIncomingValueForBlock(Preheader);
    // CanonicalIVInput should be the constant 0.
    assert(isa<Constant>(CanonicalIVInput) &&
           "Input to canonical IV from preheader is not constant.");
    Argument *StartArg = new Argument(CanonicalIV->getType(),
                                      CanonicalIV->getName()+".start");
    Inputs.insert(StartArg);
    InputMap[CanonicalIV] = StartArg;

    // for (PHINode *IV : IVs) {
    //   Value *IVInput = IV->getIncomingValueForBlock(Preheader);
    //   if (isa<Constant>(IVInput)) {
    //     Argument *StartArg = new Argument(IV->getType(), IV->getName()+".start");
    //     Inputs.insert(StartArg);
    //     InputMap[IV] = StartArg;
    //   } else {
    //     assert(BodyInputs.count(IVInput) &&
    //            "Non-constant input to IV not captured.");
    //     Inputs.insert(IVInput);
    //     InputMap[IV] = IVInput;
    //   }
    // }

    // Add argument for end.
    if (isa<Constant>(LimitVar)) {
      Argument *EndArg = new Argument(LimitVar->getType(), "end");
      Inputs.insert(EndArg);
      InputMap[LimitVar] = EndArg;
      // if (!isa<Constant>(LimitVar))
      //   VMap[LimitVar] = EndArg;
    } else {
      Inputs.insert(LimitVar);
      InputMap[LimitVar] = LimitVar;
    }

    // Add argument for grainsize.
    // Inputs.insert(GrainVar);
    if (isa<Constant>(GrainVar)) {
      Argument *GrainArg = new Argument(GrainVar->getType(), "grainsize");
      Inputs.insert(GrainArg);
      InputMap[GrainVar] = GrainArg;
    } else {
      Inputs.insert(GrainVar);
      InputMap[GrainVar] = GrainVar;
    }

    // Put all of the inputs together, and clear redundant inputs from
    // the set for the loop body.
    SmallVector<Value*, 8> BodyInputsToRemove;
    for (Value *V : BodyInputs)
      if (!Inputs.count(V))
        Inputs.insert(V);
      else
        BodyInputsToRemove.push_back(V);
    for (Value *V : BodyInputsToRemove)
      BodyInputs.remove(V);
    assert(0 == BodyOutputs.size() &&
           "All results from parallel loop should be passed by memory already.");
  }
  DEBUG({
      for (Value *V : Inputs)
        dbgs() << "EL input: " << *V << "\n";
      // for (Value *V : Outputs)
      //        dbgs() << "EL output: " << *V << "\n";
    });


  Function *Helper;
  {
    SmallVector<ReturnInst *, 4> Returns;  // Ignore returns cloned.
  
    // LowerDbgDeclare(*(Header->getParent()));

    if (DISubprogram *SP = Header->getParent()->getSubprogram()) {
      // If we have debug info, add mapping for the metadata nodes that should not
      // be cloned by CloneFunctionInto.
      auto &MD = VMap.MD();
      // dbgs() << "SP: " << *SP << "\n";
      // dbgs() << "MD:\n";
      // for (auto &Tmp : MD) {
      //   dbgs() << *(Tmp.first) << " -> " << *(Tmp.second) << "\n";
      // }
      MD[SP->getUnit()].reset(SP->getUnit());
      MD[SP->getType()].reset(SP->getType());
      MD[SP->getFile()].reset(SP->getFile());
    }
    // {
    //   dbgs() << "LS TEST CLONE\n";
    //   ValueToValueMapTy VMap2;
    //   if (DISubprogram *SP = Header->getParent()->getSubprogram()) {
    //     // If we have debug info, add mapping for the metadata nodes that should not
    //     // be cloned by CloneFunctionInto.
    //     auto &MD = VMap2.MD();
    //     dbgs() << "SP: " << *SP << "\n";
    //     for (auto &Tmp : MD) {
    //       dbgs() << *(Tmp.first) << " -> " << *(Tmp.second) << "\n";
    //     }
    //     MD[SP->getUnit()].reset(SP->getUnit());
    //     MD[SP->getType()].reset(SP->getType());
    //     MD[SP->getFile()].reset(SP->getFile());
    //   }
    //   Function *TestFunc = CloneFunction(Header->getParent(), VMap2, nullptr);
    // }

    Helper = CreateHelper(Inputs, Outputs, L->getBlocks(),
                          Header, Preheader, L->getExitBlock(),
                          VMap, Header->getParent()->getParent(),
                          /*ModuleLevelChanges=*/false, Returns, ".l2c",
                          nullptr, nullptr, nullptr);

    assert(Returns.empty() && "Returns cloned when cloning loop.");

    // // If we have debug info, update it.  "ModuleLevelChanges = true"
    // // above does the heavy lifting, we just need to repoint
    // // subprogram at the same DICompileUnit as the original function.
    // //
    // // TODO: Cleanup the template parameters in
    // // Helper->getSubprogram() to reflect the variables used in the
    // // helper.
    // if (DISubprogram *SP = Header->getParent()->getSubprogram())
    //   Helper->getSubprogram()->replaceUnit(SP->getUnit());

    // Use a fast calling convention for the helper.
    Helper->setCallingConv(CallingConv::Fast);
    // Helper->setCallingConv(Header->getParent()->getCallingConv());
  }

  // Add a sync to the helper's return.
  {
    BasicBlock *HelperExit = cast<BasicBlock>(VMap[L->getExitBlock()]);
    assert(isa<ReturnInst>(HelperExit->getTerminator()));
    BasicBlock *NewHelperExit = SplitBlock(HelperExit,
                                           HelperExit->getTerminator(),
                                           DT, LI);
    IRBuilder<> Builder(&(HelperExit->front()));
    SyncInst *NewSync = Builder.CreateSync(NewHelperExit);
    // Set debug info of new sync.
    if (isa<SyncInst>(L->getExitBlock()->getTerminator()))
      NewSync->setDebugLoc(L->getExitBlock()->getTerminator()->getDebugLoc());
    else
      DEBUG(dbgs() << "Sync not found in loop exit block.\n");
    RemapInstruction(NewSync, VMap, RF_None | RF_IgnoreMissingLocals);
    HelperExit->getTerminator()->eraseFromParent();
  }

  BasicBlock *NewPreheader = cast<BasicBlock>(VMap[Preheader]);
  PHINode *NewCanonicalIV = cast<PHINode>(VMap[CanonicalIV]);

  // Rewrite the cloned IV's to start at the start iteration argument.
  {
    // Rewrite clone of canonical IV to start at the start iteration
    // argument.
    Argument *NewCanonicalIVStart = cast<Argument>(VMap[InputMap[CanonicalIV]]);
    {
      int NewPreheaderIdx = NewCanonicalIV->getBasicBlockIndex(NewPreheader);
      assert(isa<Constant>(NewCanonicalIV->getIncomingValue(NewPreheaderIdx)) &&
             "Cloned canonical IV does not inherit a constant value from cloned preheader.");
      NewCanonicalIV->setIncomingValue(NewPreheaderIdx, NewCanonicalIVStart);
    }

    // Rewrite other cloned IV's to start at their value at the start
    // iteration.
    const SCEV *StartIterSCEV = SE.getSCEV(NewCanonicalIVStart);
    DEBUG(dbgs() << "StartIterSCEV: " << *StartIterSCEV << "\n");
    for (PHINode *IV : IVs) {
      if (CanonicalIV == IV) continue;

      // Get the value of the IV at the start iteration.
      DEBUG(dbgs() << "IV " << *IV);
      const SCEV *IVSCEV = SE.getSCEV(IV);
      DEBUG(dbgs() << " (SCEV " << *IVSCEV << ")");
      const SCEVAddRecExpr *IVSCEVAddRec = cast<const SCEVAddRecExpr>(IVSCEV);
      const SCEV *IVAtIter = IVSCEVAddRec->evaluateAtIteration(StartIterSCEV, SE);
      DEBUG(dbgs() << " expands at iter " << *StartIterSCEV <<
            " to " << *IVAtIter << "\n");

      // NOTE: Expanded code should not refer to other IV's.
      Value *IVStart = Exp.expandCodeFor(IVAtIter, IVAtIter->getType(),
                                         NewPreheader->getTerminator());


      // Set the value that the cloned IV inherits from the cloned preheader.
      PHINode *NewIV = cast<PHINode>(VMap[IV]);
      int NewPreheaderIdx = NewIV->getBasicBlockIndex(NewPreheader);
      assert(isa<Constant>(NewIV->getIncomingValue(NewPreheaderIdx)) &&
             "Cloned IV does not inherit a constant value from cloned preheader.");
      NewIV->setIncomingValue(NewPreheaderIdx, IVStart);
    }

    // Remap the newly added instructions in the new preheader to use
    // values local to the helper.
    // dbgs() << "NewPreheader:" << *NewPreheader;
    for (Instruction &II : *NewPreheader)
      RemapInstruction(&II, VMap, RF_IgnoreMissingLocals,
                       /*TypeMapper=*/nullptr, /*Materializer=*/nullptr);
  }
  
  // // Rewrite cloned IV's to start at the start iteration argument.
  // BasicBlock *NewPreheader = cast<BasicBlock>(VMap[Preheader]);
  // for (PHINode *IV : IVs) {
  //   PHINode *NewIV = cast<PHINode>(VMap[IV]);
  //   int NewPreheaderIdx = NewIV->getBasicBlockIndex(NewPreheader);
  //   if (isa<Constant>(NewIV->getIncomingValue(NewPreheaderIdx)))
  //     NewIV->setIncomingValue(NewPreheaderIdx, cast<Value>(VMap[InputMap[IV]]));
  // }
  
  // If the loop limit is constant, then rewrite the loop latch
  // condition to use the end-iteration argument.
  if (isa<Constant>(LimitVar)) {
    CmpInst *HelperCond = cast<CmpInst>(VMap[NewCond]);
    assert(HelperCond->getOperand(1) == LimitVar);
    IRBuilder<> Builder(HelperCond);
    Value *NewHelperCond = Builder.CreateICmpULT(HelperCond->getOperand(0),
                                                 VMap[InputMap[LimitVar]]);
    HelperCond->replaceAllUsesWith(NewHelperCond);
    HelperCond->eraseFromParent();
  }

  // For debugging:
  // BasicBlock *NewHeader = cast<BasicBlock>(VMap[Header]);
  // SerializeDetachedCFG(cast<DetachInst>(NewHeader->getTerminator()), nullptr);
  implementDACIterSpawnOnHelper(Helper, NewPreheader,
                                cast<BasicBlock>(VMap[Header]),
                                cast<PHINode>(VMap[CanonicalIV]),
                                cast<Argument>(VMap[InputMap[LimitVar]]),
                                cast<Argument>(VMap[InputMap[GrainVar]]),
                                /*DT=*/nullptr, /*LI=*/nullptr,
                                CanonicalSCEV->getNoWrapFlags(SCEV::FlagNUW),
                                CanonicalSCEV->getNoWrapFlags(SCEV::FlagNSW));

  // TODO: Replace this once the function is complete.
  // dbgs() << "New Helper function:" << *Helper;
  if (llvm::verifyFunction(*Helper, &dbgs()))
    return false;

  // Add call to helper function.
  {
    // Setup arguments for call.
    SetVector<Value*> TopCallArgs;
    // Add start iteration 0.
    assert(CanonicalSCEV->getStart()->isZero() &&
           "Canonical IV does not start at zero.");
    TopCallArgs.insert(ConstantInt::get(CanonicalIV->getType(), 0));
    // Add loop limit.
    TopCallArgs.insert(LimitVar);
    // Add grainsize.
    TopCallArgs.insert(GrainVar);
    // Add the rest of the arguments.
    for (Value *V : BodyInputs)
      TopCallArgs.insert(V);

    // Create call instruction.
    IRBuilder<> Builder(Preheader->getTerminator());
    CallInst *TopCall = Builder.CreateCall(Helper, TopCallArgs.getArrayRef());
    // Use a fast calling convention for the helper.
    TopCall->setCallingConv(CallingConv::Fast);
    // TopCall->setCallingConv(Helper->getCallingConv());
    TopCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
  }
  
  ++LoopsConvertedToDAC;

  return Helper;
}

/// Checks if this loop is a Tapir loop.  Right now we check that the loop is
/// in a canonical form:
/// 1) The header detaches the body.
/// 2) The loop contains a single latch.
/// 3) The loop contains a single exit block.
/// 4) The body reattaches to the latch (which is necessary for a valid
///    detached CFG).
/// 5) The loop only branches to the exit block from the header or the latch.
bool LoopSpawningPass::isTapirLoop(const Loop *L) {
  const BasicBlock *Header = L->getHeader();
  const BasicBlock *Latch = L->getLoopLatch();
  const BasicBlock *Exit = L->getExitBlock();

  // dbgs() << "LS checking structure of " << *L;

  // Header must be terminated by a detach.
  if (!isa<DetachInst>(Header->getTerminator())) {
    DEBUG(dbgs() << "LS Loop header is not terminated by a detach.\n");
    return false;
  }

  // Loop must have a unique latch.
  if (nullptr == Latch) {
    DEBUG(dbgs() << "LS Loop does not have a unique latch.\n");
    return false;
  }

  // Loop must have a unique exit block.
  if (nullptr == Exit) {
    DEBUG(dbgs() << "LS Loop does not have a unique exit block.\n");
    return false;
  }

  // Continuation of header terminator must be the latch.
  const DetachInst *HeaderDetach = cast<DetachInst>(Header->getTerminator());
  const BasicBlock *Continuation = HeaderDetach->getContinue();
  if (Continuation != Latch) {
    DEBUG(dbgs() << "LS Continuation of detach in header is not the latch.\n");
    return false;
  }

  // All other predecessors of Latch are terminated by reattach instructions.
  for (auto PI = pred_begin(Latch), PE = pred_end(Latch);  PI != PE; ++PI) {
    const BasicBlock *Pred = *PI;
    if (Header == Pred) continue;
    if (!isa<ReattachInst>(Pred->getTerminator())) {
      DEBUG(dbgs() << "LS Latch has a predecessor that is not terminated "
                   << "by a reattach.\n");
      return false;
    }
  }

  // The only predecessors of Exit inside the loop are Header and Latch.
  for (auto PI = pred_begin(Exit), PE = pred_end(Exit);  PI != PE; ++PI) {
    const BasicBlock *Pred = *PI;
    if (!L->contains(Pred))
      continue;
    if (Header != Pred && Latch != Pred) {
      DEBUG(dbgs() << "LS Loop branches to exit block from a block "
                   << "other than the header or latch.\n");
      return false;
    }
  }

  return true;
}

void LoopSpawningPass::addTapirLoop(Loop *L, SmallVectorImpl<Loop *> &V) {
  if (isTapirLoop(L)) {
    V.push_back(L);
    return;
  }
  for (Loop *InnerL : *L)
    addTapirLoop(InnerL, V);
}

#ifndef NDEBUG
/// \return string containing a file name and a line # for the given loop.
static std::string getDebugLocString(const Loop *L) {
  std::string Result;
  if (L) {
    raw_string_ostream OS(Result);
    if (const DebugLoc LoopDbgLoc = L->getStartLoc())
      LoopDbgLoc.print(OS);
    else
      // Just print the module name.
      OS << L->getHeader()->getParent()->getParent()->getModuleIdentifier();
    OS.flush();
  }
  return Result;
}
#endif

bool LoopSpawningPass::processLoop(Loop *L) {
#ifndef NDEBUG
  const std::string DebugLocStr = getDebugLocString(L);
#endif /* NDEBUG */

  DEBUG(dbgs() << "\nLS: Checking a Tapir loop in \""
               << L->getHeader()->getParent()->getName() << "\" from "
               << DebugLocStr << "\n");

  LoopSpawningHints Hints(L, *ORE);

  DEBUG(dbgs() << "LS: Loop hints:"
               << " strategy = " << Hints.printStrategy(Hints.getStrategy())
               << "\n");

  // Function containing loop
  Function *F = L->getHeader()->getParent();

  if (LoopSpawningHints::ST_NONE == Hints.getStrategy()) {
    dbgs() << "LS: Loop hints prevent conversion.\n";
    // DEBUG(dbgs() << "LS: Loop hints prevent conversion.\n");
    // return false;
  } else {
    dbgs() << "LS: " << *L << " with hint " << LoopSpawningHints::printStrategy(Hints.getStrategy()) << "\n";
  }

  // Fix-up loop preheader.
  BasicBlock *Preheader = L->getLoopPreheader();
  if (nullptr == Preheader) {
    DEBUG(dbgs() << "LS: Loop lacks a preheader.\n");
  }
  if (isa<SyncInst>(Preheader->getTerminator())) {
    BasicBlock *Header = L->getHeader();
    SplitEdge(Preheader, Header, DT, LI);
    // Report a change, and let a subsequent run of this pass deal with the
    // conversion itself.
    return true;
  }
  if (!isa<BranchInst>(Preheader->getTerminator())) {
    DEBUG(dbgs() << "LS: Loop preheader is not terminated by a branch.\n");
    return false;
  }

  // TODO: Switch conversion strategies based on hints.

  DACLoopSpawning DLS(L, *SE, LI, DT, TLI, TTI, ORE);

  if (DLS.convertLoop()) {
    // // Mark the loop as already vectorized to avoid vectorizing again.
    // Hints.setAlreadyVectorized();

    DEBUG(verifyFunction(*L->getHeader()->getParent()));
    return true;
  }
  return false;
}

bool LoopSpawningPass::runImpl(
    Function &F, ScalarEvolution &SE_, LoopInfo &LI_, TargetTransformInfo &TTI_,
    DominatorTree &DT_, TargetLibraryInfo *TLI_,
    AliasAnalysis &AA_, AssumptionCache &AC_,
    OptimizationRemarkEmitter &ORE_) {

  SE = &SE_;
  LI = &LI_;
  TTI = &TTI_;
  DT = &DT_;
  TLI = TLI_;
  AA = &AA_;
  AC = &AC_;
  ORE = &ORE_;

  // Build up a worklist of inner-loops to vectorize. This is necessary as
  // the act of vectorizing or partially unrolling a loop creates new loops
  // and can invalidate iterators across the loops.
  SmallVector<Loop *, 8> Worklist;

  for (Loop *L : *LI)
    addTapirLoop(L, Worklist);

  LoopsAnalyzed += Worklist.size();

  // Now walk the identified inner loops.
  bool Changed = false;
  while (!Worklist.empty())
    Changed |= processLoop(Worklist.pop_back_val());

  // Process each loop nest in the function.
  return Changed;

}

PreservedAnalyses LoopSpawningPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
    auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
    auto &LI = AM.getResult<LoopAnalysis>(F);
    auto &TTI = AM.getResult<TargetIRAnalysis>(F);
    auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
    auto *TLI = AM.getCachedResult<TargetLibraryAnalysis>(F);
    auto &AA = AM.getResult<AAManager>(F);
    auto &AC = AM.getResult<AssumptionAnalysis>(F);
    auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

    bool Changed =
        runImpl(F, SE, LI, TTI, DT, TLI, AA, AC, ORE);
    if (!Changed)
      return PreservedAnalyses::all();
    PreservedAnalyses PA;
    return PA;
}

char LoopSpawning::ID = 0;
// static RegisterPass<LoopSpawning> X(LS_NAME, "Transform Tapir loops to spawn iterations efficiently", false, false);
static const char ls_name[] = "Loop Spawning";
INITIALIZE_PASS_BEGIN(LoopSpawning, LS_NAME, ls_name, false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LCSSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(LoopSpawning, LS_NAME, ls_name, false, false)

namespace llvm {
Pass *createLoopSpawningPass() {
  return new LoopSpawning();
}
}


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////



// namespace {
//   class Loop2Cilk : public LoopPass {

//   public:
//     static char ID; // Pass identification, replacement for typeid
//     Loop2Cilk() : LoopPass(ID) { }

//     bool runOnLoop(Loop *L, LPPassManager &LPM) override;
//     bool performDAC(Loop *L, LPPassManager &LPM);

//     void getAnalysisUsage(AnalysisUsage &AU) const override {
//       AU.addRequiredID(LoopSimplifyID);
//       AU.addRequiredID(LCSSAID);
//       AU.addRequired<DominatorTreeWrapperPass>();
//       AU.addRequired<LoopInfoWrapperPass>();
//       AU.addRequired<ScalarEvolutionWrapperPass>();
//     }

//   protected:
//     bool verifyLoopExit(const Loop *L);
//     bool verifyLoopStructureForConversion(const Loop *L);
//     PHINode* canonicalizeIVs(Loop *L, Type *Ty, ScalarEvolution &SE, DominatorTree &DT);
//     Value* canonicalizeLoopLatch(Loop *L, PHINode *IV, Value *Limit);
//     Value* computeGrainsize(Loop *L, Value *Limit);
//     void implementDACIterSpawnOnHelper(Function *Helper,
//                                        BasicBlock *Preheader,
//                                        BasicBlock *Header,
//                                        PHINode *CanonicalIV,
//                                        Argument *Limit,
//                                        Argument *Grainsize,
//                                        DominatorTree *DT,
//                                        LoopInfo *LI,
//                                        bool CanonicalIVFlagNUW = false,
//                                        bool CanonicalIVFlagNSW = false);
//     void eraseLoop(Loop *L, ScalarEvolution &SE, DominatorTree &DT, LoopInfo &LoopInfo);
//     Function* convertLoopToDACIterSpawn(Loop *L, ScalarEvolution &SE,
//                                         DominatorTree &DT, LoopInfo &LI);
    
//   private:
//     void releaseMemory() override { }
//   };
// }

// /// Verify that the CFG between the loop exit and the subsequent sync is
// /// appropriately simple -- essentially, a straight sequence of basic blocks
// /// that unconditionally branch to the sync.  Return false if this process
// /// fails, indicating that this loop is not a simple loop for conversion.
// ///
// /// This method only checks the structure of the loop exit, rather than modify
// /// it, to preserve analyses.
// bool Loop2Cilk::verifyLoopExit(const Loop *L) {
//   BasicBlock *LoopExit = L->getExitBlock();
//   SmallPtrSet<BasicBlock *, 8> Visited;
//   Visited.insert(LoopExit);
//   // Attempt to remove empty blocks terminated by unconditional
//   // branches that follow the loop exit until the loop exit is empty
//   // aside from a sync.
//   do {
//     // LoopExit = L->getExitBlock();
//     // dbgs() << "verifyLoopExit: Checking block " << *LoopExit;
//     // // Fail if the loop exit has multiple predecessors.
//     // if (!LoopExit->getUniquePredecessor()) {
//     //   dbgs() << "L2C loop exit does not have a unique predecessor.\n";
//     //   return false;
//     // }

//     // Fail if the loop exit contains non-terminator instructions
//     // other than PHI nodes or debug instructions.
//     if (LoopExit->getFirstNonPHIOrDbg() != LoopExit->getTerminator()) {
//       dbgs() << "L2C loop exit contains non-terminator instructions other than PHI nodes and debug instructions.\n";
//       return false;
//     }

//     // If the loop exit is terminated by a sync, we're done.
//     if (isa<SyncInst>(LoopExit->getTerminator()))
//       return true;

//     // Check if the loop exit is terminated by an unconditional branch.
//     if (BranchInst *Br = dyn_cast_or_null<BranchInst>(LoopExit->getTerminator())) {
//       if (Br->isConditional()) {
//         dbgs() << "L2C loop exit is terminated by a conditional branch.\n";
//         return false;
//       }
//     } else {
//       dbgs() << "L2C loop exit is not terminated by a branch.\n";
//       return false;
//     }
//     LoopExit = LoopExit->getTerminator()->getSuccessor(0);
//   } while (Visited.insert(LoopExit).second);
//   // } while (TryToSimplifyUncondBranchFromEmptyBlock(LoopExit));

//   dbgs() << "L2C could not confirm simple exit to sync.\n";
//   return false;
// }

// /// Verify assumptions on the structure of the Loop:
// /// 1) Loop has a single preheader, latch, and exit.
// /// 2) Loop header is terminated by a detach.
// /// 3) Continuation of detach in header is the latch.
// /// 4) All other predecessors of the latch are terminated by reattach
// /// instructions.
// bool Loop2Cilk::verifyLoopStructureForConversion(const Loop *L) {
//   const BasicBlock *Header = L->getHeader();
//   const BasicBlock *Preheader = L->getLoopPreheader();
//   const BasicBlock *Latch = L->getLoopLatch();
//   const BasicBlock *Exit = L->getExitBlock();

//   // dbgs() << "L2C checking structure of " << *L;

//   // Header must be terminated by a detach.
//   if (!isa<DetachInst>(Header->getTerminator())) {
//     DEBUG(dbgs() << "L2C Loop header is not terminated by a detach.\n");
//     return false;
//   }

//   // Loop must have a single preheader.
//   if (nullptr == Preheader) {
//     DEBUG(dbgs() << "L2C Loop preheader not found.\n");
//     return false;
//   }

//   // Loop must have a unique latch.
//   if (nullptr == Latch) {
//     DEBUG(dbgs() << "L2C Loop does not have a unique latch.\n");
//     return false;
//   }

//   // Loop must have a unique exit block.
//   if (nullptr == Exit) {
//     DEBUG(dbgs() << "L2C Loop does not have a unique exit block.\n");
//     return false;
//   }

//   // Continuation of header terminator must be the latch.
//   const DetachInst *HeaderDetach = cast<DetachInst>(Header->getTerminator());
//   const BasicBlock *Continuation = HeaderDetach->getContinue();
//   if (Continuation != Latch) {
//     DEBUG(dbgs() << "L2C Continuation of detach in header is not the latch.\n");
//     return false;
//   }

//   // All other predecessors of Latch are terminated by reattach instructions.
//   for (auto PI = pred_begin(Latch), PE = pred_end(Latch);  PI != PE; ++PI) {
//     const BasicBlock *Pred = *PI;
//     if (Header == Pred) continue;
//     if (!isa<ReattachInst>(Pred->getTerminator())) {
//       DEBUG(dbgs() << "L2C Latch has a predecessor that is not terminated by a reattach.\n");
//       return false;
//     }
//   }

//   return verifyLoopExit(L);
// }

// /// Canonicalize the induction variables in the loop L.  Return the canonical
// /// induction variable created or inserted by the scalar evolution expander..
// PHINode* Loop2Cilk::canonicalizeIVs(Loop *L, Type *Ty, ScalarEvolution &SE, DominatorTree &DT) {
//   BasicBlock* Header = L->getHeader();
//   Module* M = Header->getParent()->getParent();

//   DEBUG(dbgs() << "L2C Header:" << *Header);
//   BasicBlock *Latch = L->getLoopLatch();
//   DEBUG(dbgs() << "L2C Latch:" << *Latch);

//   DEBUG(dbgs() << "L2C exiting block:" << *(L->getExitingBlock()));

//   // dbgs() << "L2C SE trip count: " << SE->getSmallConstantTripCount(L, L->getExitingBlock()) << "\n";
//   // dbgs() << "L2C SE trip multiple: " << SE->getSmallConstantTripMultiple(L, L->getExitingBlock()) << "\n";
//   DEBUG(dbgs() << "L2C SE backedge taken count: " << *(SE.getBackedgeTakenCount(L)) << "\n");
//   DEBUG(dbgs() << "L2C SE max backedge taken count: " << *(SE.getMaxBackedgeTakenCount(L)) << "\n");
//   DEBUG(dbgs() << "L2C SE exit count: " << *(SE.getExitCount(L, L->getExitingBlock())) << "\n");

//   SCEVExpander Exp(SE, M->getDataLayout(), "l2c");

//   PHINode *CanonicalIV = Exp.getOrInsertCanonicalInductionVariable(L, Ty);
//   DEBUG(dbgs() << "L2C Canonical induction variable " << *CanonicalIV << "\n");

//   SmallVector<WeakVH, 16> DeadInsts;
//   Exp.replaceCongruentIVs(L, &DT, DeadInsts);
//   // dbgs() << "Updated header:" << *(L->getHeader());
//   // dbgs() << "Updated exiting block:" << *(L->getExitingBlock());
//   for (WeakVH V : DeadInsts) {
//     DEBUG(dbgs() << "L2C erasing dead inst " << *V << "\n");
//     Instruction *I = cast<Instruction>(V);
//     I->eraseFromParent();
//   }

//   return CanonicalIV;
// }

// /// \brief Replace the Latch of Loop L to check that IV is always less
// /// than or equal to Limit.
// ///
// /// This method assumes that L has a single loop latch and a single
// /// exit block.
// Value* Loop2Cilk::canonicalizeLoopLatch(Loop *L, PHINode *IV, Value *Limit) {
//   Value *NewCondition;
//   BasicBlock *Header = L->getHeader();
//   BasicBlock *Latch = L->getLoopLatch();
//   assert(Latch && "No single loop latch found for loop.");

//   IRBuilder<> Builder(&*Latch->getFirstInsertionPt());

//   // This process assumes that IV's increment is in Latch.

//   // Create comparison between IV and Limit at top of Latch.
//   NewCondition = Builder.CreateICmpULT(IV, Limit);

//   // Replace the conditional branch at the end of Latch.
//   BranchInst *LatchBr = dyn_cast_or_null<BranchInst>(Latch->getTerminator());
//   assert(LatchBr && LatchBr->isConditional() &&
//          "Latch does not terminate with a conditional branch.");
//   Builder.SetInsertPoint(Latch->getTerminator());
//   assert(L->getExitBlock() &&
//          "Loop does not have a single exit block.");
//   Builder.CreateCondBr(NewCondition, Header, L->getExitBlock());

//   // Erase the old conditional branch.
//   LatchBr->eraseFromParent();

//   return NewCondition;
// }

// /// \brief Compute the grainsize of Loop L, based on Limit.
// ///
// /// The grainsize is computed by the following equation:
// ///
// ///     Grainsize = min(2048, ceil(Limit / (8 * workers)))
// ///
// /// This computation is inserted into the Preheader of the Loop L.
// Value* Loop2Cilk::computeGrainsize(Loop *L, Value *Limit) {
//   Value *Grainsize;
//   BasicBlock *Preheader = L->getLoopPreheader();
//   assert(Preheader && "No Preheader found for loop.");

//   IRBuilder<> Builder(Preheader->getTerminator());

//   // Get 8 * workers
//   Value *Workers8 = Builder.CreateIntCast(cilk::GetOrCreateWorker8(*Preheader->getParent()),
//                                           Limit->getType(), false);
//   // Compute ceil(limit / 8 * workers) = (limit + 8 * workers - 1) / (8 * workers)
//   Value *SmallLoopVal =
//     Builder.CreateUDiv(Builder.CreateSub(Builder.CreateAdd(Limit, Workers8),
//                                          ConstantInt::get(Limit->getType(), 1)),
//                        Workers8);
//   // Compute min
//   Value *LargeLoopVal = ConstantInt::get(Limit->getType(), 2048);
//   Value *Cmp = Builder.CreateICmpULT(LargeLoopVal, SmallLoopVal);
//   Grainsize = Builder.CreateSelect(Cmp, LargeLoopVal, SmallLoopVal);

//   return Grainsize;
// }

// /// \brief Method to help convertLoopToDACIterSpawn convert the Tapir
// /// loop cloned into function Helper to spawn its iterations in a
// /// parallel divide-and-conquer fashion.
// ///
// /// Example: Suppose that Helper contains the following Tapir loop:
// ///
// /// Helper(iter_t start, iter_t end, iter_t grain, ...) {
// ///   iter_t i = start;
// ///   ... Other loop setup ...
// ///   do {
// ///     spawn { ... loop body ... };
// ///   } while (i++ < end);
// ///   sync;
// /// }
// ///
// /// Then this method transforms Helper into the following form:
// ///
// /// Helper(iter_t start, iter_t end, iter_t grain, ...) {
// /// recur:
// ///   iter_t itercount = end - start;
// ///   if (itercount > grain) {
// ///     // Invariant: itercount >= 2
// ///     count_t miditer = start + itercount / 2;
// ///     spawn Helper(start, miditer, grain, ...);
// ///     start = miditer + 1;
// ///     goto recur;
// ///   }
// ///
// ///   iter_t i = start;
// ///   ... Other loop setup ...
// ///   do {
// ///     ... Loop Body ...
// ///   } while (i++ < end);
// ///   sync;
// /// }
// ///
// void Loop2Cilk::implementDACIterSpawnOnHelper(Function *Helper,
//                                               BasicBlock *Preheader,
//                                               BasicBlock *Header,
//                                               PHINode *CanonicalIV,
//                                               Argument *Limit,
//                                               Argument *Grainsize,
//                                               DominatorTree *DT,
//                                               LoopInfo *LI,
//                                               bool CanonicalIVFlagNUW,
//                                               bool CanonicalIVFlagNSW) {
//   // Serialize the cloned copy of the loop.
//   assert(Preheader->getParent() == Helper &&
//          "Preheader does not belong to helper function.");
//   assert(Header->getParent() == Helper &&
//          "Header does not belong to helper function.");
//   assert(CanonicalIV->getParent() == Header &&
//          "CanonicalIV does not belong to header");
//   assert(isa<DetachInst>(Header->getTerminator()) &&
//          "Cloned header is not terminated by a detach.");
//   DetachInst *DI = dyn_cast<DetachInst>(Header->getTerminator());
//   SerializeDetachedCFG(DI, DT);

//   // Convert the cloned loop into the strip-mined loop body.

//   BasicBlock *DACHead = Preheader;
//   if (&(Helper->getEntryBlock()) == Preheader)
//     // Split the entry block.  We'll want to create a backedge into
//     // the split block later.
//     DACHead = SplitBlock(Preheader, &(Preheader->front()), DT, LI);

//   BasicBlock *RecurHead, *RecurDet, *RecurCont;
//   Value *IterCount;
//   Value *CanonicalIVInput;
//   PHINode *CanonicalIVStart;
//   {
//     Instruction *PreheaderOrigFront = &(DACHead->front());
//     IRBuilder<> Builder(PreheaderOrigFront);
//     // Create branch based on grainsize.
//     DEBUG(dbgs() << "L2C CanonicalIV: " << *CanonicalIV << "\n");
//     CanonicalIVInput = CanonicalIV->getIncomingValueForBlock(DACHead);
//     CanonicalIVStart = Builder.CreatePHI(CanonicalIV->getType(), 2,
//                                          CanonicalIV->getName()+".dac");
//     CanonicalIVInput->replaceAllUsesWith(CanonicalIVStart);
//     IterCount = Builder.CreateSub(Limit, CanonicalIVStart,
//                                   "itercount");
//     Value *IterCountCmp = Builder.CreateICmpUGT(IterCount, Grainsize);
//     // dbgs() << "DAC head before split:" << *DACHead;
//     TerminatorInst *RecurTerm =
//       SplitBlockAndInsertIfThen(IterCountCmp, PreheaderOrigFront,
//                                 /*Unreachable=*/false,
//                                 /*BranchWeights=*/nullptr,
//                                 DT);
//     RecurHead = RecurTerm->getParent();
//     // Create skeleton of divide-and-conquer recursion:
//     // DACHead -> RecurHead -> RecurDet -> RecurCont -> DACHead
//     RecurDet = SplitBlock(RecurHead, RecurHead->getTerminator(),
//                           DT, LI);
//     RecurCont = SplitBlock(RecurDet, RecurDet->getTerminator(),
//                            DT, LI);
//     RecurCont->getTerminator()->replaceUsesOfWith(RecurTerm->getSuccessor(0),
//                                                   DACHead);
//     // Builder.SetInsertPoint(&(RecurCont->front()));
//     // Builder.CreateBr(DACHead);
//     // RecurCont->getTerminator()->eraseFromParent();
//   }

//   // Compute mid iteration in RecurHead.
//   Value *MidIter, *MidIterPlusOne;
//   {
//     IRBuilder<> Builder(&(RecurHead->front()));
//     MidIter = Builder.CreateAdd(CanonicalIVStart,
//                                 Builder.CreateLShr(IterCount, 1,
//                                                    "halfcount"),
//                                 "miditer",
//                                 CanonicalIVFlagNUW, CanonicalIVFlagNSW);
//   }

//   // Create recursive call in RecurDet.
//   {
//     // Create input array for recursive call.
//     IRBuilder<> Builder(&(RecurDet->front()));
//     SetVector<Value*> RecurInputs;
//     Function::arg_iterator AI = Helper->arg_begin();
//     assert(cast<Argument>(CanonicalIVInput) == &*AI &&
//            "First argument does not match original input to canonical IV.");
//     RecurInputs.insert(CanonicalIVStart);
//     ++AI;
//     assert(Limit == &*AI &&
//            "Second argument does not match original input to the loop limit.");
//     RecurInputs.insert(MidIter);
//     ++AI;
//     for (Function::arg_iterator AE = Helper->arg_end();
//          AI != AE;  ++AI)
//         RecurInputs.insert(&*AI);
//     // RecurInputs.insert(CanonicalIVStart);
//     // // for (PHINode *IV : IVs)
//     // //   RecurInputs.insert(DACStart[IV]);
//     // RecurInputs.insert(Limit);
//     // RecurInputs.insert(Grainsize);
//     // for (Value *V : BodyInputs)
//     //   RecurInputs.insert(VMap[V]);
//     DEBUG({
//         dbgs() << "RecurInputs: ";
//         for (Value *Input : RecurInputs)
//           dbgs() << *Input << ", ";
//         dbgs() << "\n";
//       });

//     // Create call instruction.
//     CallInst *RecurCall = Builder.CreateCall(Helper, RecurInputs.getArrayRef());
//     RecurCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
//     // Use a fast calling convention for the helper.
//     RecurCall->setCallingConv(CallingConv::Fast);
//     // RecurCall->setCallingConv(Helper->getCallingConv());
//   }

//   // Set up continuation of detached recursive call.  We effectively
//   // inline this tail call automatically.
//   {
//     IRBuilder<> Builder(&(RecurCont->front()));
//     MidIterPlusOne = Builder.CreateAdd(MidIter,
//                                        ConstantInt::get(Limit->getType(), 1),
//                                        "miditerplusone",
//                                        CanonicalIVFlagNUW,
//                                        CanonicalIVFlagNSW);
//   }

//   // Finish setup of new phi node for canonical IV.
//   {
//     CanonicalIVStart->addIncoming(CanonicalIVInput, Preheader);
//     CanonicalIVStart->addIncoming(MidIterPlusOne, RecurCont);
//   }

//   /// Make the recursive DAC parallel.
//   {
//     IRBuilder<> Builder(RecurHead->getTerminator());
//     // Create the detach.
//     Builder.CreateDetach(RecurDet, RecurCont);
//     RecurHead->getTerminator()->eraseFromParent();
//     // Create the reattach.
//     Builder.SetInsertPoint(RecurDet->getTerminator());
//     Builder.CreateReattach(RecurCont);
//     RecurDet->getTerminator()->eraseFromParent();
//   }
// }

// /// Recursive routine to remove a loop and all of its subloops.
// static void removeLoopAndAllSubloops(Loop *L, LoopInfo &LoopInfo) {
//   for (Loop *SL : L->getSubLoops())
//     removeLoopAndAllSubloops(SL, LoopInfo);

//   dbgs() << "Removing " << *L << "\n";

//   SmallPtrSet<BasicBlock *, 8> Blocks;
//   Blocks.insert(L->block_begin(), L->block_end());
//   for (BasicBlock *BB : Blocks)
//     LoopInfo.removeBlock(BB);

//   LoopInfo.markAsRemoved(L);
// }

// // Recursive routine to traverse the subloops of a loop and push all 
// static void collectLoopAndAllSubLoops(Loop *L,
//                                       SetVector<Loop*> &SubLoops) {
//   for (Loop *SL : L->getSubLoops())
//     collectLoopAndAllSubLoops(SL, SubLoops);
//   SubLoops.insert(L);
// }

// /// Recursive routine to mark a loop and all of its subloops as removed.
// static void markLoopAndAllSubloopsAsRemoved(Loop *L, LoopInfo &LoopInfo) {
//   for (Loop *SL : L->getSubLoops())
//     markLoopAndAllSubloopsAsRemoved(SL, LoopInfo);

//   dbgs() << "Marking as removed:" << *L << "\n";
//   LoopInfo.markAsRemoved(L);
// }

// /// Erase the specified loop, and update analysis accordingly.
// void Loop2Cilk::eraseLoop(Loop *L, ScalarEvolution &SE,
//                           DominatorTree &DT, LoopInfo &LoopInfo) {
//     // Get components of the old loop.
//     BasicBlock *Preheader = L->getLoopPreheader();
//     assert(Preheader && "Loop does not have a unique preheader.");
//     BasicBlock *ExitBlock = L->getExitBlock();
//     assert(ExitBlock && "Loop does not have a unique exit block.");
//     BasicBlock *ExitingBlock = L->getExitingBlock();
//     assert(ExitingBlock && "Loop does not have a unique exiting block.");

//     // Invalidate the analysis of the old loop.
//     SE.forgetLoop(L);

//     // Redirect the preheader to branch directly to loop exit.
//     assert(1 == Preheader->getTerminator()->getNumSuccessors() &&
//            "Preheader does not have a unique successor.");
//     Preheader->getTerminator()->replaceUsesOfWith(L->getHeader(),
//                                                   ExitBlock);
    
//     // Rewrite phis in the exit block cto get their inputs from
//     // the preheader instead of the exiting block.
//     BasicBlock::iterator BI = ExitBlock->begin();
//     while (PHINode *P = dyn_cast<PHINode>(BI)) {
//       int j = P->getBasicBlockIndex(ExitingBlock);
//       assert(j >= 0 && "Can't find exiting block in exit block's phi node!");
//       P->setIncomingBlock(j, Preheader);
//       P->removeIncomingValue(ExitingBlock);
//       ++BI;
//     }

//   // Update the dominator tree and remove the instructions and blocks that will
//   // be deleted from the reference counting scheme.
//   SmallVector<DomTreeNode*, 8> ChildNodes;
//   for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
//        LI != LE; ++LI) {
//     // Move all of the block's children to be children of the preheader, which
//     // allows us to remove the domtree entry for the block.
//     ChildNodes.insert(ChildNodes.begin(), DT[*LI]->begin(), DT[*LI]->end());
//     for (DomTreeNode *ChildNode : ChildNodes) {
//       DT.changeImmediateDominator(ChildNode, DT[Preheader]);
//     }

//     ChildNodes.clear();
//     DT.eraseNode(*LI);

//     // Remove the block from the reference counting scheme, so that we can
//     // delete it freely later.
//     (*LI)->dropAllReferences();
//   }

//   // Erase the instructions and the blocks without having to worry
//   // about ordering because we already dropped the references.
//   // NOTE: This iteration is safe because erasing the block does not remove its
//   // entry from the loop's block list.  We do that in the next section.
//   for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
//        LI != LE; ++LI)
//     (*LI)->eraseFromParent();

//   // Finally, the blocks from loopinfo.  This has to happen late because
//   // otherwise our loop iterators won't work.

//   SetVector<Loop *> SubLoops;
//   collectLoopAndAllSubLoops(L, SubLoops);
  
//   SmallPtrSet<BasicBlock *, 8> Blocks;
//   Blocks.insert(L->block_begin(), L->block_end());
//   for (BasicBlock *BB : Blocks)
//     LoopInfo.removeBlock(BB);

//   // The last step is to update LoopInfo now that we've eliminated this loop.
//   // for (Loop *SL : L->getSubLoops())
//   //   LoopInfo.markAsRemoved(SL);
//   // LoopInfo.markAsRemoved(L);
//   // markLoopAndAllSubloopsAsRemoved(L, LoopInfo);
//   // removeLoopAndAllSubloops(L, LoopInfo);
//   for (Loop *SL : SubLoops)
//     LoopInfo.markAsRemoved(SL);
// }

// /// Top-level call to convert loop to spawn its iterations in a
// /// divide-and-conquer fashion.
// Function* Loop2Cilk::convertLoopToDACIterSpawn(Loop *L, ScalarEvolution &SE,
//                                                DominatorTree &DT, LoopInfo &LI) {

//   // Verification already done in the caller.
//   // // Verify that we can extract loop.
//   // if (!verifyLoopStructureForConversion(L))
//   //   return nullptr;

//   ++LoopsAnalyzed;

//   BasicBlock *Header = L->getHeader();
//   BasicBlock *Preheader = L->getLoopPreheader();
//   BasicBlock *Latch = L->getLoopLatch();
//   Module* M = Header->getParent()->getParent();

//   DEBUG(dbgs() << "L2C loop header:" << *Header);
//   DEBUG(dbgs() << "L2C loop latch:" << *Latch);
//   DEBUG(dbgs() << "L2C exiting block:" << *(L->getExitingBlock()));

//   // dbgs() << "L2C SE trip count: " << SE->getSmallConstantTripCount(L, L->getExitingBlock()) << "\n";
//   // dbgs() << "L2C SE trip multiple: " << SE->getSmallConstantTripMultiple(L, L->getExitingBlock()) << "\n";
//   DEBUG(dbgs() << "L2C SE backedge taken count: " << *(SE.getBackedgeTakenCount(L)) << "\n");
//   DEBUG(dbgs() << "L2C SE max backedge taken count: " << *(SE.getMaxBackedgeTakenCount(L)) << "\n");
//   DEBUG(dbgs() << "L2C SE exit count: " << *(SE.getExitCount(L, L->getExitingBlock())) << "\n");

//   /// Get loop limit.
//   const SCEV *Limit = SE.getBackedgeTakenCount(L);
//   // const SCEV *Limit = SE.getAddExpr(BETC, SE.getOne(BETC->getType()));
//   DEBUG(dbgs() << "L2C Loop limit: " << *Limit << "\n");
//   // PredicatedScalarEvolution PSE(SE, *L);
//   // const SCEV *PLimit = PSE.getBackedgeTakenCount();
//   // DEBUG(dbgs() << "L2C predicated loop limit: " << *PLimit << "\n");
//   if (SE.getCouldNotCompute() == Limit) {
//     DEBUG(dbgs() << "SE could not compute loop limit.  Quitting extractLoop.\n");
//     return nullptr;
//   }

//   /// Clean up the loop's induction variables.
//   PHINode *CanonicalIV = canonicalizeIVs(L, Limit->getType(), SE, DT);
//   const SCEVAddRecExpr *CanonicalSCEV =
//     cast<const SCEVAddRecExpr>(SE.getSCEV(CanonicalIV));
//   // dbgs() << "[Loop2Cilk] Current loop preheader";
//   // dbgs() << *Preheader;
//   // dbgs() << "[Loop2Cilk] Current loop:";
//   // for (BasicBlock *BB : L->getBlocks()) {
//   //   dbgs() << *BB;
//   // }
//   if (!CanonicalIV) {
//     DEBUG(dbgs() << "Could not get canonical IV.  Quitting extractLoop.\n");
//     return nullptr;
//   }

//   // Remove all IV's other can CanonicalIV.
//   // First, check that we can do this.
//   bool CanRemoveIVs = true;
//   for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
//     PHINode *PN = cast<PHINode>(II);
//     if (CanonicalIV == PN) continue;
//     dbgs() << "IV " << *PN;
//     const SCEV *S = SE.getSCEV(PN);
//     dbgs() << " SCEV " << *S << "\n";
//     if (SE.getCouldNotCompute() == S)
//       CanRemoveIVs = false;
//   }

//   if (!CanRemoveIVs) {
//     DEBUG(dbgs() << "Could not compute SCEV for all IV's.  Quitting extractLoop.\n");
//     return nullptr;
//   }

//   ////////////////////////////////////////////////////////////////////////
//   // We now have everything we need to extract the loop.  It's time to
//   // do some surgery.

//   SCEVExpander Exp(SE, M->getDataLayout(), "l2c");

//   // Remove the IV's (other than CanonicalIV) and replace them with
//   // their stronger forms.
//   //
//   // TODO?: We can probably adapt this loop->DAC process such that we
//   // don't require all IV's to be canonical.
//   {
//     SmallVector<PHINode*, 8> IVsToRemove;
//     for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
//       PHINode *PN = cast<PHINode>(II);
//       if (PN == CanonicalIV) continue;
//       const SCEV *S = SE.getSCEV(PN);
//       Value *NewIV = Exp.expandCodeFor(S, S->getType(), CanonicalIV);
//       PN->replaceAllUsesWith(NewIV);
//       IVsToRemove.push_back(PN);
//     }
//     for (PHINode *PN : IVsToRemove)
//       PN->eraseFromParent();
//   }
//   // dbgs() << "EL Preheader after IV removal:" << *Preheader;
//   // dbgs() << "EL Header after IV removal:" << *Header;
//   // dbgs() << "EL Latch after IV removal:" << *Latch;

//   // All remaining IV's should be canonical.  Collect them.
//   //
//   // TODO?: We can probably adapt this loop->DAC process such that we
//   // don't require all IV's to be canonical.
//   SmallVector<PHINode*, 8> IVs;
//   bool AllCanonical = true;
//   for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
//     PHINode *PN = cast<PHINode>(II);
//     const SCEVAddRecExpr *PNSCEV = dyn_cast<const SCEVAddRecExpr>(SE.getSCEV(PN));
//     assert(PNSCEV && "PHINode did not have corresponding SCEVAddRecExpr.");
//     assert(PNSCEV->getStart()->isZero() && "PHINode SCEV does not start at 0.");
//     DEBUG(dbgs() << "L2C step recurrence for SCEV " << *PNSCEV << " is " << *(PNSCEV->getStepRecurrence(SE)) << "\n");
//     assert(PNSCEV->getStepRecurrence(SE)->isOne() && "PHINode SCEV step is not 1.");
//     if (ConstantInt *C = dyn_cast<ConstantInt>(PN->getIncomingValueForBlock(Preheader))) {
//       if (C->isZero())
//         IVs.push_back(PN);
//     } else {
//       AllCanonical = false;
//       DEBUG(dbgs() << "Remaining non-canonical PHI Node found: " << *PN << "\n");
//     }
//   }
//   if (!AllCanonical)
//     return nullptr;

//   // Insert the computation for the loop limit into the Preheader.
//   Value *LimitVar = Exp.expandCodeFor(Limit, Limit->getType(),
//                                       &(Preheader->front()));
//   DEBUG(dbgs() << "LimitVar: " << *LimitVar << "\n");
//   // dbgs() << "EL Preheader after adding Limit:" << *Preheader;
//   // dbgs() << "EL Header after adding Limit:" << *Header;
//   // dbgs() << "EL Latch after adding Limit:" << *Latch;

//   // Canonicalize the loop latch.
//   // dbgs() << "Loop backedge guarded by " << *(SE.getSCEV(CanonicalIV)) << " < " << *Limit <<
//   //    ": " << SE.isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT, SE.getSCEV(CanonicalIV), Limit) << "\n";
//   assert(SE.isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT, SE.getSCEV(CanonicalIV), Limit) &&
//          "Loop backedge is not guarded by canonical comparison with limit.");
//   Value *NewCond = canonicalizeLoopLatch(L, CanonicalIV, LimitVar);

//   // dbgs() << "EL Preheader after new Latch:" << *Preheader;
//   // dbgs() << "EL Header after new Latch:" << *Header;
//   // dbgs() << "EL Latch after new Latch:" << *Latch;

//   // Insert computation of grainsize into the Preheader.
//   // For debugging.
//   Value *GrainVar = ConstantInt::get(Limit->getType(), 2);
//   // Value *GrainVar = computeGrainsize(L, LimitVar);
//   DEBUG(dbgs() << "GrainVar: " << *GrainVar << "\n");
//   // dbgs() << "EL Preheader after adding grainsize computation:" << *Preheader;
//   // dbgs() << "EL Header after adding grainsize computation:" << *Header;
//   // dbgs() << "EL Latch after adding grainsize computation:" << *Latch;

//   /// Clone the loop into a new function.

//   // Get the inputs and outputs for the Loop blocks.
//   SetVector<Value*> Inputs, Outputs;
//   SetVector<Value*> BodyInputs, BodyOutputs;
//   ValueToValueMapTy VMap, InputMap;
//   // Add start iteration, end iteration, and grainsize to inputs.
//   {
//     // Get the inputs and outputs for the loop body.
//     CodeExtractor Ext(L->getBlocks(), &DT);
//     Ext.findInputsOutputs(BodyInputs, BodyOutputs);

//     // Add argument for start of CanonicalIV.
//     Value *CanonicalIVInput = CanonicalIV->getIncomingValueForBlock(Preheader);
//     // CanonicalIVInput should be the constant 0.
//     assert(isa<Constant>(CanonicalIVInput) &&
//            "Input to canonical IV from preheader is not constant.");
//     Argument *StartArg = new Argument(CanonicalIV->getType(),
//                                       CanonicalIV->getName()+".start");
//     Inputs.insert(StartArg);
//     InputMap[CanonicalIV] = StartArg;

//     // for (PHINode *IV : IVs) {
//     //   Value *IVInput = IV->getIncomingValueForBlock(Preheader);
//     //   if (isa<Constant>(IVInput)) {
//     //     Argument *StartArg = new Argument(IV->getType(), IV->getName()+".start");
//     //     Inputs.insert(StartArg);
//     //     InputMap[IV] = StartArg;
//     //   } else {
//     //     assert(BodyInputs.count(IVInput) &&
//     //            "Non-constant input to IV not captured.");
//     //     Inputs.insert(IVInput);
//     //     InputMap[IV] = IVInput;
//     //   }
//     // }

//     // Add argument for end.
//     if (isa<Constant>(LimitVar)) {
//       Argument *EndArg = new Argument(LimitVar->getType(), "end");
//       Inputs.insert(EndArg);
//       InputMap[LimitVar] = EndArg;
//       // if (!isa<Constant>(LimitVar))
//       //   VMap[LimitVar] = EndArg;
//     } else {
//       Inputs.insert(LimitVar);
//       InputMap[LimitVar] = LimitVar;
//     }

//     // Add argument for grainsize.
//     // Inputs.insert(GrainVar);
//     if (isa<Constant>(GrainVar)) {
//       Argument *GrainArg = new Argument(GrainVar->getType(), "grainsize");
//       Inputs.insert(GrainArg);
//       InputMap[GrainVar] = GrainArg;
//     } else {
//       Inputs.insert(GrainVar);
//       InputMap[GrainVar] = GrainVar;
//     }

//     // Put all of the inputs together, and clear redundant inputs from
//     // the set for the loop body.
//     SmallVector<Value*, 8> BodyInputsToRemove;
//     for (Value *V : BodyInputs)
//       if (!Inputs.count(V))
//         Inputs.insert(V);
//       else
//         BodyInputsToRemove.push_back(V);
//     for (Value *V : BodyInputsToRemove)
//       BodyInputs.remove(V);
//     assert(0 == BodyOutputs.size() &&
//            "All results from parallel loop should be passed by memory already.");
//   }
//   DEBUG({
//       for (Value *V : Inputs)
//         dbgs() << "EL input: " << *V << "\n";
//       // for (Value *V : Outputs)
//       //        dbgs() << "EL output: " << *V << "\n";
//     });


//   Function *Helper;
//   {
//     SmallVector<ReturnInst *, 4> Returns;  // Ignore returns cloned.
  
//     // LowerDbgDeclare(*(Header->getParent()));

//     if (DISubprogram *SP = Header->getParent()->getSubprogram()) {
//       // If we have debug info, add mapping for the metadata nodes that should not
//       // be cloned by CloneFunctionInto.
//       auto &MD = VMap.MD();
//       // dbgs() << "SP: " << *SP << "\n";
//       // dbgs() << "MD:\n";
//       // for (auto &Tmp : MD) {
//       //   dbgs() << *(Tmp.first) << " -> " << *(Tmp.second) << "\n";
//       // }
//       MD[SP->getUnit()].reset(SP->getUnit());
//       MD[SP->getType()].reset(SP->getType());
//       MD[SP->getFile()].reset(SP->getFile());
//     }
//     // {
//     //   dbgs() << "L2C TEST CLONE\n";
//     //   ValueToValueMapTy VMap2;
//     //   if (DISubprogram *SP = Header->getParent()->getSubprogram()) {
//     //     // If we have debug info, add mapping for the metadata nodes that should not
//     //     // be cloned by CloneFunctionInto.
//     //     auto &MD = VMap2.MD();
//     //     dbgs() << "SP: " << *SP << "\n";
//     //     for (auto &Tmp : MD) {
//     //       dbgs() << *(Tmp.first) << " -> " << *(Tmp.second) << "\n";
//     //     }
//     //     MD[SP->getUnit()].reset(SP->getUnit());
//     //     MD[SP->getType()].reset(SP->getType());
//     //     MD[SP->getFile()].reset(SP->getFile());
//     //   }
//     //   Function *TestFunc = CloneFunction(Header->getParent(), VMap2, nullptr);
//     // }

//     Helper = CreateHelper(Inputs, Outputs, L->getBlocks(),
//                           Header, Preheader, L->getExitBlock(),
//                           VMap, Header->getParent()->getParent(),
//                           /*ModuleLevelChanges=*/false, Returns, ".l2c",
//                           nullptr, nullptr, nullptr);

//     assert(Returns.empty() && "Returns cloned when cloning loop.");

//     // // If we have debug info, update it.  "ModuleLevelChanges = true"
//     // // above does the heavy lifting, we just need to repoint
//     // // subprogram at the same DICompileUnit as the original function.
//     // //
//     // // TODO: Cleanup the template parameters in
//     // // Helper->getSubprogram() to reflect the variables used in the
//     // // helper.
//     // if (DISubprogram *SP = Header->getParent()->getSubprogram())
//     //   Helper->getSubprogram()->replaceUnit(SP->getUnit());

//     // Use a fast calling convention for the helper.
//     Helper->setCallingConv(CallingConv::Fast);
//     // Helper->setCallingConv(Header->getParent()->getCallingConv());
//   }

//   // Add a sync to the helper's return.
//   {
//     BasicBlock *HelperExit = cast<BasicBlock>(VMap[L->getExitBlock()]);
//     assert(isa<ReturnInst>(HelperExit->getTerminator()));
//     BasicBlock *NewHelperExit = SplitBlock(HelperExit,
//                                            HelperExit->getTerminator(),
//                                            &DT, &LI);
//     IRBuilder<> Builder(&(HelperExit->front()));
//     SyncInst *NewSync = Builder.CreateSync(NewHelperExit);
//     // Set debug info of new sync.
//     if (isa<SyncInst>(L->getExitBlock()->getTerminator()))
//       NewSync->setDebugLoc(L->getExitBlock()->getTerminator()->getDebugLoc());
//     else
//       DEBUG(dbgs() << "Sync not found in loop exit block.\n");
//     RemapInstruction(NewSync, VMap, RF_None | RF_IgnoreMissingLocals);
//     HelperExit->getTerminator()->eraseFromParent();
//   }

//   BasicBlock *NewPreheader = cast<BasicBlock>(VMap[Preheader]);
//   PHINode *NewCanonicalIV = cast<PHINode>(VMap[CanonicalIV]);

//   // Rewrite the cloned IV's to start at the start iteration argument.
//   {
//     // Rewrite clone of canonical IV to start at the start iteration
//     // argument.
//     Argument *NewCanonicalIVStart = cast<Argument>(VMap[InputMap[CanonicalIV]]);
//     {
//       int NewPreheaderIdx = NewCanonicalIV->getBasicBlockIndex(NewPreheader);
//       assert(isa<Constant>(NewCanonicalIV->getIncomingValue(NewPreheaderIdx)) &&
//              "Cloned canonical IV does not inherit a constant value from cloned preheader.");
//       NewCanonicalIV->setIncomingValue(NewPreheaderIdx, NewCanonicalIVStart);
//     }

//     // Rewrite other cloned IV's to start at their value at the start
//     // iteration.
//     const SCEV *StartIterSCEV = SE.getSCEV(NewCanonicalIVStart);
//     DEBUG(dbgs() << "StartIterSCEV: " << *StartIterSCEV << "\n");
//     for (PHINode *IV : IVs) {
//       if (CanonicalIV == IV) continue;

//       // Get the value of the IV at the start iteration.
//       DEBUG(dbgs() << "IV " << *IV);
//       const SCEV *IVSCEV = SE.getSCEV(IV);
//       DEBUG(dbgs() << " (SCEV " << *IVSCEV << ")");
//       const SCEVAddRecExpr *IVSCEVAddRec = cast<const SCEVAddRecExpr>(IVSCEV);
//       const SCEV *IVAtIter = IVSCEVAddRec->evaluateAtIteration(StartIterSCEV, SE);
//       DEBUG(dbgs() << " expands at iter " << *StartIterSCEV <<
//             " to " << *IVAtIter << "\n");

//       // NOTE: Expanded code should not refer to other IV's.
//       Value *IVStart = Exp.expandCodeFor(IVAtIter, IVAtIter->getType(),
//                                          NewPreheader->getTerminator());


//       // Set the value that the cloned IV inherits from the cloned preheader.
//       PHINode *NewIV = cast<PHINode>(VMap[IV]);
//       int NewPreheaderIdx = NewIV->getBasicBlockIndex(NewPreheader);
//       assert(isa<Constant>(NewIV->getIncomingValue(NewPreheaderIdx)) &&
//              "Cloned IV does not inherit a constant value from cloned preheader.");
//       NewIV->setIncomingValue(NewPreheaderIdx, IVStart);
//     }

//     // Remap the newly added instructions in the new preheader to use
//     // values local to the helper.
//     // dbgs() << "NewPreheader:" << *NewPreheader;
//     for (Instruction &II : *NewPreheader)
//       RemapInstruction(&II, VMap, RF_IgnoreMissingLocals,
//                        /*TypeMapper=*/nullptr, /*Materializer=*/nullptr);
//   }
  
//   // // Rewrite cloned IV's to start at the start iteration argument.
//   // BasicBlock *NewPreheader = cast<BasicBlock>(VMap[Preheader]);
//   // for (PHINode *IV : IVs) {
//   //   PHINode *NewIV = cast<PHINode>(VMap[IV]);
//   //   int NewPreheaderIdx = NewIV->getBasicBlockIndex(NewPreheader);
//   //   if (isa<Constant>(NewIV->getIncomingValue(NewPreheaderIdx)))
//   //     NewIV->setIncomingValue(NewPreheaderIdx, cast<Value>(VMap[InputMap[IV]]));
//   // }
  
//   // If the loop limit is constant, then rewrite the loop latch
//   // condition to use the end-iteration argument.
//   if (isa<Constant>(LimitVar)) {
//     CmpInst *HelperCond = cast<CmpInst>(VMap[NewCond]);
//     assert(HelperCond->getOperand(1) == LimitVar);
//     IRBuilder<> Builder(HelperCond);
//     Value *NewHelperCond = Builder.CreateICmpULT(HelperCond->getOperand(0),
//                                                  VMap[InputMap[LimitVar]]);
//     HelperCond->replaceAllUsesWith(NewHelperCond);
//     HelperCond->eraseFromParent();
//   }

//   // DT.recalculate(*Helper);

//   // For debugging:
//   // BasicBlock *NewHeader = cast<BasicBlock>(VMap[Header]);
//   // SerializeDetachedCFG(cast<DetachInst>(NewHeader->getTerminator()), nullptr);
//   implementDACIterSpawnOnHelper(Helper, NewPreheader,
//                                 cast<BasicBlock>(VMap[Header]),
//                                 cast<PHINode>(VMap[CanonicalIV]),
//                                 cast<Argument>(VMap[InputMap[LimitVar]]),
//                                 cast<Argument>(VMap[InputMap[GrainVar]]),
//                                 /*DT=*/nullptr, /*LI=*/nullptr,
//                                 CanonicalSCEV->getNoWrapFlags(SCEV::FlagNUW),
//                                 CanonicalSCEV->getNoWrapFlags(SCEV::FlagNSW));

//   // TODO: Replace this once the function is complete.
//   // dbgs() << "New Helper function:" << *Helper;
//   if (llvm::verifyFunction(*Helper, &dbgs()))
//     return nullptr;

//   // Add call to helper function.
//   {
//     // Setup arguments for call.
//     SetVector<Value*> TopCallArgs;
//     // Add start iteration 0.
//     assert(CanonicalSCEV->getStart()->isZero() &&
//            "Canonical IV does not start at zero.");
//     TopCallArgs.insert(ConstantInt::get(CanonicalIV->getType(), 0));
//     // Add loop limit.
//     TopCallArgs.insert(LimitVar);
//     // Add grainsize.
//     TopCallArgs.insert(GrainVar);
//     // Add the rest of the arguments.
//     for (Value *V : BodyInputs)
//       TopCallArgs.insert(V);

//     // Create call instruction.
//     IRBuilder<> Builder(Preheader->getTerminator());
//     CallInst *TopCall = Builder.CreateCall(Helper, TopCallArgs.getArrayRef());
//     // Use a fast calling convention for the helper.
//     TopCall->setCallingConv(CallingConv::Fast);
//     // TopCall->setCallingConv(Helper->getCallingConv());
//     TopCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
//   }
  
//   ++LoopsConvertedToDAC;

//   return Helper;
// }

// ////////////////////////////////////////////////////////////////////////

// Value* neg(Value* V) {
//   if( Constant* C = dyn_cast<Constant>(V) ) {
//     ConstantFolder F;
//     return F.CreateNeg(C);
//   }

//   Instruction* I = nullptr;
//   bool move = false;
//   if( Argument* A = dyn_cast<Argument>(V) ) {
//     I = A->getParent()->getEntryBlock().getFirstNonPHIOrDbgOrLifetime();
//   } else if( PHINode* A = dyn_cast<PHINode>(V) ) {
//     I = A->getParent()->getFirstNonPHIOrDbgOrLifetime();
//   } else {
//     assert( isa<Instruction>(V) );
//     I = cast<Instruction>(V);
//     move = true;
//   }
//   assert(I);
//   IRBuilder<> builder(I);
//   Instruction* foo = cast<Instruction>(builder.CreateNeg(V));
//   if (move) I->moveBefore(foo);
//   return foo;
// }

// Value* subOne(Value* V, std::string s="") {
//   if( Constant* C = dyn_cast<Constant>(V) ) {
//     ConstantFolder F;
//     return F.CreateSub(C, ConstantInt::get(V->getType(), 1) );
//   }
//   Instruction* I = nullptr;
//   bool move = false;
//   if( Argument* A = dyn_cast<Argument>(V) ) {
//     I = A->getParent()->getEntryBlock().getFirstNonPHIOrDbgOrLifetime();
//   } else if( PHINode* A = dyn_cast<PHINode>(V) ) {
//     I = A->getParent()->getFirstNonPHIOrDbgOrLifetime();
//   } else {
//     assert( isa<Instruction>(V) );
//     I = cast<Instruction>(V);
//     move = true;
//   }
//   assert(I);
//   IRBuilder<> builder(I);
//   Instruction* foo = cast<Instruction>(builder.CreateSub(V, ConstantInt::get(V->getType(), 1), s));
//   if (move) I->moveBefore(foo);
//   return foo;
// }

// Value* addOne(Value* V, std::string n="") {
//   if (Constant* C = dyn_cast<Constant>(V)) {
//     ConstantFolder F;
//     return F.CreateAdd(C, ConstantInt::get(V->getType(), 1) );
//   }

//   Instruction* I = nullptr;
//   bool move = false;
//   if( Argument* A = dyn_cast<Argument>(V) ) {
//     I = A->getParent()->getEntryBlock().getFirstNonPHIOrDbgOrLifetime();
//   } else if( PHINode* A = dyn_cast<PHINode>(V) ) {
//     I = A->getParent()->getFirstNonPHIOrDbgOrLifetime();
//   } else {
//     assert( isa<Instruction>(V) );
//     I = cast<Instruction>(V);
//     move = true;
//   }
//   assert(I);
//   IRBuilder<> builder(I);
//   Instruction* foo = cast<Instruction>(builder.CreateAdd(V, ConstantInt::get(V->getType(), 1), n));
//   if (move) I->moveBefore(foo);
//   return foo;
// }

// Value* uncast(Value* V) {
//   if (TruncInst* in = dyn_cast<TruncInst>(V)) {
//     return uncast(in->getOperand(0));
//   }
//   if (SExtInst* in = dyn_cast<SExtInst>(V)) {
//     return uncast(in->getOperand(0));
//   }
//   if (ZExtInst* in = dyn_cast<ZExtInst>(V)) {
//     return uncast(in->getOperand(0));
//   }
//   return V;
// }

// size_t countPHI(BasicBlock* b){
//     int phi = 0;
//     BasicBlock::iterator i = b->begin();
//     while (isa<PHINode>(i) ) { ++i; phi++; }
//     return phi;
// }

// int64_t getInt(Value* v, bool & failed){
//   if (ConstantInt* CI = dyn_cast<ConstantInt>(v)) {
//     failed = false;
//     return CI->getSExtValue();
//   }
//   failed = true;
//   return -1;
// }

// bool isOne(Value* v){
//   bool m = false;
//   return getInt(v, m) == 1;
// }

// bool isZero(Value* v){
//   bool m = false;
//   return getInt(v, m) == 0;
// }

// bool attemptRecursiveMoveHelper(Instruction* toMoveAfter, Instruction* toCheck, DominatorTree& DT, std::vector<Instruction*>& candidates) {
//   if (DT.dominates(toMoveAfter, toCheck)) return true;

//   if (toCheck->mayHaveSideEffects()) {
//     llvm::errs() << "invalid move\n"; toCheck->dump();
//     return false;
//   }

//   for (auto & u2 : toCheck->uses() ) {
//     if (!DT.dominates(toMoveAfter, u2) ) {
//       assert( isa<Instruction>(u2.getUser()) );
//       if (!attemptRecursiveMoveHelper(toMoveAfter, cast<Instruction>(u2.getUser()), DT, candidates)) return false;
//     }
//   }

//   candidates.push_back(toCheck);
//   return true;
// }

// bool attemptRecursiveMoveAfter(Instruction* toMoveAfter, Instruction* toCheck, DominatorTree& DT) {
//   std::vector<Instruction*> candidates;
//   bool b = attemptRecursiveMoveHelper(toMoveAfter, toCheck, DT, candidates);
//   if (!b) return false;

//   auto last = toMoveAfter;
//   for (Instruction* cand : candidates) {
//     cand->moveBefore(last);
//     last = cand;
//   }
//   if (last != toMoveAfter) toMoveAfter->moveBefore(last);

//   if (!DT.dominates(toMoveAfter, toCheck)) {
//     toMoveAfter->dump();
//     toCheck->dump();
//     toMoveAfter->getParent()->getParent()->dump();
//     return false;
//   }
//   assert(DT.dominates(toMoveAfter, toCheck));
//   return true;
// }

// bool recursiveMoveBefore(Instruction* toMoveBefore, Value* toMoveVal, DominatorTree& DT, std::string nm) {
//   Instruction* toMoveI = dyn_cast<Instruction>(toMoveVal);
//   if (!toMoveI) return true;

//   std::vector<Value*> toMove;
//   toMove.push_back(toMoveI);
//   Instruction* pi = toMoveBefore;

//   while (!toMove.empty()) {
//     auto b = toMove.back();
//     toMove.pop_back();
//     if( Instruction* inst = dyn_cast<Instruction>(b) ) {
//       if( !DT.dominates(inst, toMoveBefore) ) {
//         for (User::op_iterator i = inst->op_begin(), e = inst->op_end(); i != e; ++i) {
//           Value *v = *i;
//           toMove.push_back(v);
//           //if (isa<Instruction>(v)) { llvm::errs()<<"for "; inst->dump(); v->dump(); }
//         }
//         if (inst->mayHaveSideEffects()) {
//           errs() << "something side fx\n";
//           return false;
//         }
//         if (isa<PHINode>(inst)) {
//           errs() << "some weird phi stuff trying to move, move before:" << nm << "\n";
//           inst->dump();
//           toMoveBefore->dump();
//           return false;
//         }
//         inst->moveBefore(pi);
//         pi = inst;
//       }
//     }
//   }
//   return true;
// }

// /* Returns ind var / number of iterations */
// std::pair<PHINode*,Value*> getIndVar(Loop *L, BasicBlock* detacher, DominatorTree& DT, bool actualFix=true) {
//   BasicBlock *H = L->getHeader();

//   BasicBlock *Incoming = nullptr, *Backedge = nullptr;
//   pred_iterator PI = pred_begin(H);
//   assert(PI != pred_end(H) && "Loop must have at least one backedge!");
//   Backedge = *PI++;
//   if (PI == pred_end(H)) return make_pair(nullptr,nullptr);  // dead loop
//   Incoming = *PI++;
//   if (PI != pred_end(H)) return make_pair(nullptr,nullptr);  // multiple backedges?

//   if (L->contains(Incoming)) {
//     if (L->contains(Backedge)) return make_pair(nullptr,nullptr);
//     std::swap(Incoming, Backedge);
//   } else if (!L->contains(Backedge)) return make_pair(nullptr,nullptr);

//   assert( L->contains(Backedge) );
//   assert( !L->contains(Incoming) );
//   llvm::CmpInst* cmp = 0;
//   int cmpIdx = -1;
//   llvm::Value* opc = 0;

//   BasicBlock* cmpNode = Backedge;
//   if (H != detacher) {
//     cmpNode = detacher->getUniquePredecessor();
//     if(cmpNode==nullptr) return make_pair(nullptr,nullptr);
//   }

//   if (BranchInst* brnch = dyn_cast<BranchInst>(cmpNode->getTerminator()) ) {
//     if (!brnch->isConditional()) goto cmp_error;
//     cmp = dyn_cast<CmpInst>(brnch->getCondition());
//     if (cmp == nullptr) {
//       errs() << "no comparison inst from backedge\n";
//       cmpNode->getTerminator()->dump();
//       return make_pair(nullptr,nullptr);
//     }
//     if (!L->contains(brnch->getSuccessor(0))) {
//       cmp->setPredicate(CmpInst::getInversePredicate(cmp->getPredicate()));
//       brnch->swapSuccessors();
//     }
//     if (!cmp->isIntPredicate() || cmp->getPredicate() == CmpInst::ICMP_EQ ) {
//       cmpNode->getParent()->dump();
//       cmpNode->dump();
//       cmp->dump();
//       brnch->dump();
//       return make_pair(nullptr,nullptr);
//     }
//   } else {
//     cmp_error:
//     errs() << "<no comparison from backedge>\n";
//     cmpNode->getTerminator()->dump();
//     cmpNode->getParent()->dump();
//     errs() << "</no comparison from backedge>\n";
//     return make_pair(nullptr,nullptr);
//   }

//   for (unsigned i=0; i<2; i++) {
//     LoadInst* inst = dyn_cast<LoadInst>(uncast(cmp->getOperand(i)));
//     if (!inst) continue;
//     AllocaInst* alloca = dyn_cast<AllocaInst>(inst->getOperand(0));
//     if (!alloca) continue;
//     if (isAllocaPromotable(alloca, DT)) {
//       PromoteMemToReg({alloca}, DT, nullptr, nullptr);
//     }
//   }
//   if (!actualFix) return make_pair(nullptr,nullptr);

//   // Loop over all of the PHI nodes, looking for a canonical indvar.
//   PHINode* RPN = nullptr;
//   Instruction* INCR = nullptr;
//   Value* amt = nullptr;
//   std::vector<std::tuple<PHINode*,Instruction*,Value*>> others;
//   for (BasicBlock::iterator I = H->begin(); isa<PHINode>(I); ) {
//     assert( isa<PHINode>(I) );
//     PHINode *PN = cast<PHINode>(I);
//     dbgs() << "examining PHI Node " << *PN << "\n";
//     if (LoadInst* ld = dyn_cast<LoadInst>(uncast(PN->getIncomingValueForBlock(Incoming)))) {
//       if (LoadInst* ld2 = dyn_cast<LoadInst>(uncast(PN->getIncomingValueForBlock(Backedge)))) {
//         LoadInst *t1 = ld, *t2 = ld2;
//         bool valid = false;
//         while (t1 && t2) {
//           if(t1->getPointerOperand() == t1->getPointerOperand()) { valid = true; break; }
//           uncast(t1->getPointerOperand())->dump();
//           uncast(t2->getPointerOperand())->dump();

//           /// TODO GEP inst
//           ///if (LoadInst* ld = dyn_cast<LoadInst>(uncast(PN->getIncomingValueForBlock(Incoming)))) {
//           ///  if (LoadInst* ld2 = dyn_cast<LoadInst>(uncast(PN->getIncomingValueForBlock(Backedge)))) {


//           t1 = dyn_cast<LoadInst>(uncast(t1->getPointerOperand()));
//           t2 = dyn_cast<LoadInst>(uncast(t2->getPointerOperand()));
//         }
//         if (valid) {
//           ++I;
//           ld2->replaceAllUsesWith(ld);
//           PN->replaceAllUsesWith(PN->getIncomingValueForBlock(Incoming));
//           PN->eraseFromParent();
//           ld2->eraseFromParent();
//           continue;
//         } else {
//           llvm::errs() << "phinode cmp uses odd load with diff values\n";
//           ld->dump();
//           ld2->dump();
//           H->getParent()->dump();
//         }
//       }
//     }

//     if( !PN->getType()->isIntegerTy() ) {
//       errs() << "phinode uses non-int type\n";
//       PN->dump();
//       H->getParent()->dump();
//       return make_pair(nullptr,nullptr);
//     }
//     if (BinaryOperator* Inc = dyn_cast<BinaryOperator>(PN->getIncomingValueForBlock(Backedge))) {
//       if (Inc->getOpcode() == Instruction::Sub && Inc->getOperand(0) == PN) {
//         IRBuilder<> build(Inc);
//         auto val = build.CreateNeg(Inc->getOperand(1));
//         auto tmp = build.CreateAdd(PN, val);
//         assert( isa<BinaryOperator>(tmp) );
//         auto newI = cast<BinaryOperator>(tmp);
//         Inc->replaceAllUsesWith(newI);
//         for (auto& tup : others) {
//           if (std::get<1>(tup) == Inc) std::get<1>(tup) = newI;
//           if (std::get<2>(tup) == Inc) std::get<2>(tup) = newI;
//         }
//         Inc->eraseFromParent();
//         Inc = newI;
//       }
//       if (Inc->getOpcode() == Instruction::Add && (uncast(Inc->getOperand(0)) == PN || uncast(Inc->getOperand(1)) == PN) ) {
//         if ( uncast(Inc->getOperand(1)) == PN ) Inc->swapOperands();
//         assert( uncast(Inc->getOperand(0)) == PN);
//         bool rpnr = false;
//         bool incr = false;
//         for(unsigned i = 0; i < cmp->getNumOperands(); i++) {
//           bool hadr = uncast(cmp->getOperand(i)) == PN;
//           rpnr |= hadr;
//           bool hadi = uncast(cmp->getOperand(i)) == Inc;
//           incr |= hadi;
//           if (hadr | hadi) { assert(cmpIdx == -1); cmpIdx = i; }
//         }
//         assert( !rpnr || !incr );
//         if( rpnr | incr ) {
//           amt = Inc->getOperand(1);
//           RPN = PN;
//           INCR = Inc;
//           opc = rpnr?RPN:INCR;
//         } else {
//           others.push_back( std::make_tuple(PN,Inc,Inc->getOperand(1)) );
//         }
//         assert( !isa<PHINode>(Inc->getOperand(1)) );
//         if (!recursiveMoveBefore(Incoming->getTerminator(), Inc->getOperand(1), DT, "1")) return make_pair(nullptr, nullptr);
//         //assert( !isa<PHINode>(PN->getIncomingValueForBlock(Incoming)) );
//         if (!recursiveMoveBefore(Incoming->getTerminator(), PN->getIncomingValueForBlock(Incoming), DT, "2")) return make_pair(nullptr, nullptr);
//       } else {
//         errs() << "no add found for:\n"; PN->dump(); Inc->dump();
//         H->getParent()->dump();
//         return make_pair(nullptr,nullptr);
//       }
//     } else {
//       errs() << "no inc found for:\n"; PN->dump(); PN->getParent()->getParent()->dump();
//       return make_pair(nullptr, nullptr);
//     }
//     ++I;
//   }

//   if (RPN == 0) {
//     errs() << "<no RPN>\n";
//     cmp->dump();
//     errs() << "<---->\n";
//     H->dump();
//     errs() << "<---->\n";
//     for( auto a : others ) { std::get<0>(a)->dump(); }
//     errs() << "</no RPN>\n";
//     return make_pair(nullptr,nullptr);
//   }

//   llvm::Value* mul;
//   llvm::Value* newV;

//   SmallPtrSet<llvm::Value*, 4> toIgnore;
//   {
//     BasicBlock* Spawned = detacher->getTerminator()->getSuccessor(0);

//     if (cilk::getNumPred(Spawned) > 1) {
//       BasicBlock* ts = BasicBlock::Create(Spawned->getContext(), Spawned->getName()+".fx", Spawned->getParent(), detacher);
//       IRBuilder<> b(ts);
//       b.CreateBr(Spawned);
//       detacher->getTerminator()->setSuccessor(0,ts);
//       llvm::BasicBlock::iterator i = Spawned->begin();
//       while (auto phi = llvm::dyn_cast<llvm::PHINode>(i)) {
//         int idx = phi->getBasicBlockIndex(detacher);
//         phi->setIncomingBlock(idx, ts);
//         ++i;
//       }
//       Spawned = ts;
//     }

//     IRBuilder<> builder(Spawned->getFirstNonPHIOrDbgOrLifetime());
//     if( isOne(amt) ) mul = RPN;
//     else toIgnore.insert(mul = builder.CreateMul(RPN, amt, "indmul"));
//     if( isZero(RPN->getIncomingValueForBlock(Incoming) )) newV = mul;
//     else toIgnore.insert(newV = builder.CreateAdd(mul, RPN->getIncomingValueForBlock(Incoming), "indadd"));

//     for( auto a : others ) {
//       llvm::Value* val = builder.CreateSExtOrTrunc(RPN, std::get<0>(a)->getType());
//       if (val != RPN) toIgnore.insert(val);
//       llvm::Value* amt0 = std::get<2>(a);
//       if( !isOne(amt0) ) val = builder.CreateMul(val,amt0, "vmul");
//       if (val != RPN) toIgnore.insert(val);
//       llvm::Value* add0 = std::get<0>(a)->getIncomingValueForBlock(Incoming);
//       if( !isZero(add0) ) val = builder.CreateAdd(val,add0, "vadd");
//       if (val != RPN) toIgnore.insert(val);
//       assert( isa<Instruction>(val) );
//       Instruction* ival = cast<Instruction>(val);

//       for (auto& u : std::get<0>(a)->uses()) {
//         assert( isa<Instruction>(u.getUser()) );
//         Instruction *user = cast<Instruction>(u.getUser());

//         //No need to override use in PHINode itself
//         if (user == std::get<0>(a)) continue;
//         //No need to override use in increment
//         if (user == std::get<1>(a)) continue;

//         if (!attemptRecursiveMoveAfter(ival, user, DT)) {
//           val->dump();
//           user->dump();
//           std::get<0>(a)->dump();
//           H->getParent()->dump();
//           llvm::errs() << "FAILED TO MOVE\n";
//           return make_pair(nullptr, nullptr);
//         }
//         assert(DT.dominates(ival, user));
//       }
//       {
//         auto tmp = std::get<0>(a);
//         tmp->replaceAllUsesWith(val);
//         for (auto& tup : others) {
//           if (std::get<1>(tup) == tmp) std::get<1>(tup) = tmp;
//           if (std::get<2>(tup) == tmp) std::get<2>(tup) = tmp;
//         }
//         tmp->eraseFromParent();
//       }
//       if(std::get<1>(a)->getNumUses() == 0) {
//         auto tmp = std::get<1>(a);
//         tmp->eraseFromParent();
//       }
//     }
//   }

//   std::vector<Use*> uses;
//   for( Use& U : RPN->uses() ) uses.push_back(&U);
//   for( Use* Up : uses ) {
//     Use &U = *Up;
//     assert( isa<Instruction>(U.getUser()) );
//     Instruction *I = cast<Instruction>(U.getUser());
//     if( I == INCR ) INCR->setOperand(1, ConstantInt::get( RPN->getType(), 1 ) );
//     else if( toIgnore.count(I) > 0 && I != RPN ) continue;
//     else if( uncast(I) == cmp || I == cmp->getOperand(0) || I == cmp->getOperand(1) || uncast(I) == cmp || I == RPN || I->getParent() == cmp->getParent() || I->getParent() == detacher) continue;
//     else {
//       assert( isa<Instruction>(newV) );
//       Instruction* ival = cast<Instruction>(newV);
//       assert( isa<Instruction>(U.getUser()) );
//       if (!attemptRecursiveMoveAfter(ival, cast<Instruction>(U.getUser()), DT)) {
//         llvm::errs() << "newV: ";
//         newV->dump();
//         llvm::errs() << "U: ";
//         U->dump();
//         llvm::errs() << "I: ";
//         I->dump();
//         llvm::errs() << "uncast(I): ";
//         uncast(I)->dump();
//         llvm::errs() << "errs: ";
//         cmp->dump();
//         llvm::errs() << "RPN: ";
//         RPN->dump();
//         H->getParent()->dump();
//         llvm::errs() << "FAILED TO MOVE2\n";
//         return make_pair(nullptr, nullptr);
//       }
//       assert( DT.dominates((Instruction*) newV, U) );
//       U.set( newV );
//     }
//   }

//   IRBuilder<> build(cmp);
//   llvm::Value* val = build.CreateSExtOrTrunc(cmp->getOperand(1-cmpIdx),RPN->getType());
//   llvm::Value* adder = RPN->getIncomingValueForBlock(Incoming);
//   llvm::Value* amt0  = amt;

//   int cast_type = 0;
//   if (isa<TruncInst>(RPN)) cast_type = 1;
//   if (isa<SExtInst>(RPN))  cast_type = 2;
//   if (isa<ZExtInst>(RPN))  cast_type = 3;

//   switch(cast_type) {
//     default:;
//     case 1: amt0 = build.CreateTrunc(amt0,RPN->getType());
//     case 2: amt0 = build.CreateSExt( amt0,RPN->getType());
//     case 3: amt0 = build.CreateZExt( amt0,RPN->getType());
//   }
//   switch(cast_type){
//     default:;
//     case 1: adder = build.CreateTrunc(adder,RPN->getType());
//     case 2: adder = build.CreateSExt( adder,RPN->getType());
//     case 3: adder = build.CreateZExt( adder,RPN->getType());
//   }

//   {
//     Value *bottom = adder, *top = val;
//     if (opc == RPN && DT.dominates(detacher->getTerminator(), cmp)) {
//       cmp->setOperand(cmpIdx, INCR);
//       top = build.CreateAdd(top, amt0, "toplen");
//     }
//     int dir = 0;
//     switch (cmp->getPredicate()) {
//       case CmpInst::ICMP_UGE:
//       case CmpInst::ICMP_UGT:
//       case CmpInst::ICMP_SGE:
//       case CmpInst::ICMP_SGT:
//         dir = -1; break;
//       case CmpInst::ICMP_ULE:
//       case CmpInst::ICMP_ULT:
//       case CmpInst::ICMP_SLE:
//       case CmpInst::ICMP_SLT:
//         dir = +1;break;
//       default:
//         dir = 0;break;
//     }
//     if ( (dir < 0 && cmpIdx == 0) || (dir > 0 && cmpIdx != 0))
//       std::swap(bottom, top);

//     if (!isZero(bottom)) val = build.CreateSub(top, bottom, "sublen");
//     else val = top;

//     switch (cmp->getPredicate() ) {
//       case CmpInst::ICMP_UGT:
//       case CmpInst::ICMP_SGT:
//       case CmpInst::ICMP_ULT:
//       case CmpInst::ICMP_SLT:
//         val = subOne(val, "subineq");
//         break;
//       case CmpInst::ICMP_SLE:
//       case CmpInst::ICMP_ULE:
//       case CmpInst::ICMP_SGE:
//       case CmpInst::ICMP_UGE:
//       default:
//         break;
//     }
//   }
//   {
//     switch (cmp->getPredicate() ) {
//       case CmpInst::ICMP_SLE:
//       case CmpInst::ICMP_ULE:
//       case CmpInst::ICMP_ULT:
//       case CmpInst::ICMP_SLT:
//         if (cmpIdx == 1) amt0 = neg(amt0); break;
//       case CmpInst::ICMP_SGE:
//       case CmpInst::ICMP_UGE:
//       case CmpInst::ICMP_UGT:
//       case CmpInst::ICMP_SGT:
//         if (cmpIdx == 0) amt0 = neg(amt0); break;
//       case CmpInst::ICMP_NE:
//         //amt0 = build.CreateSelect(build.CreateICmpSGT(amt0,ConstantInt::get(val->getType(), 0)),amt0,neg(amt0));
//       default:
//         break;
//     }
//     if (!isOne(amt0)) val = build.CreateSDiv(val, amt0, "divlen");
//     if (cmp->getPredicate()!=CmpInst::ICMP_NE) val = addOne(val, "nepred");
//   }

//   cmp->setPredicate(CmpInst::ICMP_NE);
//   cmp->setOperand(cmpIdx, RPN);
//   cmp->setOperand(1-cmpIdx, val);

//   RPN->setIncomingValue(RPN->getBasicBlockIndex(Incoming),  ConstantInt::get(RPN->getType(), 0));

//   return make_pair(RPN, val);
// }

// void removeFromAll(Loop* L, BasicBlock* B){
//   if( !L ) return;
//   if( L->contains(B) ) L->removeBlockFromLoop(B);
//   removeFromAll(L->getParentLoop(), B);
// }

// template<typename A, typename B> bool contains(const A& a, const B& b) {
//   return std::find(a.begin(), a.end(), b) != a.end();
// }

// BasicBlock* getTrueExit(Loop *L){
//   SmallVector< BasicBlock *, 32> exitBlocks;
//   L->getExitBlocks(exitBlocks);
//   SmallPtrSet<BasicBlock *, 32> exits(exitBlocks.begin(), exitBlocks.end());
//   SmallPtrSet<BasicBlock *, 32> alsoLoop;

//   bool toRemove = true;
//   while (toRemove) {
//     toRemove = false;
//     if (exits.size() >= 2) {
//       for (auto tempExit : exits) {
//         SmallPtrSet<BasicBlock *, 32> reachable;
//         std::vector<BasicBlock*> Q = { tempExit };
//         bool valid = true;
//         while(!Q.empty() && valid) {
//           auto m = Q.back();
//           Q.pop_back();
//           if( isa<UnreachableInst>(m->getTerminator()) ) { reachable.insert(m); continue; }
//           else if( auto b = dyn_cast<BranchInst>(m->getTerminator()) ) {
//             reachable.insert(m);
//             for( unsigned i=0; i<b->getNumSuccessors(); i++ ) {
//                auto suc = b->getSuccessor(i);
//                if( L->contains(suc) || contains(exitBlocks,suc) || contains(alsoLoop, suc) || contains(reachable, suc) ) {

//                } else{
//                 Q.push_back(suc);
//                 break;
//               }
//             }
//           }
//           else valid = false;
//         }
//         if (valid && reachable.size() > 0) {
//           for( auto b : reachable){
//             exits.erase(b);
//             alsoLoop.insert(b);
//           }
//           toRemove = true;
//         }
//       }
//     }
//   }

//   if (exits.size() == 1) return *exits.begin();
//   return nullptr;
// }

// BasicBlock* continueToFindSync(BasicBlock* endL) {
//   //TODO consider lifetime intrinsics
//   while (endL && !isa<SyncInst>(endL->getTerminator())) {
//     if (getNonPhiSize(endL) == 1 && isa<BranchInst>(endL->getTerminator()) && endL->getTerminator()->getNumSuccessors() == 1) {
//       endL = endL->getTerminator()->getSuccessor(0);
//     }
//     else
//       endL = nullptr;
//   }

//   if (endL)
//     assert(endL && isa<SyncInst>(endL->getTerminator()));

//   return endL;
// }


// /*
// cilk_for_recursive(count_t low, count_t high, ...) {
// tail_recurse:
//   count_t count = high - low;
//   if (count > grain)
//   {
//       // Invariant: count >= 2
//       count_t mid = low + count / 2;
//       spawn cilk_for_recursive(low, mid, grain, ...);
//       low = mid;
//       goto tail_recurse;
//   }

//   for(int i=low; i<high; i++) {
//     body(i, data);
//   }
//   sync;
// */
// bool createDACOnExtractedFunction(Function* extracted, LLVMContext &Ctx, std::vector<Value*>& ext_args) {
//   Function::arg_iterator args = extracted->arg_begin();
//   Argument *low0 = &*args;
//   args++;
//   Argument *high0 = &*args;
//   args++;
//   Argument *grain = &*args;

//   BasicBlock *entry = &extracted->getEntryBlock();
//   BasicBlock *body = entry->getTerminator()->getSuccessor(0);
//   BasicBlock *tail_recurse = entry->splitBasicBlock(entry->getTerminator(), "tail_recurse");
//   BasicBlock *recur = BasicBlock::Create(Ctx, "recur", extracted);
//   recur->moveAfter(tail_recurse);
//   tail_recurse->getTerminator()->eraseFromParent();

//   IRBuilder<> trbuilder(tail_recurse);
//   PHINode* low = trbuilder.CreatePHI(low0->getType(), 2, "low");
//   low0->replaceAllUsesWith(low);
//   Value* count = trbuilder.CreateSub(high0, low, "count");
//   Value* cond = trbuilder.CreateICmpUGT(count, grain);
//   trbuilder.CreateCondBr(cond, recur, body);
//   low->addIncoming(low0, entry);

//   IRBuilder<> rbuilder(recur);
//   Value *mid = rbuilder.CreateAdd(low, rbuilder.CreateUDiv(count, ConstantInt::get(count->getType(), 2)), "mid");
//   BasicBlock *detached = BasicBlock::Create(Ctx, "detached", extracted);
//   detached->moveAfter(recur);
//   BasicBlock *reattached = BasicBlock::Create(Ctx, "reattached", extracted);
//   reattached->moveAfter(detached);
//   rbuilder.CreateDetach(detached, reattached);

//   IRBuilder<> dbuilder(detached);

//   //Fill in closure arguments
//   std::vector<Value*> next_args;
//   args = extracted->arg_begin();
//   for (unsigned i=0, len = ext_args.size(); i<len; i++) {
//     next_args.push_back(&*args);
//     args++;
//   }

//   //Replace the bounds arguments
//   next_args[0] = low;
//   next_args[1] = mid;
//   next_args[2] = grain;

//   dbuilder.CreateCall(extracted, next_args);
//   dbuilder.CreateReattach(reattached);

//   IRBuilder<> rebuilder(reattached);
//   rebuilder.CreateBr(tail_recurse);
//   low->addIncoming(mid, reattached);

//   SmallVector<BasicBlock *, 32> blocks;
//   for (BasicBlock& BB : *extracted) { blocks.push_back(&BB); }

//   for (BasicBlock* BB : blocks) {
//     if (ReturnInst *Ret = dyn_cast<ReturnInst>(BB->getTerminator())) {
//       auto tret = BB->splitBasicBlock(Ret);
//       BB->getTerminator()->eraseFromParent();
//       IRBuilder<> build(BB);
//       build.CreateSync(tret);
//     }
//   }
//   return true;
// }

// bool Loop2Cilk::performDAC(Loop *L, LPPassManager &LPM) {

//   BasicBlock* Header = L->getHeader();
//   Module* M = Header->getParent()->getParent();
//   LLVMContext &Ctx = M->getContext();
//   assert(Header);

//   Loop* parentL = L->getParentLoop();
//   LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
//   DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
//   ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();

//   TerminatorInst* T = Header->getTerminator();
//   if (!isa<BranchInst>(T)) {
//     BasicBlock *Preheader = L->getLoopPreheader();
//     if (isa<BranchInst>(Preheader->getTerminator())) {
//       T = Preheader->getTerminator();
//       Header = Preheader;
//     } else if (isa<SyncInst>(Preheader->getTerminator())) {
//       BasicBlock *NewPreheader = SplitEdge(Preheader, Header, &DT, &LI);
//       // SyncInst::Create(NewPreheader, Preheader->getTerminator());
//       // Preheader->getTerminator()->eraseFromParent();
//       // T = BranchInst::Create(Header, NewPreheader->getTerminator());
//       // NewPreheader->getTerminator()->eraseFromParent();
//       T = NewPreheader->getTerminator();
//       Header = NewPreheader;
//     } else {
//       llvm::errs() << "Loop not entered via branch instance\n";
//       T->dump();
//       Preheader->dump();
//       Header->dump();
//       return false;
//     }
//   }

//   // dbgs() << "L2C evaluating " << *L << "\n";
//   // for (BasicBlock *BB : L->getBlocks())
//   //   dbgs() << *BB;

//   // Header = L->getHeader();
//   // DEBUG(dbgs() << "L2C Header:" << *Header);
//   // BasicBlock *Latch = L->getLoopLatch();
//   // DEBUG(dbgs() << "L2C Latch:" << *Latch);
//   // DEBUG(dbgs() << "L2C SE exit count: " << *(SE.getExitCount(L, L->getExitingBlock())) << "\n");

//   // Verify that we can extract loop.
//   if (!verifyLoopStructureForConversion(L))
//     return false;

//   // PredicatedScalarEvolution PSE(SE, *L);

//   if (convertLoopToDACIterSpawn(L, SE, DT, LI)) {
//     // Erase the old loop.
//     Loop *ParentLoop = L->getParentLoop();
//     eraseLoop(L, SE, DT, LI);
//     // Verify parent loop, if one exits.
//     if (ParentLoop)
//       ParentLoop->verifyLoop();
//     return true;
//   }

//   dbgs() << "L2C Failed to transform verified " << *L << "\n";
//   return false;
// }

// bool Loop2Cilk::runOnLoop(Loop *L, LPPassManager &LPM) {
//   Function *ParentF = L->getHeader()->getParent();
//   if (llvm::verifyFunction(*ParentF, &llvm::errs())) {
//     ParentF->dump();
//     assert(0);
//   }
//   if (skipLoop(L)) {
//     return false;
//   }
//   bool ans = performDAC(L, LPM);
//   if (llvm::verifyFunction(*ParentF, &llvm::errs())) {
//     ParentF->dump();
//     assert(0);
//   }
//   return ans;
// }
