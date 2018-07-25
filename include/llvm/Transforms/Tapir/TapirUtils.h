//===- TapirUtils.h - Tapir Helper functions                ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#ifndef TAPIR_UTILS_H_
#define TAPIR_UTILS_H_

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Transforms/Tapir/TapirTypes.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {
class OptimizationRemarkEmitter;

bool verifyDetachedCFG(const DetachInst &Detach, DominatorTree &DT,
                       bool error = true);

bool populateDetachedCFG(const DetachInst &Detach, DominatorTree &DT,
                         SmallPtrSetImpl<BasicBlock *> &functionPieces,
                         SmallVectorImpl<ReattachInst*> &reattachB,
                         SmallPtrSetImpl<BasicBlock *> &ExitBlocks,
                         bool error = true);

bool populateDetachedCFG(BasicBlock* startSearch, const DetachInst& Detach,
                         DominatorTree &DT,
                         SmallPtrSetImpl<BasicBlock *> &functionPieces,
                         SmallVectorImpl<ReattachInst*> &reattachB,
                         SmallPtrSetImpl<BasicBlock *> &ExitBlocks,
                         bool error = true);

Function *extractDetachBodyToFunction(DetachInst &Detach,
                                      DominatorTree &DT, AssumptionCache &AC,
                                      CallInst **call = nullptr);

/// Utility class for getting and setting loop spawning hints in the form
/// of loop metadata.
/// This class keeps a number of loop annotations locally (as member variables)
/// and can, upon request, write them back as metadata on the loop. It will
/// initially scan the loop for existing metadata, and will update the local
/// values based on information in the loop.
class LoopSpawningHints {
public:
  enum SpawningStrategy {
    ST_SEQ,
    ST_DAC,
    ST_GPU,
    ST_END,
  };

private:
  enum HintKind { HK_STRATEGY, HK_GRAINSIZE };

  /// Hint - associates name and validation with the hint value.
  struct Hint {
    const char *Name;
    unsigned Value; // This may have to change for non-numeric values.
    HintKind Kind;

    Hint(const char *Name, unsigned Value, HintKind Kind)
        : Name(Name), Value(Value), Kind(Kind) {}

    bool validate(unsigned Val);
  };

  /// Spawning strategy
  Hint Strategy;
  /// Grainsize
  Hint Grainsize;


public:
  /// Return the loop metadata prefix.
  static inline StringRef Prefix() { return "tapir.loop."; }
  static inline const char* StrategyPrefix() { return "spawn.strategy"; }
  static inline const char* GrainsizePrefix() { return "grainsize"; }

  static inline std::string printStrategy(enum SpawningStrategy Strat) {
    switch(Strat) {
    case LoopSpawningHints::ST_SEQ:
      return "Spawn iterations sequentially";
    case LoopSpawningHints::ST_DAC:
      return "Use divide-and-conquer";
    case LoopSpawningHints::ST_GPU:
      return "Use gpu";
    default:
      return "Unknown";
    }
  }

  LoopSpawningHints(Loop *L);

  SpawningStrategy getStrategy() const;

  unsigned getGrainsize() const;

  /// The loop these hints belong to.
  Loop * const TheLoop;

  /// Find hints specified in the loop metadata and update local values.
  static inline std::vector<std::pair<StringRef, Metadata*>> getHintsFromMetadata(Loop* L) {
      MDNode *LoopID = L->getLoopID();
      std::vector<std::pair<StringRef, Metadata*>> hints;

      if (!LoopID)
        return hints;

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
        if (Args.size() == 1) {
          if (!Name.startswith(Prefix()))
            continue;
          hints.emplace_back(std::make_pair(Name, Args[0]));
        }
      }
      return hints;
  }

private:
  /// Checks string hint with one operand and set value if valid.
  void setHint(StringRef Name, Metadata *Arg);

  /// Create a new hint from name / value pair.
  MDNode *createHintMetadata(StringRef Name, unsigned V) const;

  /// Matches metadata with hint name.
  bool matchesHintMetadataName(MDNode *Node, ArrayRef<Hint> HintTypes);

  /// Sets current hints into loop metadata, keeping other values intact.
  void writeHintsToMetadata(ArrayRef<Hint> HintTypes);

};

//! Identify if a loop could should be handled manually by a parallel loop backend
static inline bool isBackendParallelFor(Loop* L) {
  for(auto& hints: LoopSpawningHints::getHintsFromMetadata(L)) {
    auto Name = hints.first;
    auto Arg = hints.second;

    Name = Name.substr(LoopSpawningHints::Prefix().size(), StringRef::npos);
    if (Name != LoopSpawningHints::StrategyPrefix()) continue;

    const ConstantInt *C = mdconst::dyn_extract<ConstantInt>(Arg);
    if (!C) continue;

    unsigned Val = C->getZExtValue();

    if (Val >= LoopSpawningHints::ST_END) continue;
    if (Val != LoopSpawningHints::ST_SEQ) return true;
  }
  return false;
}


class TapirTarget {
public:
  virtual ~TapirTarget() {};
  //! For use in loopspawning grainsize calculation
  virtual Value *GetOrCreateWorker8(Function &F) = 0;
  virtual void createSync(SyncInst &inst,
                          ValueToValueMapTy &DetachCtxToStackFrame) = 0;
  virtual Function *createDetach(DetachInst &Detach,
                                 ValueToValueMapTy &DetachCtxToStackFrame,
                                 DominatorTree &DT, AssumptionCache &AC) = 0;
  virtual bool shouldProcessFunction(const Function &F);
  virtual void preProcessFunction(Function &F) = 0;
  virtual void postProcessFunction(Function &F) = 0;
  virtual void postProcessHelper(Function &F) = 0;
  virtual bool processMain(Function &F) = 0;
  virtual bool processLoop(LoopSpawningHints LSH, LoopInfo &LI, ScalarEvolution &SE, DominatorTree &DT,
                           AssumptionCache &AC, OptimizationRemarkEmitter &ORE) = 0;
  //! Helper to perform DAC
  bool processDACLoop(LoopSpawningHints LSH, LoopInfo &LI, ScalarEvolution &SE, DominatorTree &DT,
                           AssumptionCache &AC, OptimizationRemarkEmitter &ORE);
};

TapirTarget *getTapirTargetFromType(TapirTargetType Type);

bool doesDetachedInstructionAlias(AliasSetTracker &CurAST, const Instruction& I, bool FoundMod, bool FoundRef);
// Any reads/writes done in must be done in CurAST
// cannot have any writes/reads, in detached region, respectively
bool doesDetachedRegionAlias(AliasSetTracker &CurAST, const SmallPtrSetImpl<BasicBlock*>& functionPieces);
void moveDetachInstBefore(Instruction* moveBefore, DetachInst& det,
                          const SmallVectorImpl<ReattachInst*>& reattaches,
                          DominatorTree* DT, Value* newSyncRegion=nullptr);
bool attemptSyncRegionElimination(Instruction *SyncRegion);
bool isConstantMemoryFreeOperation(Instruction* inst, bool allowsyncregion=false);
bool isConstantOperation(Instruction* inst, bool allowsyncregion=false);
}  // end namepsace llvm

#endif
