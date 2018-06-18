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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Tapir/TapirTypes.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

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

class TapirTarget {
public:
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
