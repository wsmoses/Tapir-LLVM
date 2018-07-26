//===- llvm/Transforms/Tapir/Outline.h - Outlining for Tapir -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines helper functions for outlining portions of code containing
// Tapir instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_OUTLINE_H
#define LLVM_TRANSFORMS_TAPIR_OUTLINE_H

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

typedef SetVector<Value *> ValueSet;

/// definedInRegion - Return true if the specified value is used in the
/// extracted region.
template<class BasicBlockPtrContainer>
static inline size_t countUseInRegion(const BasicBlockPtrContainer &Blocks,
                                Value *V) {
  size_t count = 0;
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    for (User *U : I->users()) {
      if (Instruction *Inst = dyn_cast<Instruction>(U)) {
        if (std::find(Blocks.begin(), Blocks.end(), Inst->getParent()) != Blocks.end()) {
          count++;
        }
      }
    }
  }
  return count;
}

/// definedInRegion - Return true if the specified value is defined in the
/// extracted region.
template<class BasicBlockPtrContainer>
static inline bool definedInRegion(const BasicBlockPtrContainer &Blocks,
                            Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (std::find(Blocks.begin(), Blocks.end(), I->getParent()) != Blocks.end())
      return true;
  return false;
}

/// definedInCaller - Return true if the specified value is defined in the
/// function being code extracted, but not in the region being extracted.
/// These values must be passed in as live-ins to the function.
template<class BasicBlockPtrContainer>
static inline bool definedInCaller(const BasicBlockPtrContainer &Blocks,
                            Value *V) {
  if (isa<Argument>(V)) return true;
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (std::find(Blocks.begin(), Blocks.end(), I->getParent()) == Blocks.end())
      return true;
  return false;
}

// findInputsOutputs - Find inputs and outputs for Blocks.  Any blocks in
// ExitBlocks are handled in a special manner: PHI nodes in Exit Blocks are
// ignored when determining inputs.
// Handles rvalues (should be equivalent to lvalue code below)
template<class BasicBlockPtrContainer>
static inline void findInputsOutputs(const BasicBlockPtrContainer &&Blocks,
                             ValueSet &Inputs, ValueSet &Outputs,
                             DominatorTree& DT,
                             const SmallPtrSetImpl<BasicBlock *> *ExitBlocks = nullptr) {
  for (BasicBlock *BB : Blocks) {
    // If a used value is defined outside the region, it's an input.  If an
    // instruction is used outside the region, it's an output.
    for (Instruction &II : *BB) {
      for (User::op_iterator OI = II.op_begin(), OE = II.op_end(); OI != OE;
           ++OI) {
        // The PHI nodes in each exit block will be updated after the exit block
        // is cloned.  Hence, we don't want to count their uses of values
        // defined outside the region.
        if (ExitBlocks && ExitBlocks->count(BB))
          if (PHINode *PN = dyn_cast<PHINode>(&II))
            if (std::find(Blocks.begin(), Blocks.end(), PN->getIncomingBlock(*OI)) == Blocks.end())
              continue;
        if (definedInCaller(Blocks, *OI))
          Inputs.insert(*OI);
      }

      // Ignore outputs from exit blocks.
      if (!ExitBlocks || !ExitBlocks->count(BB)) {
        for (User *U : II.users()) {
          if (!definedInRegion(Blocks, U)) {
            // It looks like we have a use outside of the given blocks, but it's
            // possible for the use to appear in a basic block that is no longer
            // alive.  We use the DT to check that this use is still alive.
            if (Instruction *I = dyn_cast<Instruction>(U)) {
              if (DT.isReachableFromEntry(I->getParent())) {
                Outputs.insert(&II);
                break;
              }
            }
          }
        }
      }
    }
  }
}

// findInputsOutputs - Find inputs and outputs for Blocks.  Any blocks in
// ExitBlocks are handled in a special manner: PHI nodes in Exit Blocks are
// ignored when determining inputs.
// Handles lvalues (should be equivalent to rvalue code above)
template<class BasicBlockPtrContainer>
static inline void findInputsOutputs(const BasicBlockPtrContainer &Blocks,
                             ValueSet &Inputs, ValueSet &Outputs,
                             DominatorTree& DT,
                             const SmallPtrSetImpl<BasicBlock *> *ExitBlocks = nullptr) {
  for (BasicBlock *BB : Blocks) {
    // If a used value is defined outside the region, it's an input.  If an
    // instruction is used outside the region, it's an output.
    for (Instruction &II : *BB) {
      for (User::op_iterator OI = II.op_begin(), OE = II.op_end(); OI != OE;
           ++OI) {
        // The PHI nodes in each exit block will be updated after the exit block
        // is cloned.  Hence, we don't want to count their uses of values
        // defined outside the region.
        if (ExitBlocks && ExitBlocks->count(BB))
          if (PHINode *PN = dyn_cast<PHINode>(&II))
            if (std::find(Blocks.begin(), Blocks.end(), PN->getIncomingBlock(*OI)) == Blocks.end())
              continue;
        if (definedInCaller(Blocks, *OI))
          Inputs.insert(*OI);
      }

      // Ignore outputs from exit blocks.
      if (!ExitBlocks || !ExitBlocks->count(BB)) {
        for (User *U : II.users()) {
          if (!definedInRegion(Blocks, U)) {
            // It looks like we have a use outside of the given blocks, but it's
            // possible for the use to appear in a basic block that is no longer
            // alive.  We use the DT to check that this use is still alive.
            if (Instruction *I = dyn_cast<Instruction>(U)) {
              if (DT.isReachableFromEntry(I->getParent())) {
                Outputs.insert(&II);
                break;
              }
            }
          }
        }
      }
    }
  }
}

/// Clone Blocks into NewFunc, transforming the old arguments into references to
/// VMap values.
///
/// TODO: Fix the std::vector part of the type of this function.
void CloneIntoFunction(
    Function *NewFunc, const Function *OldFunc,
    std::vector<BasicBlock *> Blocks, ValueToValueMapTy &VMap,
    bool ModuleLevelChanges, SmallVectorImpl<ReturnInst *> &Returns,
    const StringRef NameSuffix,
    SmallPtrSetImpl<BasicBlock *> *ExitBlocks = nullptr,
    DISubprogram *SP = nullptr, ClonedCodeInfo *CodeInfo = nullptr,
    ValueMapTypeRemapper *TypeMapper = nullptr,
    ValueMaterializer *Materializer = nullptr);

/// Create a helper function whose signature is based on Inputs and
/// Outputs as follows: f(in0, ..., inN, out0, ..., outN)
///
/// TODO: Fix the std::vector part of the type of this function.
Function *CreateHelper(const ValueSet &Inputs,
                       const ValueSet &Outputs,
                       std::vector<BasicBlock *> Blocks,
                       BasicBlock *Header,
                       const BasicBlock *OldEntry,
                       const BasicBlock *OldExit,
                       ValueToValueMapTy &VMap,
                       Module *DestM,
                       bool ModuleLevelChanges,
                       SmallVectorImpl<ReturnInst *> &Returns,
                       const StringRef NameSuffix,
                       SmallPtrSetImpl<BasicBlock *> *ExitBlocks = nullptr,
                       const Instruction *InputSyncRegion = nullptr,
                       ClonedCodeInfo *CodeInfo = nullptr,
                       ValueMapTypeRemapper *TypeMapper = nullptr,
                       ValueMaterializer *Materializer = nullptr);

// Add alignment assumptions to parameters of outlined function, based on known
// alignment data in the caller.
void AddAlignmentAssumptions(
    const Function *Caller, const ValueSet &Inputs, ValueToValueMapTy &VMap,
    const Instruction *CallSite, AssumptionCache *AC, DominatorTree *DT);

} // End llvm namespace

#endif
