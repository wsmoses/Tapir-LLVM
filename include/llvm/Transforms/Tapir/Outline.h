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
#ifndef LLVM_TRANSFORMS_OUTLINE_H
#define LLVM_TRANSFORMS_OUTLINE_H

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

// Clone Blocks into NewFunc, transforming the old arguments into references to
// VMap values.
//
/// TODO: Fix the std::vector part of the type of this function.
void CloneIntoFunction(Function *NewFunc, const Function *OldFunc,
                       std::vector<BasicBlock *> Blocks,
                       ValueToValueMapTy &VMap,
                       bool ModuleLevelChanges,
                       SmallVectorImpl<ReturnInst *> &Returns,
                       const StringRef NameSuffix,
                       ClonedCodeInfo *CodeInfo = nullptr,
                       ValueMapTypeRemapper *TypeMapper = nullptr,
                       ValueMaterializer *Materializer = nullptr);

  
/// Create a helper function whose signature is based on Inputs and
/// Outputs as follows: f(in0, ..., inN, out0, ..., outN)
///
/// TODO: Fix the std::vector part of the type of this function.
Function *CreateHelper(const SetVector<Value *> &Inputs,
                       const SetVector<Value *> &Outputs,
                       std::vector<BasicBlock *> Blocks,
                       BasicBlock *Header,
                       const BasicBlock *OldEntry,
                       const BasicBlock *OldExit,
                       ValueToValueMapTy &VMap,
                       Module *DestM,
                       bool ModuleLevelChanges,
                       SmallVectorImpl<ReturnInst *> &Returns,
                       const StringRef NameSuffix,
                       ClonedCodeInfo *CodeInfo = nullptr,
                       ValueMapTypeRemapper *TypeMapper = nullptr,
                       ValueMaterializer *Materializer = nullptr);

// Move static allocas in a cloned block into the entry block of helper.  Leave
// lifetime markers behind for those static allocas.  Returns true if the cloned
// block still contains dynamic allocas, which cannot be moved.
bool MoveStaticAllocasInClonedBlock(
    Function *Helper,
    BasicBlock *ClonedBlock,
    SmallVectorImpl<Instruction *> &ClonedExitPoints);

// Add alignment assumptions to parameters of outlined function, based on known
// alignment data in the caller.
void AddAlignmentAssumptions(const Function *Caller,
                             const SetVector<Value *> &Inputs,
                             ValueToValueMapTy &VMap,
                             const Instruction *CallSite,
                             AssumptionCache *AC,
                             DominatorTree *DT);

} // End llvm namespace

#endif
