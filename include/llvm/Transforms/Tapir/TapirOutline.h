//===- llvm/Transforms/Tapir/Outlining.h - Outlining for Tapir -*- C++ -*--===//
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
#ifndef LLVM_TRANSFORMS_TAPIR_OUTLINING_H
#define LLVM_TRANSFORMS_TAPIR_OUTLINING_H

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/IR/IRBuilder.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/Support/Debug.h"

namespace llvm {

// Clone Blocks into NewFunc, transforming the old arguments into references to
// VMap values.
//
/// TODO: Fix the std::vector part of the type of this function.
void CloneIntoFunction(Function *NewFunc, const Function *OldFunc,
                       std::vector<BasicBlock*> Blocks,
                       ValueToValueMapTy &VMap,
                       bool ModuleLevelChanges,
                       SmallVectorImpl<ReturnInst*> &Returns,
                       const char *NameSuffix,
                       ClonedCodeInfo *CodeInfo = nullptr,
                       ValueMapTypeRemapper *TypeMapper = nullptr,
                       ValueMaterializer *Materializer = nullptr);

  
/// Create a helper function whose signature is based on Inputs and
/// Outputs as follows: f(in0, ..., inN, out0, ..., outN)
///
/// TODO: Fix the std::vector part of the type of this function.
Function *CreateHelper(const SetVector<Value *> &Inputs,
                       const SetVector<Value *> &Outputs,
                       std::vector<BasicBlock*> Blocks,
                       BasicBlock *Header,
                       const BasicBlock *OldEntry,
                       const BasicBlock *OldExit,
                       ValueToValueMapTy &VMap,
                       Module *DestM,
                       bool ModuleLevelChanges,
                       SmallVectorImpl<ReturnInst*> &Returns,
                       const char *NameSuffix,
                       ClonedCodeInfo *CodeInfo = nullptr,
                       ValueMapTypeRemapper *TypeMapper = nullptr,
                       ValueMaterializer *Materializer = nullptr);
} // End llvm namespace

#endif
