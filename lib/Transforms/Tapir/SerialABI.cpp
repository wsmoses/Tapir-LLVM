//===- SerialABI.cpp - Replace Tapir with Serial Elison -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SerialABI interface, which is used to convert Tapir
// instructions -- detach, reattach, and sync -- to calls into the Serial
// runtime system.  
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/SerialABI.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "serialabi"

SerialABI::SerialABI() { }
SerialABI::~SerialABI() { }

void SerialABI::createSync(SyncInst &SI, ValueToValueMapTy &DetachCtxToStackFrame) {}

Function* SerialABI::createDetach(DetachInst &detach,
				    ValueToValueMapTy &DetachCtxToStackFrame,
				    DominatorTree &DT, AssumptionCache &AC) { return nullptr; }

void SerialABI::preProcessFunction(Function &F) {
  for(BasicBlock &BB : F){
    Instruction* term = BB.getTerminator();
    if (DetachInst *de = dyn_cast<DetachInst>(term)) 
      ReplaceInstWithInst(de, BranchInst::Create(de->getDetached()));
    else if (ReattachInst *re = dyn_cast<ReattachInst>(term) )
      ReplaceInstWithInst(re, BranchInst::Create(re->getDetachContinue()));
    else if (SyncInst *si = dyn_cast<SyncInst>(term))
      ReplaceInstWithInst(si, BranchInst::Create(si->getSuccessor(0)));
  }
}

void SerialABI::postProcessFunction(Function &F) {}

void SerialABI::postProcessHelper(Function &F) {}

bool SerialABI::processMain(Function &F) {return false;}

