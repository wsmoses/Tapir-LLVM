//===-- Scaffold.cpp ---------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements common infrastructure for libLLVMTapirOpts.a, which
// implements several transformations over the Tapir/LLVM intermediate
// representation, including the C bindings for that library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/LegacyPassManager.h"

#define TapirTarget_of(tt) ((TapirTarget*)tt)

using namespace llvm;

/// initializeScaffold - Initialize all passes linked into the
/// Scaffold library.
void llvm::initializeScaffold(PassRegistry &Registry) {
  initializeResourceCountPass(Registry);
  initializeGenRKQCPass(Registry);
  initializeGenQASMPass(Registry);
  initializeGenOpenQASMPass(Registry);
  initializeOptimizePass(Registry);
  initializeFlattenModulePass(Registry);
  initializeFunctionClonePass(Registry);
  initializeFunctionReversePass(Registry);
  initializeRotationsPass(Registry);
  initializeRTFreqEstHybPass(Registry);
  initializeRTResourceEst_MemPass(Registry);
  initializeRTResourceEstPass(Registry);
  initializeSortCloneArgumentsPass(Registry);
  initializeToffoliReplacePass(Registry);
  initializeXformCbitStoresPass(Registry);
}
