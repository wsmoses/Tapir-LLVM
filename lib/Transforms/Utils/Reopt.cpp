//===- Reopt.cpp - The -detach2cilk pass, a wrapper around the Utils lib ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the Reopt function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"

#define DEBUG_TYPE "reopt"

using namespace llvm;

namespace {

struct Reopt : public FunctionPass {
	static char ID; // Pass identification, replacement for typeid
	Reopt() : FunctionPass(ID) {
	}

	bool runOnFunction(Function &F) override;

};
}  // end of anonymous namespace

char Reopt::ID = 0;
static RegisterPass<Reopt> X("reopt", "Remove DisableOpt flag", false, false);
INITIALIZE_PASS_BEGIN(Reopt, "reopt", "Remove DisableOpt flag", false, false)
INITIALIZE_PASS_END(Reopt, "reopt", "Remove DisableOpt flag",   false, false)

bool Reopt::runOnFunction(Function &F) {
	bool Changed  = false;
  if (F.hasFnAttribute(Attribute::DisableOpts)) {
    Changed = true;
    F.removeFnAttr(Attribute::DisableOpts);
  }
  if (F.hasFnAttribute(Attribute::RepeatLoopOpts)) {
    Changed = true;
    F.removeFnAttr(Attribute::RepeatLoopOpts);
  }
  return Changed;
}

// createReopt - Provide an entry point to create this pass.
//
FunctionPass *llvm::createReoptPass() {
	return new Reopt();
}
