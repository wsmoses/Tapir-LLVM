//===- Detach2Cilk.cpp - The -detach2cilk pass, a wrapper around the Utils lib ----===//
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

#include "llvm/Transforms/CilkABI.h"

#define DEBUG_TYPE "detach2cilk"

using namespace llvm;

namespace {

struct CilkPass : public FunctionPass {
	static char ID; // Pass identification, replacement for typeid
	CilkPass() : FunctionPass(ID) {
	}

	// runOnFunction - To run this pass, first we calculate the alloca
	// instructions that are safe for promotion, then we promote each one.
	//
	bool runOnFunction(Function &F) override;

	void getAnalysisUsage(AnalysisUsage &AU) const override {
		//AU.addRequired<AssumptionCacheTracker>();
		//AU.addRequired<DominatorTreeWrapperPass>();
		//AU.setPreservesCFG();
		// This is a cluster of orthogonal Transforms
		//AU.addPreserved<UnifyFunctionExitNodes>();
		//AU.addPreservedID(LowerSwitchID);
		//AU.addPreservedID(LowerInvokePassID);
	}
};
}  // end of anonymous namespace

static cl::opt<bool>  ClInstrumentCilk(
    "instrument-cilk", cl::init(false),
    cl::desc("Instrument Cilk events"), cl::Hidden);

char CilkPass::ID = 0;
static RegisterPass<CilkPass> X("detach2cilk", "Promote Detach to Cilk Runtime", false, false);
//INITIALIZE_PASS_BEGIN(CilkPass, "detach2cilk", "Promote Detach to Cilk Runtime",
//                false, false)
//INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
//INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
//INITIALIZE_PASS_END(CilkPass, "detach2cilk", "Promote Detach to Cilk Runtime",
//                false, false)

bool CilkPass::runOnFunction(Function &F) {

	bool Changed  = false;
	for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
		TerminatorInst* term = i->getTerminator();
		if( term == nullptr ) continue;
		if( DetachInst* inst = llvm::dyn_cast<DetachInst>(term) ) {
		  llvm::cilk::createDetach(*inst, ClInstrumentCilk);
		  Changed = true;
		} else continue;
	}

	for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
		TerminatorInst* term = i->getTerminator();
		if( term == nullptr ) continue;
    		if( SyncInst* inst = llvm::dyn_cast<SyncInst>(term) ) {
		  llvm::cilk::createSync(*inst, ClInstrumentCilk);
		}
	}

	return Changed;
}

// createPromoteMemoryToRegister - Provide an entry point to create this pass.
//
FunctionPass *llvm::createPromoteDetachToCilkPass() {
	return new CilkPass();
}
