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
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"

#define DEBUG_TYPE "detach2cilk"

using namespace llvm;

namespace {

struct CilkPass : public FunctionPass {
	static char ID; // Pass identification, replacement for typeid
  bool DisablePostOpts;
  bool Instrument;
	CilkPass(bool disablePostOpts=false, bool instrument=false) : FunctionPass(ID), DisablePostOpts(disablePostOpts), Instrument(instrument) {
	}

	// runOnFunction - To run this pass, first we calculate the alloca
	// instructions that are safe for promotion, then we promote each one.
	//
	bool runOnFunction(Function &F) override;

	void getAnalysisUsage(AnalysisUsage &AU) const override {
		//AU.addRequired<AssumptionCacheTracker>();
		AU.addRequired<DominatorTreeWrapperPass>();
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

cl::opt<bool>  fastCilk(
    "fast-cilk", cl::init(false),
    cl::desc("Attempt faster cilk call implementation"), cl::Hidden);

char CilkPass::ID = 0;
static RegisterPass<CilkPass> X("detach2cilk", "Promote Detach to Cilk Runtime", false, false);
INITIALIZE_PASS_BEGIN(CilkPass, "detach2cilk", "Promote Detach to Cilk Runtime", false, false)
//INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(CilkPass, "detach2cilk", "Promote Detach to Cilk Runtime",   false, false)

bool CilkPass::runOnFunction(Function &F) {
	bool Changed  = false;
  if (fastCilk && F.getName()=="main") {
    IRBuilder<> start(F.getEntryBlock().getFirstNonPHIOrDbg());
    auto m = start.CreateCall(CILKRTS_FUNC(init, *F.getParent()));
    m->moveBefore(F.getEntryBlock().getTerminator());
  }
  
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
	for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
		TerminatorInst* term = i->getTerminator();
		if( term == nullptr ) continue;
		if( DetachInst* inst = llvm::dyn_cast<DetachInst>(term) ) {
          //errs() << "<D2C>\n";
          //F.dump();
          //errs() << "</PRE>\n";
      //if (!Changed) F.dump();
		  llvm::cilk::createDetach(*inst, DT, ClInstrumentCilk || Instrument);
          //errs() << "<POST>\n";
          //F.dump();
          if( llvm::verifyFunction(F, nullptr) ) {
            F.dump(); 
          }
          assert( !llvm::verifyFunction(F, &llvm::errs()) );

          //errs() << "</D2C>\n";
		  Changed = true;
		} else continue;
	}

  if (Changed)
	for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
		TerminatorInst* term = i->getTerminator();
		if( term == nullptr ) continue;
    		if( SyncInst* inst = llvm::dyn_cast<SyncInst>(term) ) {
    		  llvm::cilk::createSync(*inst, ClInstrumentCilk || Instrument);

          if( llvm::verifyFunction(F, nullptr) ) {
            F.dump(); 
          }
          assert( !llvm::verifyFunction(F, &llvm::errs()) );
		}
	}

  bool inlining = true;
  while (inlining) {
    inlining = false;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
      if (auto cal = dyn_cast<CallInst>(&*I)) {
        if (auto fn = cal->getCalledFunction()) {
          if (fn->getName().startswith("__cilk")) {
            InlineFunctionInfo ifi;
            if (InlineFunction(cal,ifi)) {
              if (fn->getNumUses()==0) fn->eraseFromParent();
              Changed |= true;
              inlining = true;
              break;
            } else {
              //llvm::errs() << "failed to inline function: " << fn->getName() << " in " << F.getName() << "\n";
            }
          }
        }
      }
  }

  if (Changed) {
    if (llvm::verifyFunction(F, nullptr)) {
      F.dump(); 
    }
  }

  assert( !llvm::verifyFunction(F, &llvm::errs()) );
  
	return Changed;
}

// createPromoteMemoryToRegister - Provide an entry point to create this pass.
//
FunctionPass *llvm::createPromoteDetachToCilkPass(bool DisablePostOpts, bool Instrument) {
	return new CilkPass(DisablePostOpts, Instrument);
}
