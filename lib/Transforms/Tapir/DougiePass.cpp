//===- TailRecursionElimination.cpp - Eliminate Tail Calls ----------------===//

#include "llvm/Transforms/Tapir.h"

#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"

using namespace llvm;

namespace {
struct Dougie : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  Dougie() : FunctionPass(ID) {
    //initializeDougiePass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    //AU.addRequired<TargetTransformInfoWrapperPass>();
    //AU.addPreserved<GlobalsAAWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    errs() << "Dougie pass found function: " << F.getName() << "\n";
    F.setName("dougie_"+F.getName());

    for (BasicBlock &BB: F)
      if (isa<SyncInst>(BB.getTerminator()))
      errs() << "Saw basicblock with sync instance " << BB.getName() << "\n";

    return true;
  }
};
}

char Dougie::ID = 0;
static RegisterPass<Dougie> X("dougie", "Do dougie's pass", false, false);
//INITIALIZE_PASS_BEGIN(Dougie, "dougie", "Do dougie's pass",
//                      false, false)
//INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
//INITIALIZE_PASS_END(Dougie, "dougie", "Do dougie's pass",
//                    false, false)

// Public interface to the Dougie pass
FunctionPass *llvm::createDougiePass() {
  return new Dougie();
}

/*
PreservedAnalyses TailCallElimPass::run(Function &F,
                                        FunctionAnalysisManager &AM) {

  TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);

  bool Changed = eliminateTailRecursion(F, &TTI);

  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<GlobalsAA>();
  return PA;
}
*/
