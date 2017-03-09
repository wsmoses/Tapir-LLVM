//===- SyncElimination.cpp - Eliminate unnecessary sync calls ----------------===//

#include "llvm/Transforms/Tapir.h"

#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"

using namespace llvm;

namespace {

struct SyncElimination : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid

  SyncElimination() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    errs() << "SyncElimination pass found function: " << F.getName() << "\n";
    F.setName("sync-elimination_"+F.getName());

    for (BasicBlock &BB: F)
      if (isa<SyncInst>(BB.getTerminator()))
      errs() << "Saw basicblock with sync instance " << BB.getName() << "\n";

    return true;
  }
};

}

char SyncElimination::ID = 0;
static RegisterPass<SyncElimination> X("sync-elimination", "Do sync-elimination's pass", false, false);

// Public interface to the SyncElimination pass
FunctionPass *llvm::createSyncEliminationPass() {
  return new SyncElimination();
}
