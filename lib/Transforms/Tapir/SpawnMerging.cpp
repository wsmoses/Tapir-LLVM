//===- SpawnMerging.cpp - Merge adjacent spawns if it's worth it ----------------===//

#include "llvm/Transforms/Tapir.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/CFG.h"
#include "llvm/ADT/SmallSet.h"

#include <deque>
#include <map>

using namespace llvm;

namespace {

typedef SmallSet<const BasicBlock *, 32> BasicBlockSet;
typedef std::deque<const BasicBlock *> BasicBlockDeque;

struct SpawnMerging : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid

  SpawnMerging() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    errs() << "SpawnMerging: Found function: " << F.getName() << "\n";

    bool ChangedAny = false;

    while (true) {
      bool Changed = false;

      for (BasicBlock &block: F) {
        if (processBlock(block)) {
          Changed = true;
          ChangedAny = true;
          break;
        }
      }

      if (!Changed) {
        break;
      }
    }

    return ChangedAny;
  }

private:
  bool processBlock(BasicBlock &BB) {
    errs() << "SpawnMerging: Found sync block: " << BB.getName() << "\n";

    if (!isa<DetachInst>(block.getTerminator())) {
      errs() << "Not a detach terminated block.\n";
      return false;
    }

    DetachInst *BBDetach = dyn_cast<DetachInst>(block.getTerminator());
    BasicBlock *ContinueBlock = DetachInst->getContinue();

    if (!isa<DetachInst>(ContinueBlock.getTerminator)) {
      errs() << "Continue block is not a detach terminated block.\n";
      return false;
    }

    DetachInst *ContinueBlockDetach = dyn_cast<DetachInst>(ContinueBlock.getTerminator());
    BasicBlock *EndBlock = ContinueBlockDetach->getContinue();

    // Reattach to the end of the second spawn
    BBDetach.setSuccessor(1, EndBlock);

    // Don't reattach after the first spawn
    // TODO

    // Don't detach after the first spawn
    // TODO
    // IRBuilder<> Builder(Sync);
    // Builder.CreateBr(suc);
    // Sync->eraseFromParent();

    return true;
  }
};

}

char SpawnMerging::ID = 0;
static RegisterPass<SpawnMerging> X("spawn-merging", "Do spawn-merging's pass", false, false);

// Public interface to the SpawnMerging pass
FunctionPass *llvm::createSpawnMergingPass() {
  return new SpawnMerging();
}
