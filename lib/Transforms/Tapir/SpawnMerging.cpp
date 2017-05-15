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

  BasicBlock *getSpawnExitBlock(BasicBlock &DetachBlock, BasicBlock &ContinueBlock) {
    for (BasicBlock *Pred: predecessors(&ContinueBlock)) {
      if (isa<ReattachInst>(Pred->getTerminator())) {
        // TODO check that DetachBlock dominates it
        // TODO is it possible to have more than one?
        return Pred;
      }
    }

    llvm_unreachable("Cannot find spawn exit block. ");
  }

  void replaceTerminatorWithBranch(BasicBlock *BB) {
    TerminatorInst *terminator = BB->getTerminator();
    BasicBlock *successor = terminator->getSuccessor(0);
    IRBuilder<> Builder(terminator);
    Builder.CreateBr(successor);
    terminator->eraseFromParent();
  }

  bool processBlock(BasicBlock &DetachBlock) {
    errs() << "SpawnMerging: Found sync block: " << DetachBlock.getName() << "\n";

    if (!isa<DetachInst>(DetachBlock.getTerminator())) {
      errs() << "SpawnMerging: Not a detach terminated block.\n";
      return false;
    }

    DetachInst *DetachBlockTerminator = dyn_cast<DetachInst>(DetachBlock.getTerminator());
    BasicBlock *ContinueBlock = DetachBlockTerminator->getContinue();

    if (!isa<DetachInst>(ContinueBlock->getTerminator())) {
      errs() << "SpawnMerging: Continue block is not a detach terminated block.\n";
      return false;
    }

    DetachInst *ContinueBlockDetach = dyn_cast<DetachInst>(ContinueBlock->getTerminator());
    BasicBlock *EndBlock = ContinueBlockDetach->getContinue();

    // Reattach to the end of the second spawn
    DetachBlockTerminator->setSuccessor(1, EndBlock);

    // Don't reattach after the first spawn
    BasicBlock *SpawnExitBlock = getSpawnExitBlock(DetachBlock, *ContinueBlock);
    replaceTerminatorWithBranch(SpawnExitBlock);

    // Don't detach after the first spawn
    replaceTerminatorWithBranch(ContinueBlock);

    errs() << "SpawnMerging: Merged two spawns. YEAHHHHHHHHHHH!!! \n";

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
