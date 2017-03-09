//===- SyncElimination.cpp - Eliminate unnecessary sync calls ----------------===//

#include "llvm/Transforms/Tapir.h"

#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/CFG.h"
#include "llvm/ADT/SmallSet.h"

#include <deque>

using namespace llvm;

namespace {

typedef SmallSet<const BasicBlock *, 32> BasicBlockSet;
typedef std::deque<const BasicBlock *> BasicBlockDeque;

struct SyncElimination : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid

  SyncElimination() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
  }

  // We will explain what Rosetta and Vegas are later. Or rename them.
  // We promise.

  // Rosetta-finding code
  // TODO

  // Vegas-finding code
  //
  //
  // We run BFS starting from the sync block, following all foward edges, and stop a branch whenever
  // we hit another sync block.

  void findVegas(const BasicBlock &BB, BasicBlockSet &OutputSet) {
    assert(isa<SyncInst>(BB.getTerminator()));

    BasicBlockSet Visited;
    BasicBlockDeque Frontier;

    Frontier.push_back(&BB);

    while (!Frontier.empty()) {
      const BasicBlock *Current = Frontier.front();
      Frontier.pop_front();

      for (const BasicBlock *Succ: successors(Current)) {
        if (Visited.count(Succ) > 0) {
          continue;
        }

        OutputSet.insert(Succ);
        Visited.insert(Succ);

        // We need to include blocks whose terminator is another sync.
        // Therefore we still insert the block into OutputSet in this case.
        // However we do not search any further past the sync block.
        if (!isa<SyncInst>(Succ->getTerminator())) {
          Frontier.push_back(Succ);
        }
      }
    }
  }

  // Entry point code

  void processSyncInstBlock(BasicBlock &BB) {
    errs() << "SyncElimination: Found sync block: " << BB.getName() << "\n";

    BasicBlockSet RosettaSet, VegasSet;

    findVegas(BB, VegasSet);

    errs() << "SyncElimination:     Blocks found in the Vegas set: " << "\n";
    for (const BasicBlock *BB: VegasSet) {
      errs() << "SyncElimination:         " + BB->getName() << "\n";
    }
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    errs() << "SyncElimination: Found function: " << F.getName() << "\n";
    F.setName("sync-elimination_"+F.getName());

    for (BasicBlock &block: F)
      if (isa<SyncInst>(block.getTerminator()))
        processSyncInstBlock(block);

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
