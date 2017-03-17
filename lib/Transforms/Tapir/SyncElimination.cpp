//===- SyncElimination.cpp - Eliminate unnecessary sync calls ----------------===//

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

struct SyncElimination : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid

  SyncElimination() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
  }

  // We will explain what Rosetta and Vegas are later. Or rename them.
  // We promise.

  // Rosetta-finding code

  void findRosetta(const BasicBlock &BB, BasicBlockSet &OutputSet) {
    assert(isa<SyncInst>(BB.getTerminator()));

    BasicBlockSet Visited;
    BasicBlockDeque Frontier;
    std::map<const BasicBlock *, int> DetachLevel;

    DetachLevel[&BB] = 0;
    Frontier.push_back(&BB);

    while (!Frontier.empty()) {
      const BasicBlock *Current = Frontier.front();
      Frontier.pop_front();

      for (const BasicBlock *Pred: predecessors(Current)) {
        if (Visited.count(Pred) > 0) {
          continue;
        }

        if (isa<SyncInst>(Pred->getTerminator())) {
          continue;
        }

        Visited.insert(Pred);

        DetachLevel[Pred] = DetachLevel[Current];

        if (isa<ReattachInst>(Pred->getTerminator())) {
          DetachLevel[Pred] ++;
        } else if (isa<DetachInst>(Pred->getTerminator())) {
          DetachLevel[Pred] --;
        }

        if (DetachLevel[Pred] > 0) {
          OutputSet.insert(Pred);
        }

        if (DetachLevel[Pred] >= 0) {
          Frontier.push_back(Pred);
        }
      }
    }
  }

  // Vegas-finding code
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

        Visited.insert(Succ);
        OutputSet.insert(Succ);

        // We need to include blocks whose terminator is another sync.
        // Therefore we still insert the block into OutputSet in this case.
        // However we do not search any further past the sync block.
        if (!isa<SyncInst>(Succ->getTerminator())) {
          Frontier.push_back(Succ);
        }
      }
    }
  }

  bool isSyncEliminationLegal(const BasicBlockSet &RosettaSet, const BasicBlockSet &VegasSet) {
    AliasAnalysis *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

    for (const BasicBlock *RBB : RosettaSet) {
      for (const Instruction &RI : *RBB) {
        for (const BasicBlock *VBB : VegasSet) {
          for (const Instruction &VI : *VBB) {
            ImmutableCallSite RC(&RI), VC(&VI);

            if (!RC ||
                !VC ||
                AA->getModRefInfo(RC, VC) != MRI_NoModRef ||
                AA->getModRefInfo(VC, RC) != MRI_NoModRef) {
              errs() << "SyncElimination:     Conflict found between " << RI << " and " << VI << "\n";
              return false;
            }
          }
        }
      }
    }

    return true;
  }

  bool processSyncInstBlock(BasicBlock &BB) {
    errs() << "SyncElimination: Found sync block: " << BB.getName() << "\n";

    BasicBlockSet RosettaSet, VegasSet;

    findRosetta(BB, RosettaSet);
    findVegas(BB, VegasSet);

    errs() << "SyncElimination:     Blocks found in the Rosetta set: " << "\n";
    for (const BasicBlock *BB: RosettaSet) {
      errs() << "SyncElimination:         " + BB->getName() << "\n";
    }

    errs() << "SyncElimination:     Blocks found in the Vegas set: " << "\n";
    for (const BasicBlock *BB: VegasSet) {
      errs() << "SyncElimination:         " + BB->getName() << "\n";
    }

    if (isSyncEliminationLegal(RosettaSet, VegasSet)) {
      SyncInst *Sync = dyn_cast<SyncInst>(BB.getTerminator());
      assert(Sync != NULL);
      BasicBlock* suc = Sync->getSuccessor(0);
      IRBuilder<> Builder(Sync);
      Builder.CreateBr(suc);
      Sync->eraseFromParent();
      errs() << "SyncElimination:     A sync is removed. " << "\n";
      return true;
    }

    return false;
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    errs() << "SyncElimination: Found function: " << F.getName() << "\n";
    F.setName("sync-elimination_"+F.getName());

    bool ChangedAny = false;
    for (bool Changed = false; Changed; Changed = false) {
      for (BasicBlock &block: F) {
        if (isa<SyncInst>(block.getTerminator())) {
          if (processSyncInstBlock(block)) {
            Changed = true;
            ChangedAny = true;
            break;
          }
        }
      }
    }

    return ChangedAny;
  }
};

}

char SyncElimination::ID = 0;
static RegisterPass<SyncElimination> X("sync-elimination", "Do sync-elimination's pass", false, false);

// Public interface to the SyncElimination pass
FunctionPass *llvm::createSyncEliminationPass() {
  return new SyncElimination();
}
