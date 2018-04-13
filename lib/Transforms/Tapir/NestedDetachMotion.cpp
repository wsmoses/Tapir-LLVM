#include "llvm/Transforms/Tapir.h"

#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Tapir/TapirUtils.h"
#include "llvm/IR/CFG.h"



using namespace llvm;

namespace {
struct NestedDetachMotion : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  NestedDetachMotion() : FunctionPass(ID) {
    initializeNestedDetachMotionPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
  }

  bool attemptDetachMotion(DetachInst* det, DominatorTree& DT, AliasAnalysis& AA) {
    bool changed = false;

    start:
      if (auto det2 = dyn_cast<DetachInst>(det->getDetached()->getTerminator())) {
        SmallPtrSet<BasicBlock *, 4> functionPieces;
        SmallVector<ReattachInst*, 4> reattachB;
        SmallPtrSet<BasicBlock *, 4> ExitBlocks;
        populateDetachedCFG(*det2, DT, functionPieces, reattachB, ExitBlocks, false);
        bool legal = true;

        AliasSetTracker CurAST(AA);
        for (Instruction &I : *det->getDetached()) {
          if (det2->getSyncRegion() == &I)
            continue;
          if (&I == det2)
            break;

          // Order of throws may be messed up
          if (I.mayThrow()) {
            legal = false;
            break;
          }

          CurAST.add(&I);

          for (User *U : I.users()) {
            if (Instruction *Inst = dyn_cast<Instruction>(U)) {
              if (functionPieces.count(Inst->getParent())) {
                legal = false;
                break;
              }
              //TODO also check if alias inside of
            }
          }
        }

        if (doesDetachedRegionAlias(CurAST, functionPieces))
          legal = false;

        if (legal) {
          changed = true;
          moveDetachInstBefore(det, *det2, reattachB, &DT, det->getSyncRegion());
          goto start;
        }
      }

    return changed;
  };

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    bool DetachingFunction = false;
    for (BasicBlock &BB : F)
      if (isa<DetachInst>(BB.getTerminator()))
        DetachingFunction = true;

    if (!DetachingFunction)
      return false;

    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

    bool Changed = false;
    tryMotion:
    for (BasicBlock &BB : F)
      if (auto det = dyn_cast<DetachInst>(BB.getTerminator())) {
        bool b = attemptDetachMotion(det, DT, AA);
        Changed |= b;
        if (b) goto tryMotion;

      }

    return Changed;
  }
};
}


char NestedDetachMotion::ID = 0;
static const char LS_NAME[] = "nesteddetach";
static const char ls_name[] = "Nested Detach Motion";
INITIALIZE_PASS_BEGIN(NestedDetachMotion, LS_NAME, ls_name, false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(NestedDetachMotion, LS_NAME, ls_name, false, false)

namespace llvm {
FunctionPass *createNestedDetachMotionPass() {
  return new NestedDetachMotion();
}
}
