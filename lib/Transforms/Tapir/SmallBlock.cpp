
#include "llvm/Transforms/Tapir.h"

#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Tapir/TapirUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

namespace {
struct SmallBlock : public FunctionPass {
  static const int threshold = 10;
  static char ID; // Pass identification, replacement for typeid
  SmallBlock() : FunctionPass(ID) {
    initializeSmallBlockPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
  }

  bool attemptSmallBlock(DetachInst* det, DominatorTree& DT) {
    //TODO generalize to handle if/etc (generally things without loops)
    //TODO cost model
    BasicBlock* current = det->getDetached();
    size_t cost = 0;
    while(current) {
        for(Instruction& I : *current) {
            if (isa<TerminatorInst>(&I)) break;
            if (!isConstantOperation(&I, /*bool allowsyncregion=*/true))
                return false;
            cost += 1;
        }
        auto term = current->getTerminator();
        if (term->getNumSuccessors() != 1) return false;
        if (isa<BranchInst>(term)) {
            current = term->getSuccessor(0);
            cost += 1;
        } else if (isa<ReattachInst>(term)) {
            break;
        } else {
            return false;
        }
    }
    if (cost > 20) {
        return false;
    }
    SerializeDetachedCFG(det, &DT);
    return true;
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    bool DetachingFunction = false;
    for (BasicBlock &BB : F)
      if (isa<DetachInst>(BB.getTerminator()))
        DetachingFunction = true;

    if (!DetachingFunction)
      return false;

    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

    //TODO motion non memory ops
    bool Changed = false;
    tryMotion:
    for (BasicBlock &BB : F)
      if (auto det = dyn_cast<DetachInst>(BB.getTerminator())) {
        bool b = attemptSmallBlock(det, DT);
        Changed |= b;
        if (b) goto tryMotion;
      }

    return Changed;
  }
};
}



char SmallBlock::ID = 0;
static const char LS_NAME[] = "smallblock";
static const char ls_name[] = "Small Block Elimination";
INITIALIZE_PASS_BEGIN(SmallBlock, LS_NAME, ls_name, false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(SmallBlock, LS_NAME, ls_name, false, false)

namespace llvm {
FunctionPass *createSmallBlockPass() {
  return new SmallBlock();
}
}
