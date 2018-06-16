
#include "llvm/Transforms/Tapir.h"

#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Tapir/TapirUtils.h"
#include "llvm/IR/CFG.h"

using namespace llvm;

//
// This pass transforms loops that contain branches on loop-invariant conditions
// to multiple loops.  For example, it turns the left into the right code:
//
//  spawn {                  if (lic)
//    A                          spawn { A; B; C }
//    if (lic) B             else
//    C                          spawn { A; C }
//  }

/*
spawn {
  if () {
     A()
  } else {
     B()
  }
}
*/
//
// This can increase the size of the code exponentially (doubling it every time
// a loop is unswitched) so we only unswitch if the resultant code will be
// smaller than a threshold.

// This requires

//
// This pass expects Detaching LICM to be run before it to hoist invariant conditions out
// of the loop, to make the unswitching opportunity obvious.

namespace {
struct DetachUnswitch : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  DetachUnswitch() : FunctionPass(ID) {
    initializeDetachUnswitchPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
  }

  bool attemptUnswitch(DetachInst* det, DominatorTree& DT, AliasAnalysis& AA) {
    bool changed = false;

    auto splitB = det->getDetached();
    auto term = splitB->getTerminator();
    if (!isa<BranchInst>(term)) return changed;
    if (term->getNumSuccessors() == 1) return changed;

    SmallVector<AllocaInst*, 4> allocas;
    SmallVector<CallInst*, 4> syncregions;

    for(Instruction& I : *splitB) {
        if (&I == term) continue;
        if (auto alloc = dyn_cast<AllocaInst>(&I)) {
            allocas.push_back(alloc);
            continue;
        }
        if (auto call = dyn_cast<CallInst>(&I)) {
            auto id = call->getCalledFunction()->getIntrinsicID();
            if (id == Intrinsic::syncregion_start) {
                syncregions.push_back(call);
                continue;
            }
        }
        if (!llvm::isConstantMemoryFreeOperation(&I, /*allowsyncregion=*/false))
            return changed;
    }

    //TODO move alloca initialization down if doesn't affect condition
    // currently we abort
    for(auto a : allocas) {
        auto UI = a->use_begin(), E = a->use_end();
        for (; UI != E;) {
         Use &U = *UI;
         ++UI;
         auto *Usr = dyn_cast<Instruction>(U.getUser());
         if (Usr->getParent() == splitB) {
             return changed;
         }
        }
    }

    size_t ns = term->getNumSuccessors();
    auto blocks =     new SmallPtrSet<BasicBlock *, 4>[ns];
    auto reattachB  = new SmallVector<ReattachInst*, 4>[ns];
    auto ExitBlocks = new SmallPtrSet<BasicBlock *, 4>[ns];
    bool valid = true;
    for(unsigned i=0; i<ns; i++) {
      populateDetachedCFG(term->getSuccessor(i), *det, DT, blocks[i], reattachB[i], ExitBlocks[i], false);
      for(unsigned j=0; j<i; j++) {
        for(auto b : blocks[i]) {
          if (blocks[j].count(b)) {
            valid = false;
            goto endpopulate;
          }
        }
      }
    }
  
    endpopulate:
    if(valid) {
        assert(term->getNumSuccessors() == ns);
        for(unsigned i=0; i<ns; i++) {
            assert(term->getNumSuccessors() == ns);
            auto newDetacher = term->getSuccessor(i);
            auto newDetached = newDetacher->splitBasicBlock(newDetacher->getFirstInsertionPt());
            changed = true;
            auto toReplace = DetachInst::Create(newDetached, det->getSuccessor(1), det->getSyncRegion());
            ReplaceInstWithInst(newDetacher->getTerminator(), toReplace);
            //TODO don't copy if no uses
            for(auto a : allocas) {
                auto newinst = a->clone();
                newinst->insertBefore(newDetached->getFirstNonPHIOrDbgOrLifetime());
                auto UI = a->use_begin(), E = a->use_end();
                bool used = false;
                for (; UI != E;) {
                    Use &U = *UI;
                    ++UI;
                    auto *Usr = dyn_cast<Instruction>(U.getUser());
                    if (Usr && ( blocks[i].count(Usr->getParent()) || Usr->getParent() == newDetached) ) {
                        U.set(newinst);
                        used = true;
                    }
                }
                if (!used) newinst->eraseFromParent();
            }
            for(auto a : syncregions) {
                auto newinst = a->clone();
                newinst->insertBefore(newDetached->getFirstNonPHIOrDbgOrLifetime());
                auto UI = a->use_begin(), E = a->use_end();
                bool used = false;
                for (; UI != E;) {
                    Use &U = *UI;
                    ++UI;
                    auto *Usr = dyn_cast<Instruction>(U.getUser());
                    if (Usr && ( blocks[i].count(Usr->getParent()) || Usr->getParent() == newDetached) ) {
                        U.set(newinst);
                        used = true;
                    }
                }
                if (!used) newinst->eraseFromParent();
            }
        }
        changed = true;
        for(auto a : allocas) {
            a->eraseFromParent();
        }
        for(auto a : syncregions) {
            a->eraseFromParent();
        }
        auto toReplace = BranchInst::Create(det->getDetached());
        ReplaceInstWithInst(det, toReplace);
    }
    delete[] blocks;
    delete[] reattachB;
    delete[] ExitBlocks;

    return changed;
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

    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

    //TODO motion non memory ops
    bool Changed = false;
    tryMotion:
    for (BasicBlock &BB : F)
      if (auto det = dyn_cast<DetachInst>(BB.getTerminator())) {
        bool b = attemptUnswitch(det, DT, AA);
        Changed |= b;
        if (b) goto tryMotion;
      }

    return Changed;
  }
};
}


char DetachUnswitch::ID = 0;
static const char LS_NAME[] = "detachunswitch";
static const char ls_name[] = "Detach Unswitching";
INITIALIZE_PASS_BEGIN(DetachUnswitch, LS_NAME, ls_name, false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(DetachUnswitch, LS_NAME, ls_name, false, false)

namespace llvm {
FunctionPass *createDetachUnswitchPass() {
  return new DetachUnswitch();
}
}
