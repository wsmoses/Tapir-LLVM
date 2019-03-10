//===- TapirCleanup - Cleanup leftover Tapir tasks for code generation ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass serializes any remaining Tapir instructions before code generation.
// Typically this pass should have no effect, because Tapir instructions should
// have been lowered already to a particular parallel runtime.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
//#include "llvm/IR/Instructions.h"
//#include "llvm/IR/Module.h"
//#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
//#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
//#include <cstddef>

using namespace llvm;

#define DEBUG_TYPE "tapircleanup"

STATISTIC(NumTasksSerialized, "Number of Tapir tasks serialized");

namespace {

  class TapirCleanup : public FunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid.

    TapirCleanup() : FunctionPass(ID) {}

    bool runOnFunction(Function &Fn) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override;

    StringRef getPassName() const override {
      return "Tapir last-minute cleanup for CodeGen";
    }
  };

} // end anonymous namespace

char TapirCleanup::ID = 0;

INITIALIZE_PASS_BEGIN(TapirCleanup, DEBUG_TYPE,
                      "Cleanup Tapir", false, false)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_END(TapirCleanup, DEBUG_TYPE,
                    "Cleanup Tapir", false, false)

FunctionPass *llvm::createTapirCleanupPass() { return new TapirCleanup(); }

void TapirCleanup::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TaskInfoWrapperPass>();
}

bool TapirCleanup::runOnFunction(Function &F) {
  TaskInfo &TI = getAnalysis<TaskInfoWrapperPass>().getTaskInfo();
  if (TI.isSerial())
    return false;

  // If we haven't lowered the Tapir task to a particular parallel runtime by
  // this point, simply serialize the task.
  for (Task *T : post_order(TI.getRootTask())) {
    if (T->isRootTask())
      continue;
    SerializeDetach(T->getDetach(), T);
    NumTasksSerialized++;
  }

  return true;
}
