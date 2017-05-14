
#include "llvm/Transforms/Tapir.h"

#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/CFG.h"

using namespace llvm;

namespace {
struct SpawnUnswitch : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  SpawnUnswitch() : FunctionPass(ID) {
    //initializeSpawnUnswitchPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    //AU.addRequired<TargetTransformInfoWrapperPass>();
    //AU.addPreserved<GlobalsAAWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    F.setName("SpawnUnswitch_"+F.getName());

    
    return true;
  }
};
}

char SpawnUnswitch::ID = 0;
static RegisterPass<SpawnUnswitch> X("spawnunswitch", "Do SpawnUnswitch pass", false, false);

// Public interface to the RedundantSpawn pass
FunctionPass *llvm::createSpawnUnswitchPass() {
  return new SpawnUnswitch();
}
