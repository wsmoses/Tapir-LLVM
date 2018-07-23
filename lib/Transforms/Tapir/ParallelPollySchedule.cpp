
#include "llvm/Transforms/Tapir.h"

#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "polly/ScopPass.h"

#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"

#include "llvm/IR/CFG.h"

using namespace llvm;
using namespace polly;

extern bool polly::PollyProcessUnprofitable;

namespace {

struct ParallelPollySchedule : public ScopPass {
  static char ID; // Pass identification, replacement for typeid
  ParallelPollySchedule() : ScopPass(ID) {
    initializeParallelPollySchedulePass(*PassRegistry::getPassRegistry());
  }

  /// Export the SCoP @p S to a JSON file.
  bool runOnScop(Scop &S) override {
    llvm::errs() << "running on scop!\n";
    S.print(llvm::errs());
    return false;
  }

  /// Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ScopInfoRegionPass>();
    AU.setPreservesAll();
  }
};

/*
struct ParallelPollySchedule : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  ParallelPollySchedule() : FunctionPass(ID) {
    initializeParallelPollySchedulePass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
      // We also need AA and RegionInfo when we are verifying analysis.
      AU.addRequiredTransitive<AAResultsWrapperPass>();
      AU.addRequiredTransitive<RegionInfoPass>();
      AU.setPreservesAll();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
        return false;
    return false;

    auto tmp = polly::PollyProcessUnprofitable;
    polly::PollyProcessUnprofitable = true;
    std::unique_ptr<polly::ScopDetection> Result;

    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &RI = getAnalysis<RegionInfoPass>().getRegionInfo();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
    Result.reset(new polly::ScopDetection(F, DT, SE, LI, RI, AA, ORE));
    polly::PollyProcessUnprofitable = tmp;
    return false;
  }
};
*/
}


char ParallelPollySchedule::ID = 0;
static const char LS_NAME[] = "parallelpollyschedule";
static const char ls_name[] = "Do polly transformations on tapir in llvm proper for mapping";
INITIALIZE_PASS_BEGIN(ParallelPollySchedule, LS_NAME, ls_name, false, false)
INITIALIZE_PASS_DEPENDENCY(ScopInfoRegionPass)
//INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
//INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
//INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
//INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
//INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
//INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(ParallelPollySchedule, LS_NAME, ls_name, false, false)

// Public interface to the RedundantSpawn pass
Pass *llvm::createParallelPollySchedulePass() {
  return new ParallelPollySchedule();
}
