//===- Detach2Cilk.cpp - Convert Tapir into Cilk runtime calls ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass converts functions that include Tapir instructions to call out to
// the Cilk runtime system.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/CilkABI.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Tapir.h"

#define DEBUG_TYPE "tapir2cilk"

using namespace llvm;

static cl::opt<bool> ClInstrumentCilk("instrument-cilk", cl::init(false),
                                      cl::Hidden,
                                      cl::desc("Instrument Cilk events"));

cl::opt<bool> fastCilk("fast-cilk", cl::init(false), cl::Hidden,
                       cl::desc("Attempt faster cilk call implementation"));

namespace {

struct LowerTapirToCilk : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  bool DisablePostOpts;
  bool Instrument;
  explicit LowerTapirToCilk(bool DisablePostOpts = false, bool Instrument = false)
      : ModulePass(ID), DisablePostOpts(DisablePostOpts),
        Instrument(Instrument) {
    initializeLowerTapirToCilkPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Simple Lowering of Tapir to Cilk ABI";
  }

  // // runOnFunction - To run this pass, first we find appropriate instructions,
  // // then we promote each one.
  // //
  // bool runOnFunction(Function &F) override;

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
  }
private:
  ValueToValueMapTy DetachCtxToStackFrame;
  SmallVectorImpl<Function *> *processFunction(Function &F, DominatorTree &DT);
};
}  // End of anonymous namespace

char LowerTapirToCilk::ID = 0;
INITIALIZE_PASS_BEGIN(LowerTapirToCilk, "tapir2cilk",
                      "Simple Lowering of Tapir to Cilk ABI", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(LowerTapirToCilk, "tapir2cilk",
                    "Simple Lowering of Tapir to Cilk ABI", false, false)

SmallVectorImpl<Function *>
*LowerTapirToCilk::processFunction(Function &F, DominatorTree &DT) {
  // if (verifyFunction(F, &errs())) {
  //   F.dump();
  //   assert(0);
  // }

  if (fastCilk && F.getName()=="main") {
    IRBuilder<> start(F.getEntryBlock().getFirstNonPHIOrDbg());
    auto m = start.CreateCall(CILKRTS_FUNC(init, *F.getParent()));
    m->moveBefore(F.getEntryBlock().getTerminator());
  }

  SmallVector<Function *, 4> *NewHelpers = new SmallVector<Function *, 4>();
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    if (DetachInst* DI = dyn_cast_or_null<DetachInst>(I->getTerminator())) {
      Function *Helper = cilk::createDetach(*DI, DetachCtxToStackFrame, DT,
                                            ClInstrumentCilk || Instrument);
      // Check new helper function to see if it must be processed.
      NewHelpers->push_back(Helper);
    } else if (SyncInst* SI = dyn_cast_or_null<SyncInst>(I->getTerminator())) {
      cilk::createSync(*SI, DetachCtxToStackFrame,
                       ClInstrumentCilk || Instrument);
    }
  }

  if (verifyFunction(F, &errs())) {
    F.dump();
    assert(0);
  }

  bool inlining = true;
  while (inlining) {
    inlining = false;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (auto cal = dyn_cast<CallInst>(&*I)) {
        if (auto fn = cal->getCalledFunction()) {
          if (fn->getName().startswith("__cilk")) {
            InlineFunctionInfo ifi;
            if (InlineFunction(cal,ifi)) {
              if (fn->getNumUses()==0) fn->eraseFromParent();
              inlining = true;
              break;
            }
          }
        }
      }
    }
  }

  if (verifyFunction(F, &errs())) {
    F.dump();
    assert(0);
  }
  return NewHelpers;
}

bool LowerTapirToCilk::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  // Add functions that detach to the work list.
  SmallVector<Function *, 4> WorkList;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      if (isa<DetachInst>(BB.getTerminator()))
        WorkList.push_back(&F);

  if (WorkList.empty())
    return false;

  bool Changed = false;
  while (!WorkList.empty()) {
    // Process the next function.
    Function *F = WorkList.back();
    WorkList.pop_back();
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree();
    SmallVectorImpl<Function *> *NewHelpers = processFunction(*F, DT);
    Changed |= !NewHelpers->empty();
    // Check the generated helper functions to see if any need to be processed.
    for (Function *Helper : *NewHelpers)
      for (BasicBlock &BB : *Helper)
        if (isa<DetachInst>(BB.getTerminator()))
          WorkList.push_back(Helper);
  }
  return Changed;
}

// createLowerTapirToCilkPass - Provide an entry point to create this pass.
//
namespace llvm {
ModulePass *createLowerTapirToCilkPass(bool DisablePostOpts, bool Instrument) {
  return new LowerTapirToCilk(DisablePostOpts, Instrument);
}
}
