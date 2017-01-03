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

#define DEBUG_TYPE "detach2cilk"

using namespace llvm;

namespace {

struct CilkPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  bool DisablePostOpts;
  bool Instrument;
  CilkPass(bool DisablePostOpts = false, bool Instrument = false)
      : ModulePass(ID), DisablePostOpts(DisablePostOpts),
        Instrument(Instrument) {
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
  bool processFunction(Function &F, DominatorTree &DT);
};
}  // End of anonymous namespace

static cl::opt<bool>  ClInstrumentCilk(
    "instrument-cilk", cl::init(false),
    cl::desc("Instrument Cilk events"), cl::Hidden);

cl::opt<bool>  fastCilk(
    "fast-cilk", cl::init(false),
    cl::desc("Attempt faster cilk call implementation"), cl::Hidden);

char CilkPass::ID = 0;
INITIALIZE_PASS_BEGIN(CilkPass, "detach2cilk", "Simple Lowering of Tapir to Cilk ABI", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(CilkPass, "detach2cilk", "Simple Lowering of Tapir to Cilk ABI",   false, false)

bool CilkPass::processFunction(Function &F, DominatorTree &DT) {
  if (verifyFunction(F, &errs())) {
    F.dump();
    assert(0);
  }

  bool Changed = false;
  if (fastCilk && F.getName()=="main") {
    IRBuilder<> start(F.getEntryBlock().getFirstNonPHIOrDbg());
    auto m = start.CreateCall(CILKRTS_FUNC(init, *F.getParent()));
    m->moveBefore(F.getEntryBlock().getTerminator());
  }

  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
    if (DetachInst* inst = dyn_cast_or_null<DetachInst>(i->getTerminator())) {
      cilk::createDetach(*inst, DT, ClInstrumentCilk || Instrument);
      Changed = true;
    }
  }

  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
    if (SyncInst* inst = dyn_cast_or_null<SyncInst>(i->getTerminator())) {
      cilk::createSync(*inst, ClInstrumentCilk || Instrument);
      Changed = true;
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
              Changed |= true;
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

  return Changed;
}

bool CilkPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  // Find functions that detach for processing.
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
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree();
    Changed |= processFunction(*F, DT);
    WorkList.pop_back();
  }
  return Changed;
}

// createPromoteDetachToCilkPass - Provide an entry point to create this pass.
//
ModulePass *llvm::createPromoteDetachToCilkPass(bool DisablePostOpts, bool Instrument) {
  return new CilkPass(DisablePostOpts, Instrument);
}
