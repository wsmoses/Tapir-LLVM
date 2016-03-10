//===- Loop2Cilk.cpp - Induction Variable Elimination ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation analyzes and transforms the induction variables (and
// computations derived from them) into simpler forms suitable for subsequent
// analysis and transformation.
//
// If the trip count of a loop is computable, this pass also makes the following
// changes:
//   1. The exit condition for the loop is canonicalized to compare the
//      induction value against the exit value.  This turns loops like:
//        'for (i = 7; i*i < 1000; ++i)' into 'for (i = 0; i != 25; ++i)'
//   2. Any use outside of the loop of an expression derived from the indvar
//      is changed to compute the derived value outside of the loop, eliminating
//      the dependence on the exit value of the induction variable.  If the only
//      purpose of the loop is to compute the exit value of some derived
//      expression, this transformation will make the loop dead.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"
#include "llvm/Transforms/CilkABI.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

using namespace llvm;

#define DEBUG_TYPE "loop2cilk"

namespace {
  class Loop2Cilk : public LoopPass {

  public:

    static char ID; // Pass identification, replacement for typeid
    Loop2Cilk()
        : LoopPass(ID)
//, LI(nullptr), SE(nullptr), DT(nullptr), Changed(false)
{
      //initializeIndVarSimplifyPass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
//      AU.addRequired<PromotePass>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
//      AU.addPreserved<ScalarEvolutionWrapperPass>();
//      AU.addPreservedID(LoopSimplifyID);
//      AU.addPreservedID(LCSSAID);
//      AU.setPreservesCFG();
    }

  private:
    void releaseMemory() override {
    }

  };
}

char Loop2Cilk::ID = 0;
static RegisterPass<Loop2Cilk> X("loop2cilk", "Find cilk for loops and use more efficient runtime", false, false);

INITIALIZE_PASS_BEGIN(Loop2Cilk, "loop2cilk",
                "Find cilk for loops and use more efficient runtime", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
//INITIALIZE_PASS_DEPENDENCY(PromotePass)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_END(Loop2Cilk, "loop2cilk",
                "Find cilk for loops and use more efficient runtime", false, false)

Pass *llvm::createLoop2CilkPass() {
  return new Loop2Cilk();
}

//===----------------------------------------------------------------------===//
//  Loop2Cilk driver. Manage several subpasses of IV simplification.
//===----------------------------------------------------------------------===//

Value* addOne( Value* V ) {
  if( Constant* C = dyn_cast<Constant>(V) ) {
    ConstantFolder F;
    return F.CreateAdd(C, ConstantInt::get(V->getType(), 1) );
  }

  if( Instruction* I = dyn_cast<Instruction>(V) ) {
    IRBuilder<> builder(I);
    return builder.CreateAdd( V, ConstantInt::get(V->getType(), 1) );
  }
  assert( 0 );
  return nullptr;
}

bool Loop2Cilk::runOnLoop(Loop *L, LPPassManager &) {
  if (skipOptnoneFunction(L))
    return false;

  //errs() << "Loop: \n";
  //L->dump();
  //errs() << "</Loop>\n";
  BasicBlock* Header = L->getHeader();
  assert(Header);
  TerminatorInst* T = Header->getTerminator();
  if( !isa<BranchInst>(T) ) return false;
  BranchInst* B = (BranchInst*)T;
  if( B->getNumSuccessors() != 2 ) return false;
  BasicBlock *detacher = B->getSuccessor(0), *syncer = B->getSuccessor(1);
  if( isa<SyncInst>(detacher->getTerminator()) ){
    BasicBlock* temp = detacher;
    detacher = syncer;
    syncer = temp;
  } else if( !isa<SyncInst>(syncer->getTerminator()) )
    return false;

  DetachInst* det = dyn_cast<DetachInst>(detacher->getTerminator() );
  if( det == nullptr ) return false;
  if( detacher->size() != 1 ) return false;
  if( syncer->size() != 1 ) return false;

  //errs() << "Found candidate for cilk for!\n";

  // If LoopSimplify form is not available, stay out of trouble. Some notes:
  //  - LSR currently only supports LoopSimplify-form loops. Indvars'
  //    canonicalization can be a pessimization without LSR to "clean up"
  //    afterwards.
  //  - We depend on having a preheader; in particular,
  //    Loop::getCanonicalInductionVariable only supports loops with preheaders,
  //    and we're in trouble if we can't find the induction variable even when
  //    we've manually inserted one.
  if (!L->isLoopSimplifyForm())
    return false;

  BasicBlock* body = det->getSuccessor(0);
  PHINode* oldvar = L->getCanonicalInductionVariable();
  if( !oldvar ) return false;

  BasicBlock* done = L->getUniqueExitBlock();
  if( !done ) return false;
  if( done != syncer ) return false;

  //PHINode* var = PHINode::Create( oldvar->getType(), 1, "", &body->front() );
  //ReplaceInstWithInst( var, oldvar );

  auto H = L->getHeader();
  Value* cmp = 0;
  for (BasicBlock::iterator I = H->begin(); I != H->end(); ++I) {
    Instruction* M = &*I;
    if( M == oldvar ) continue;
    if( BranchInst* b = dyn_cast<BranchInst>(M) ) {
      if( b->getNumSuccessors() != 2 ) return false;
      if( b->getSuccessor(0) == detacher ) {
        if( b->getSuccessor(1) != syncer ) return false;
        else continue;
      }
      if( b->getSuccessor(1) == detacher ) {
        if( b->getSuccessor(0) != syncer ) return false;
        else continue;
      }
      return false;
    }
    llvm::CmpInst* is = dyn_cast<CmpInst>(M);
    if( !is ) return false;
    if( !is->isIntPredicate() ) return false;
    auto P = is->getPredicate();
    if( is->getOperand(0) != oldvar ) {
      if( is->getOperand(1) == oldvar )
        P = is->getSwappedPredicate();
      else
        return false;
    }
    if( cmp ) return false;
    cmp = is->getOperand(1);
    //assums non infinite detach loop
    switch( P ) {
      case llvm::CmpInst::ICMP_EQ:
      case llvm::CmpInst::ICMP_NE:
        break;
      case llvm::CmpInst::ICMP_UGT:
      case llvm::CmpInst::ICMP_ULT:
      case llvm::CmpInst::ICMP_SGT:
      case llvm::CmpInst::ICMP_SLT:
        break;
      case llvm::CmpInst::ICMP_UGE:
      case llvm::CmpInst::ICMP_ULE:
      case llvm::CmpInst::ICMP_SGE:
      case llvm::CmpInst::ICMP_SLE:
        cmp = addOne(cmp);
        break;
      default:
        return false;
    }
    //TODO actually check is correct

  }


  llvm::CallInst* call = 0;
  llvm::Value*    closure = 0;
  Function* extracted = llvm::cilk::extractDetachBodyToFunction( *det, &call, /*closure*/ oldvar, &closure );
  if( !extracted ) return false;


	Module* M = extracted->getParent();

  oldvar->removeIncomingValue( 1U );
  oldvar->removeIncomingValue( 0U );

  auto a1 = det->getSuccessor(0);
  while(a1->size() > 0 ){
    Instruction* m = & a1->back();
    m->eraseFromParent();
  }

  auto a2 = det->getSuccessor(1);
  while(a2->size() > 0 ){
    Instruction* m = & a2->back();
    m->eraseFromParent();
  }

  while(H->size() > 0 ){
    Instruction* m = & H->back();
    m->eraseFromParent();
  }

  IRBuilder<> b1(H);
  b1.CreateBr( detacher );
  MergeBlockIntoPredecessor( detacher );

  det->eraseFromParent();
  a1->eraseFromParent();
  a2->eraseFromParent();
  IRBuilder<> b(H);

  llvm::Function* F;
  if( ((llvm::IntegerType*)cmp->getType())->getBitWidth() == 32 )
    F = CILKRTS_FUNC(cilk_for_32, *M);
  else {
    assert( ((llvm::IntegerType*)cmp->getType())->getBitWidth() == 64 );
    F = CILKRTS_FUNC(cilk_for_64, *M);
  }

  llvm::Value* args[] = { b.CreatePointerCast(extracted, F->getFunctionType()->getParamType(0) ), b.CreatePointerCast( closure, F->getFunctionType()->getParamType(1) ), cmp, ConstantInt::get( llvm::Type::getIntNTy( cmp->getContext(), 8*sizeof(int) ), 0 ) };
  b.CreateCall(F, args );

  assert( syncer->size() == 1 );
  b.CreateBr( syncer->getTerminator()->getSuccessor(0) );

  syncer->eraseFromParent();

  LoopInfo &loopInfo = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  loopInfo.updateUnloop(L);

  return true;
}
