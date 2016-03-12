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
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/CilkABI.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

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
//      AU.addRequired<SimplifyCFGPass>();
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

size_t countPredecessors(BasicBlock* syncer) {
  size_t count = 0;
  for (auto it = pred_begin(syncer), et = pred_end(syncer); it != et; ++it) {
    count++;
  }
  return count;
}

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

Value* uncast( Value* V ){
  if( auto* in = dyn_cast<TruncInst>(V) ) {
    return uncast(in->getOperand(0));
  }
  if( auto* in = dyn_cast<SExtInst>(V) ) {
    return uncast(in->getOperand(0));
  }
  if( auto* in = dyn_cast<ZExtInst>(V) ) {
    return uncast(in->getOperand(0));
  }
  return V;
}

size_t getNonPhiSize(BasicBlock* b){
    int bad = 0;
    BasicBlock::iterator i = b->begin();
    while (isa<PHINode>(i) || isa<DbgInfoIntrinsic>(i)) { ++i; bad++; }
    return b->size() - bad;
}
bool Loop2Cilk::runOnLoop(Loop *L, LPPassManager &) {
  if (skipOptnoneFunction(L))
    return false;

  errs() << "<Loop>:\n--------------------------------------------------------------------------------------------------------------------------------";
  L->dump();
  errs() << "</Loop>\n<------------------------------------------------------------------------------------------------>\n";

  if (!L->isLoopSimplifyForm()) {
    errs() << "not simplify form\n";
    simplifyLoop(L, nullptr, nullptr, nullptr, nullptr, false);
    //return false;
  }

  BasicBlock* Header = L->getHeader();
  assert(Header);

  errs() << "<F>:\n******************************************************************************************************************************************";
  Header->getParent()->dump();
  errs() << "</F>:\n******************************************************************************************************************************************";

  TerminatorInst* T = Header->getTerminator();
  if( !isa<BranchInst>(T) ) {
    BasicBlock *Preheader = L->getLoopPreheader();
    if( isa<BranchInst>(Preheader->getTerminator()) ) { T = Preheader->getTerminator(); Header = Preheader; }
    else { errs() << "not branch inst" << "\n";
    T->dump();
    return false;
  }
  }
  BranchInst* B = (BranchInst*)T;
  BasicBlock *detacher, *syncer;
  if( B->getNumSuccessors() != 2 ) {
    BasicBlock* endL = L->getExitBlock();
    while( endL && !isa<SyncInst>( endL->getTerminator() ) ) {
      errs() << "THING: " << endL->size() << " " << isa<BranchInst>(endL->getTerminator()) << " " << (endL->getTerminator()->getNumSuccessors() ) << "\n";
      endL->dump();
      if( getNonPhiSize(endL) == 1 && isa<BranchInst>(endL->getTerminator()) && endL->getTerminator()->getNumSuccessors() == 1 ) {
        //TODO merging
        endL->dump();
        endL->getTerminator()->getSuccessor(0)->dump();
        auto temp = endL->getTerminator()->getSuccessor(0);
        //bool success = TryToSimplifyUncondBranchFromEmptyBlock(endL);
//        bool success = MergeBlockIntoPredecessor(endL->getTerminator()->getSuccessor(0));
        //if( !success ) {
        //  endL = nullptr;
        //  errs() << "no success :(\n";
        //} else {
          endL = temp;
        //}
      }
      else
        endL = nullptr;
    }

    if( endL ) {
      syncer = endL;
      detacher = B->getSuccessor(0);
    } else {
      errs() << "L\n";
      Header->getParent()->dump();
      errs() << "nsucc != 2" << "\n";
      if( endL ) endL->dump();
      else errs() << "no endl" << "\n";
      T->dump();
      return false;
    }


  } else {
    detacher = B->getSuccessor(0);
    syncer = B->getSuccessor(1);


    if( isa<SyncInst>(detacher->getTerminator()) ){
      BasicBlock* temp = detacher;
      detacher = syncer;
      syncer = temp;
    } else if( !isa<SyncInst>(syncer->getTerminator()) ) {
      errs() << "none sync" << "\n";
      syncer->dump();
      detacher->dump();
      return false;
    }

    BasicBlock* done = L->getExitingBlock();
    if( !done ) {
      errs() << "no unique exit block\n";
      return false;
    }
    if( done != syncer ) {
      errs() << "exit != sync\n";
      done->dump();
      syncer->dump();
      syncer->getParent()->dump();
      return false;
    }

  }

  DetachInst* det = dyn_cast<DetachInst>(detacher->getTerminator() );
  if( det == nullptr ) {
    errs() << "other not detach" << "\n";
    detacher->dump();
    return false;
  }
  if( getNonPhiSize(detacher)!=1 ) {
    errs() << "invalid detach size of " << getNonPhiSize(detacher) << "|" << detacher->size() << "\n";
    return false;
  }
  if( getNonPhiSize(syncer)!=1 ) {
    errs() << "invalid sync size" << "\n";
    return false;
  }
  errs() << "Found candidate for cilk for!\n";

  BasicBlock* body = det->getSuccessor(0);
  PHINode* oldvar = L->getCanonicalInductionVariable();
  if( !oldvar ) {
      errs() << "no induction var\n";
      return false;
  }
  //PHINode* var = PHINode::Create( oldvar->getType(), 1, "", &body->front() );
  //ReplaceInstWithInst( var, oldvar );
  Value* adder = 0;
  for( unsigned i=0; i<oldvar->getNumIncomingValues(); i++){
    if( oldvar->getIncomingBlock(i) == Header ){
      if( ConstantInt* ci = dyn_cast<ConstantInt>(oldvar->getIncomingValue(i))) {
        if( !ci->isZero() ) {
          errs() << "nonzero start";
          return false;
        }
      } else {
        errs() << "non-constant start\n";
        return false;
      }
    } else {
      if( BinaryOperator* bo = dyn_cast<BinaryOperator>(oldvar->getIncomingValue(i))) {
        if( bo->getOpcode() != Instruction::Add ) {
          errs() << "non-adding phi node";
          return false;
        }
        if( oldvar != bo->getOperand(0) ) bo->swapOperands();
        if( oldvar != bo->getOperand(0) ) {
          errs() << "old indvar not part of loop inc?\n";
        }
        if( ConstantInt* ci = dyn_cast<ConstantInt>(bo->getOperand(1))) {
          if( !ci->isOne() ) {
            errs() << "non one inc";
            return false;
          }
        } else {
          errs() << "non-constant inc\n";
          oldvar->getIncomingValue(i)->dump();
          return false;
        }
        adder = bo;
      } else {
        errs() << "non-constant start\n";
        return false;
      }
    }
  }

  if( adder == 0 ) {
    errs() << "couldn't check for increment\n";
    return false;
  }
  Value* cmp = 0;

  bool simplified = false;
  while( !simplified ){
    simplified = true;
    for (auto it = pred_begin(syncer), et = pred_end(syncer); it != et; ++it) {
      BasicBlock* endL = *it;
      if( getNonPhiSize(endL) == 1 && isa<BranchInst>(endL->getTerminator()) && endL->getTerminator()->getNumSuccessors() == 1 ) {
        bool success = TryToSimplifyUncondBranchFromEmptyBlock(endL);
        if(success) {
          simplified = false;
          break;
        }
      }
    }
  }

  for (auto it = pred_begin(syncer), et = pred_end(syncer); it != et; ++it) {
    BasicBlock* pred = *it;
    if( pred == Header ) break;
    if( cmp != 0 ){
      errs() << "comparisoin already set\n";
      return false;
    }
    BranchInst* b = dyn_cast<BranchInst>(pred->getTerminator());
    if( b == nullptr ) {
      errs() << "loop term not branch\n";
      return false;
    }

    if( b->getNumSuccessors() != 2 ) {
      errs() << "branch != 2 succ \n";
      return false;
    }
    if( !(b->getSuccessor(0) == detacher && b->getSuccessor(1) == syncer || b->getSuccessor(1) == detacher && b->getSuccessor(0) == syncer) ) {
      errs() << "invalid branching\n";
      return false;
    }
    llvm::CmpInst* is = dyn_cast<CmpInst>(b->getCondition());
    if( !is ) {
      errs() << "condition was not in block\n";
      return false;
    }
    if( !is->isIntPredicate() ) {
      errs() << "non-integral condition\n";
      return false;
    }
    auto P = is->getPredicate();

    {
      if( uncast(is->getOperand(0)) != adder ) {
        if( uncast(is->getOperand(1)) == adder )
          P = is->getSwappedPredicate();
        else {
          errs() << "none are \n";
          is->dump();
          goto oldvarB;
        }
      }

      cmp = is->getOperand(1);
      //assums non infinite detach loop
      //TODO check!
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
          errs() << "weird opcode2\n";
          return false;
      }
      goto endT;
    }

    oldvarB:
    if( uncast(is->getOperand(0)) != oldvar ) {
      if( uncast(is->getOperand(1)) == oldvar )
        P = is->getSwappedPredicate();
      else {
        errs() << "none are \n";
        is->dump();
        return false;
      }
    }
    cmp = is->getOperand(1);
    //assums non infinite detach loop
    //TODO check!
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
        errs() << "weird opcode\n";
        return false;
    }

  }

  endT:
  llvm::CallInst* call = 0;
  llvm::Value*    closure = 0;

  Function* extracted = llvm::cilk::extractDetachBodyToFunction( *det, &call, /*closure*/ oldvar, &closure );

  if( !extracted ) {
    errs() << "not extracted\n";
    return false;
  }
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

  det->eraseFromParent();
  a1->eraseFromParent();
  a2->eraseFromParent();

  Header->getTerminator()->eraseFromParent();
  IRBuilder<> b2(Header);
  b2.CreateBr( detacher );
  IRBuilder<> b(detacher);

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

  errs() << "<M>:\n*##########################################3*****************************************************************************************************************************************";
  Header->getParent()->dump();
  errs() << "</M>:\n*############################################################33*****************************************************************************************************************************************";
  M->dump();

  syncer->replaceAllUsesWith( syncer->getTerminator()->getSuccessor(0) );

  //TODO assumes no other detaches were sync'd by this
  syncer->eraseFromParent();
  Header->getParent()->dump();

  LoopInfo &loopInfo = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  loopInfo.updateUnloop(L);
  errs() << "TRANSFORMED LOOP!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
  return true;
}
