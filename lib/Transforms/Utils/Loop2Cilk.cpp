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

size_t countPHI(BasicBlock* b){
    int phi = 0;
    BasicBlock::iterator i = b->begin();
    while (isa<PHINode>(i) ) { ++i; phi++; }
    return phi;
}

int64_t getInt(Value* v, bool & failed){
  if( ConstantInt* CI = dyn_cast<ConstantInt>(v) ) {
    failed = false;
    return CI->getSExtValue();
  }
  failed = true;
  return -1;
}

bool isOne(Value* v){
  bool m = false;
  return getInt(v, m) == 1;
}

bool isZero(Value* v){
  bool m = false;
  return getInt(v, m) == 0;
}

PHINode* getIndVar(Loop *L, BasicBlock* detacher) {
  BasicBlock *H = L->getHeader();

  ////H->getParent()->dump();

  BasicBlock *Incoming = nullptr, *Backedge = nullptr;
  pred_iterator PI = pred_begin(H);
  assert(PI != pred_end(H) && "Loop must have at least one backedge!");
  Backedge = *PI++;
  if (PI == pred_end(H)) return nullptr;  // dead loop
  Incoming = *PI++;
  if (PI != pred_end(H)) return nullptr;  // multiple backedges?
  if (L->contains(Incoming)) {
    if (L->contains(Backedge)) return nullptr;
    std::swap(Incoming, Backedge);
  } else if (!L->contains(Backedge)) return nullptr;

  llvm::CmpInst* cmp = 0;
  int cmpIdx = -1;
  llvm::Value* opc = 0;

  BasicBlock* cmpNode = Backedge;
  if(H!=detacher) {
    cmpNode = detacher->getUniquePredecessor();
    if(cmpNode==nullptr) return nullptr;
  }

  if( auto brnch = dyn_cast<BranchInst>(cmpNode->getTerminator()) ) {
    if(!brnch->isConditional()) goto cmp_error;
    if( cmp = dyn_cast<CmpInst>(brnch->getCondition()) ) {
    } else {
      errs() << "no comparison inst from backedge\n";
      cmpNode->getTerminator()->dump();
      return nullptr;
    }
  } else {
    cmp_error:
    errs() << "<no comparison from backedge>\n";
    cmpNode->getTerminator()->dump();
    cmpNode->getParent()->dump();
    errs() << "</no comparison from backedge>\n";
    return nullptr;
  }

  // Loop over all of the PHI nodes, looking for a canonical indvar.
  PHINode* RPN = nullptr;
  Instruction* INCR = nullptr;
  Value* amt = nullptr;
  std::vector<std::tuple<PHINode*,Instruction*,Value*>> others;
  for (BasicBlock::iterator I = H->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    if( !PN->getType()->isIntegerTy() ) {
      errs() << "phinode uses non-int\n";
      return nullptr;
    }
    if(auto Inc = dyn_cast<Instruction>(PN->getIncomingValueForBlock(Backedge))) {
      if (Inc->getOpcode() == Instruction::Add && ( Inc->getOperand(0) == PN || Inc->getOperand(1) == PN ) ) {
        if( Inc->getOperand(1) == PN ) ((BinaryOperator*)Inc)->swapOperands();
        assert( Inc->getOperand(0) == PN );
        bool rpnr = false;
        bool incr = false;
        for(int i = 0; i<cmp->getNumOperands(); i++) {
          rpnr |= uncast(cmp->getOperand(i)) == PN;
          incr |= uncast(cmp->getOperand(i)) == Inc;
          if( rpnr | incr ) cmpIdx = i;
        }
        assert( !rpnr || !incr );
        if( rpnr | incr ) {
          amt = Inc->getOperand(1);
          RPN = PN;
          INCR = Inc;
          opc = rpnr?RPN:INCR;
        } else {
          others.push_back( std::make_tuple(PN,Inc,Inc->getOperand(1)) );
        }
      } else {
        errs() << "no add found for:\n"; PN->dump(); Inc->dump();
        H->getParent()->dump();
        return nullptr;
      }
    } else {
      errs() << "no inc found for:\n"; PN->dump();
    }
  }

  if( RPN == 0 ) {
    errs() << "<no RPN>\n";
    cmp->dump();
    errs() << "<---->\n";
    H->dump();
    errs() << "<---->\n";
    for( auto a : others ) { std::get<0>(a)->dump(); }
    errs() << "</no RPN>\n";
    return nullptr;
  }

  llvm::Value* mul;
  llvm::Value* newV;
  {
    IRBuilder<> builder(detacher->getTerminator()->getSuccessor(0)->getFirstNonPHIOrDbgOrLifetime());
    if( isOne(amt) ) mul = RPN;
    else mul = builder.CreateMul(RPN, amt);
    if( isZero(RPN->getIncomingValueForBlock(Incoming) )) newV = mul;
    else newV = builder.CreateAdd(mul, RPN->getIncomingValueForBlock(Incoming) );

    //  std::vector<Value*> replacements;
    for( auto a : others ) {
      llvm::Value* val = builder.CreateSExtOrTrunc(RPN,std::get<0>(a)->getType());
      llvm::Value* amt0 = std::get<2>(a);
      if( !isOne(amt0) ) val = builder.CreateMul(val,amt0);
      llvm::Value* add0 = std::get<0>(a)->getIncomingValueForBlock(Incoming);
      if( !isZero(add0) ) val = builder.CreateAdd(val,add0);
      std::get<0>(a)->dump();
      std::get<0>(a)->replaceAllUsesWith(val);
      std::get<0>(a)->eraseFromParent();
      if(std::get<1>(a)->getNumUses() == 0) std::get<1>(a)->eraseFromParent();
      //replacements.push_back(val);
    }

    ////errs() << "RPN  :\n"; RPN->dump();
    ////errs() << "MUL  :\n"; mul->dump();
    ////errs() << "NEWV :\n"; newV->dump();
    ////errs() << "NEWVP:\n"; ((Instruction*)newV)->getParent()->dump();
  }


  std::vector<Use*> uses;
  for( auto& U : RPN->uses() ) uses.push_back(&U);
  for( auto Up : uses ) {
    auto&U = *Up;
    Instruction *I = cast<Instruction>(U.getUser());
    if( I == INCR ) INCR->setOperand(1, ConstantInt::get( RPN->getType(), 1 ) );
    else if( I == mul && mul != RPN ) continue;
    else if( I == newV && newV != RPN ) continue;
    else if( I == cmp ) continue;
    else {
      U.set( newV );
    }
  }

  ////errs() << "CMP: (idx=" << cmpIdx << ")\n"; cmp->dump();
  IRBuilder<> build(cmp);
  llvm::Value* val = cmp->getOperand(cmpIdx);
  llvm::Value* adder = RPN->getIncomingValueForBlock(Incoming);
  llvm::Value* amt0  = amt;

  int cast_type = 0;
  if( isa<TruncInst>(val) ) cast_type = 1;
  if( isa<SExtInst>(val) ) cast_type = 2;
  if( isa<ZExtInst>(val) ) cast_type = 3;

  if( !isZero(adder) ) {
    switch(cast_type){
      default:;
      case 1: adder = build.CreateTrunc(adder,val->getType());
      case 2: adder = build.CreateSExt( adder,val->getType());
      case 3: adder = build.CreateZExt( adder,val->getType());
    }
    val = build.CreateSub(val, adder);
  }
  if( !isOne(amt0) ) {
    switch(cast_type){
      default:;
      case 1: amt0 = build.CreateTrunc(amt0,val->getType());
      case 2: amt0 = build.CreateSExt( amt0,val->getType());
      case 3: amt0 = build.CreateZExt( amt0,val->getType());
    }
    val = build.CreateSDiv(val, amt0);
  }
  cmp->setOperand(cmpIdx, val);

  RPN->setIncomingValue( RPN->getBasicBlockIndex(Incoming),  ConstantInt::get( RPN->getType(), 0 ) );


  //RPN->getParent()->getParent()->dump();
  ////RPN->dump();
  //mul->dump();
  //newV->dump();
  //INCR->dump();
  ////((Instruction*)newV)->getParent()->dump();
//   exit(1);

  return RPN;
}

void removeFromAll(Loop* L, BasicBlock* B){
  if( !L ) return;
  if( L->contains(B) ) L->removeBlockFromLoop(B);
  removeFromAll(L->getParentLoop(), B);
}

BasicBlock* getTrueExit(Loop *L){

    SmallVector< BasicBlock *, 32> exitBlocks;
    L->getExitBlocks(exitBlocks);
    BasicBlock* endL = 0;
    SmallPtrSet<BasicBlock *, 32> exits(exitBlocks.begin(), exitBlocks.end());
    SmallPtrSet<BasicBlock *, 32> alsoLoop;

    exitRemoval:
    if( exits.size() >= 2 ) {
      for( auto tempExit : exits ) {
        SmallPtrSet<BasicBlock *, 32> reachable;
        std::vector<BasicBlock*> Q = { tempExit };
        bool valid = true;
        while(!Q.empty() && valid) {
          auto m = Q.back();
          Q.pop_back();
          if( isa<UnreachableInst>(m->getTerminator()) ) { reachable.insert(m); continue; }
          else if( auto b = dyn_cast<BranchInst>(m->getTerminator()) ) {
            bool bad = false;
            reachable.insert(m);
            for( unsigned i=0; i<b->getNumSuccessors(); i++ ) {
               auto suc = b->getSuccessor(i);
               if( L->contains(suc) || std::find(exitBlocks.begin(), exitBlocks.end(), suc) != exitBlocks.end() || std::find(alsoLoop.begin(), alsoLoop.end(), suc) != alsoLoop.end() || std::find(reachable.begin(), reachable.end(), suc) != reachable.end() ) {

               } else{
                Q.push_back(suc);
                bad =  true;
                break;
              }
            }
          }
          else valid = false;
        }
        if( valid && reachable.size() > 0 ) {
          for( auto b : reachable){
            exits.erase(b);
            alsoLoop.insert(b);
          }
          goto exitRemoval;
        }
      }
    }

    //errs() << "<blocks>\n";
    //for(auto a : exits ) a->dump();
    //errs() << "</blocks>\n";
    if( exits.size() == 1 ) endL = * exits.begin();
    return endL;
}

bool Loop2Cilk::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipOptnoneFunction(L))
    return false;

  errs() << "<Loop>:\n--------------------------------------------------------------------------------------------------------------------------------";
  //for(auto a: L->blocks()){
  //  a->dump();
  //}
  L->dump();
  errs() << "</Loop>\n<------------------------------------------------------------------------------------------------>\n";

  if (!L->isLoopSimplifyForm()) {
    //errs() << "not simplify form\n";
    simplifyLoop(L, nullptr, nullptr, nullptr, nullptr, false);
    //return false;
  }

  BasicBlock* Header = L->getHeader();
  assert(Header);

  auto parentL = L->getParentLoop();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  //errs() << "<F>:\n******************************************************************************************************************************************";
  //Header->getParent()->dump();
  //errs() << "</F>:\n******************************************************************************************************************************************";

  TerminatorInst* T = Header->getTerminator();
  if( !isa<BranchInst>(T) ) {
    BasicBlock *Preheader = L->getLoopPreheader();
    if( isa<BranchInst>(Preheader->getTerminator()) ) { T = Preheader->getTerminator(); Header = Preheader; }
    else {
      errs() << "not branch inst" << "\n";
      T->dump();
    return false;
  }
  }
  BranchInst* B = (BranchInst*)T;
  BasicBlock *detacher, *syncer;
  if( B->getNumSuccessors() != 2 ) {
    BasicBlock* endL = getTrueExit(L);
    auto oendL = endL;
    while( endL && !isa<SyncInst>( endL->getTerminator() ) ) {
      //errs() << "THING: " << endL->size() << " " << isa<BranchInst>(endL->getTerminator()) << " " << (endL->getTerminator()->getNumSuccessors() ) << "\n";
      //endL->dump();
      if( getNonPhiSize(endL) == 1 && isa<BranchInst>(endL->getTerminator()) && endL->getTerminator()->getNumSuccessors() == 1 ) {
        //TODO merging
        //endL->dump();
        //endL->getTerminator()->getSuccessor(0)->dump();
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
      //errs() << "L\n";
      //Header->getParent()->dump();
      errs() << "nsucc != 2" << "\n";
      if( endL ) endL->dump();
      else errs() << "no endl" << "\n";
      if( oendL ) oendL->dump();
      T->dump();
      //T->getParent()->getParent()->dump();
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
      //syncer->dump();
      //detacher->dump();
      return false;
    }

    BasicBlock* done = getTrueExit(L);
    if( !done ) {
      errs() << "no unique exit block\n";
      return false;
    }
    if( auto BI = dyn_cast<BranchInst>(done->getTerminator()) ) {
      if( BI->getNumSuccessors() == 2 ) {
        if( BI->getSuccessor(0) == detacher && BI->getSuccessor(1) == syncer )
          done = syncer;
        if( BI->getSuccessor(1) == detacher && BI->getSuccessor(0) == syncer )
          done = syncer;
      }
    }
    if( getUniquePred(done) == syncer ){
      //errs() << "has unique pred\n";
      auto term = done->getTerminator();
      bool good = true;
      for(unsigned i=0; i<term->getNumSuccessors(); i++)
        if( L->contains( term->getSuccessor(i)) ){
          //errs() << "loop contains succ " << term->getSuccessor(i)->getName() << "\n";
          good = false;
          break;
        }
      if( good ) done = syncer;
    }
    if( done != syncer ) {
      errs() << "exit != sync\n";
      //done->dump();
      //syncer->dump();
      //syncer->getParent()->dump();
      return false;
    }

  }

  DetachInst* det = dyn_cast<DetachInst>(detacher->getTerminator() );
  if( det == nullptr ) {
    errs() << "other not detach" << "\n";
    detacher->dump();
    return false;
  }
  nps_begin:
  if( getNonPhiSize(detacher)!=1 ) {
    Instruction* badInst = getLastNonTerm(detacher);
    errs() << "badInst:\n"; badInst->dump();
    if( !badInst->mayWriteToMemory() ) {
      DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      DT.recalculate(*L->getHeader()->getParent());

      for (const Use &U : badInst->uses()) {
        const Instruction *I = cast<Instruction>(U.getUser());
        auto BB = I->getParent();
        if( !DT.dominates(BasicBlockEdge(detacher, det->getSuccessor(0) ), U) ) { errs() << "use not dominated:\n"; U->dump(); goto nps_error; }
      }
      badInst->moveBefore( getFirstPostPHI(det->getSuccessor(0)) );
      goto nps_begin;
    } else errs() << "mayWrite:\n"; 
    nps_error:
    errs() << "invalid detach size of " << getNonPhiSize(detacher) << "|" << detacher->size() << "\n";
    return false;
  }
  while( getNonPhiSize(syncer)!=1 ) {
    Instruction* badInst = getLastNonTerm(syncer);
    errs() << "badInst2:\n"; badInst->dump();
    badInst->moveBefore( getFirstPostPHI(syncer->getTerminator()->getSuccessor(0)) );
//    errs() << "invalid sync size" << "\n";
//    return false;
  }
  errs() << "Found candidate for cilk for!\n";

  BasicBlock* body = det->getSuccessor(0);
  PHINode* oldvar = oldvar = getIndVar( L, detacher);//L->getCanonicalInductionVariable();
  if( !oldvar ) {
      errs() << "no induction var\n";
      return false;
  }
  auto tmpH = L->getHeader();
  if( tmpH->size() != getNonPhiSize(tmpH) + 1 ) {
    //Instruction* badPHI = nullptr;
    //for (Instruction &I : *Header)
    //  if (&I != oldvar){ badPHI = &I; break; }
    //if( badPHI->getType() != oldvar->getType() ) {
    //  PHINode* pn = (PHINode*)badPHI;
    //  llvm::Value* = 
    //}
    errs() << "Can only cilk_for loops with only 1 phi node " << tmpH->size() << "|" << getNonPhiSize(tmpH) << "\n";
    tmpH->dump();
    return false;
  }

  //PHINode* var = PHINode::Create( oldvar->getType(), 1, "", &body->front() );
  //ReplaceInstWithInst( var, oldvar );
  Value* adder = 0;
  for( unsigned i=0; i<oldvar->getNumIncomingValues(); i++){
    if( !L->contains(oldvar->getIncomingBlock(i) ) || oldvar->getIncomingBlock(i) == Header ){
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
          //oldvar->getIncomingValue(i)->dump();
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
          removeFromAll(parentL, endL);
          LI.changeLoopFor(endL, nullptr);
          LI.removeBlock(endL);
          simplified = false;
          break;
        }
      }
    }
  }

  for (auto it = pred_begin(syncer), et = pred_end(syncer); it != et; ++it) {
    BasicBlock* pred = *it;
    //errs() << "checking " << pred->getName() << " for cmp\n";
    if( !L->contains(pred) ) continue;

    if( cmp != 0 ){
      errs() << "comp already set\n";
      errs() << "prev cmp:\n"; cmp->dump();
      errs() << "new cmp:\n"; pred->dump();
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
          errs() << "none are 0\n";
          //is->dump();
          goto oldvarB;
        }
      }

      cmp = is->getOperand(1);
      is->setOperand(0, ConstantInt::get(cmp->getType(), 0));
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
        errs() << "none are 1\n";
        is->dump();
        return false;
      }
    }
    cmp = is->getOperand(1);
    is->setOperand(0, ConstantInt::get(cmp->getType(), 0));
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
  if( cmp == 0 ) {
    errs() << "cannot find cmp\n";
    return false;
  }
  llvm::CallInst* call = 0;
  llvm::Value*    closure = 0;

  std::vector<Value*> toMove;
  toMove.push_back(cmp);
  Instruction* pi = detacher->getTerminator();
  {
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  DT.recalculate(*L->getHeader()->getParent());

  while( !toMove.empty() ) {
    auto b = toMove.back();
    toMove.pop_back();
    if( Instruction* inst = dyn_cast<Instruction>(b) ) {
      if( !DT.dominates(inst, detacher) ) {
        errs() << "moving: ";
        b->dump();
        for (User::op_iterator i = inst->op_begin(), e = inst->op_end(); i != e; ++i) {
          Value *v = *i;
          toMove.push_back(v);
        }
        if( inst->mayHaveSideEffects() ) {
          errs() << "something side fx\n";
          return false;
        }
        inst->moveBefore(pi);
        pi = inst;
      }
    }
  }
  }
  //errs() << "<cmp>\n";
  //cmp->dump();
  //Header->getParent()->dump();
  //detacher->dump();
  //errs() << "</cmp>\n";

  Function* extracted = llvm::cilk::extractDetachBodyToFunction( *det, &call, /*closure*/ oldvar, &closure );

  if( !extracted ) {
    errs() << "not extracted\n";
    return false;
  }

  {
    for( BasicBlock& b : extracted->getBasicBlockList() )
      if( true ) {
        removeFromAll(parentL, &b);
        LI.changeLoopFor(&b, nullptr);
        LI.removeBlock(&b);
      }
  }

  //extracted->dump();

  Module* M = extracted->getParent();
  auto a1 = det->getSuccessor(0);
  auto a2 = det->getSuccessor(1);

  oldvar->removeIncomingValue( 1U );
  oldvar->removeIncomingValue( 0U );

  assert( det->use_empty() );
  det->eraseFromParent();
  if( countPredecessors(a2) == 0 ){
    auto tmp = a1;
    a1 = a2;
    a2 = tmp;
  }

  if( parentL ) parentL->removeChildLoop( std::find(parentL->getSubLoops().begin(), parentL->getSubLoops().end(), L) );
  LI.removeBlock(a1);

  removeFromAll(parentL, a1);
  DeleteDeadBlock(a1);
  if( a1 != a2 ) {
    LI.removeBlock(a2);
    removeFromAll(parentL, a2);
    DeleteDeadBlock(a2);
  }

  assert( Header->getTerminator()->use_empty() );
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
  b.CreateBr( syncer );

  //L->invalidate();
  //errs() << "<M>:\n*##########################################3*****************************************************************************************************************************************";
  //Header->getParent()->dump();
  //errs() << "</M>:\n*############################################################33*****************************************************************************************************************************************";
  //M->dump();
  auto term = syncer->getTerminator()->getSuccessor(0);
  syncer->getTerminator()->eraseFromParent();
  IRBuilder<> sbuild(syncer);
  sbuild.CreateBr( term );

  //Header->getParent()->dump();

  errs() << "TRANSFORMED LOOP!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
  //M->dump();

  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  SE.forgetLoop(L);

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  DT.recalculate(*Header->getParent());
  std::vector<BasicBlock*> blocks;
  //for(auto& b : L->blocks()) blocks.push_back(b);
  //for(auto& b : blocks) {
  //  if( extracted != b->getParent() ) continue;
  //  LI.changeLoopFor(b, nullptr);
  //  LI.removeBlock(b);
  //}

  //for(auto& b : blocks) {
  //  if( parentL && parentL->contains(b) ) parentL->removeBlockFromLoop(b);
  //  if( L->contains(b) ) L->removeBlockFromLoop(b);
  //}
  L->invalidate();
  //LI.markAsRemoved(L);

  //L->verifyLoop();
  //Header->getParent()->dump();
  if( parentL ) parentL->verifyLoop();
  //LI.verify();

  //LPM.verifyAnalysis();

  return true;
}
