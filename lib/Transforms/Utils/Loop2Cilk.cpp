//===- Loop2Cilk.cpp - Convert Loops of Detaches to use the Cilk Runtime ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Convert Loops of Detaches to use the Cilk Runtime
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
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include <utility>
using std::make_pair;

using namespace llvm;

#define DEBUG_TYPE "loop2cilk"

namespace {
  class Loop2Cilk : public LoopPass {

  public:

    static char ID; // Pass identification, replacement for typeid
    Loop2Cilk() : LoopPass(ID) { }

    bool runOnLoop(Loop *L, LPPassManager &LPM) override;
    bool performDAC(Loop *L, LPPassManager &LPM);

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      //AU.addRequired<ScalarEvolutionWrapperPass>();
      AU.addRequiredID(LoopSimplifyID);
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
//INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(Loop2Cilk, "loop2cilk", "Find cilk for loops and use more efficient runtime", false, false)

Pass *llvm::createLoop2CilkPass() {
  return new Loop2Cilk();
}

Value* neg(Value* V) {
  if( Constant* C = dyn_cast<Constant>(V) ) {
    ConstantFolder F;
    return F.CreateNeg(C);
  }

  Instruction* I = nullptr;
  bool move = false;
  if( Argument* A = dyn_cast<Argument>(V) ) {
    I = A->getParent()->getEntryBlock().getFirstNonPHIOrDbgOrLifetime();
  } else if( PHINode* A = dyn_cast<PHINode>(V) ) {
    I = A->getParent()->getFirstNonPHIOrDbgOrLifetime();
  } else {
    assert( isa<Instruction>(V) );
    I = cast<Instruction>(V);
    move = true;
  }
  assert(I);
  IRBuilder<> builder(I);
  Instruction* foo = cast<Instruction>(builder.CreateNeg(V));
  if (move) I->moveBefore(foo);
  return foo;
}

Value* subOne(Value* V, std::string s="") {
  if( Constant* C = dyn_cast<Constant>(V) ) {
    ConstantFolder F;
    return F.CreateSub(C, ConstantInt::get(V->getType(), 1) );
  }
  Instruction* I = nullptr;
  bool move = false;
  if( Argument* A = dyn_cast<Argument>(V) ) {
    I = A->getParent()->getEntryBlock().getFirstNonPHIOrDbgOrLifetime();
  } else if( PHINode* A = dyn_cast<PHINode>(V) ) {
    I = A->getParent()->getFirstNonPHIOrDbgOrLifetime();
  } else {
    assert( isa<Instruction>(V) );
    I = cast<Instruction>(V);
    move = true;
  }
  assert(I);
  IRBuilder<> builder(I);
  Instruction* foo = cast<Instruction>(builder.CreateSub(V, ConstantInt::get(V->getType(), 1), s));
  if (move) I->moveBefore(foo);
  return foo;
}

Value* addOne(Value* V, std::string n="") {
  if (Constant* C = dyn_cast<Constant>(V)) {
    ConstantFolder F;
    return F.CreateAdd(C, ConstantInt::get(V->getType(), 1) );
  }

  Instruction* I = nullptr;
  bool move = false;
  if( Argument* A = dyn_cast<Argument>(V) ) {
    I = A->getParent()->getEntryBlock().getFirstNonPHIOrDbgOrLifetime();
  } else if( PHINode* A = dyn_cast<PHINode>(V) ) {
    I = A->getParent()->getFirstNonPHIOrDbgOrLifetime();
  } else {
    assert( isa<Instruction>(V) );
    I = cast<Instruction>(V);
    move = true;
  }
  assert(I);
  IRBuilder<> builder(I);
  Instruction* foo = cast<Instruction>(builder.CreateAdd(V, ConstantInt::get(V->getType(), 1), n));
  if (move) I->moveBefore(foo);
  return foo;
}

Value* uncast(Value* V) {
  if (TruncInst* in = dyn_cast<TruncInst>(V)) {
    return uncast(in->getOperand(0));
  }
  if (SExtInst* in = dyn_cast<SExtInst>(V)) {
    return uncast(in->getOperand(0));
  }
  if (ZExtInst* in = dyn_cast<ZExtInst>(V)) {
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
  if (ConstantInt* CI = dyn_cast<ConstantInt>(v)) {
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

bool attemptRecursiveMoveHelper(Instruction* toMoveAfter, Instruction* toCheck, DominatorTree& DT, std::vector<Instruction*>& candidates) {
  if (DT.dominates(toMoveAfter, toCheck)) return true;

  if (toCheck->mayHaveSideEffects()) {
    llvm::errs() << "invalid move\n"; toCheck->dump();
    return false;
  }

  for (auto & u2 : toCheck->uses() ) {
    if (!DT.dominates(toMoveAfter, u2) ) {
      assert( isa<Instruction>(u2.getUser()) );
      if (!attemptRecursiveMoveHelper(toMoveAfter, cast<Instruction>(u2.getUser()), DT, candidates)) return false;
    }
  }

  candidates.push_back(toCheck);
  return true;
}

bool attemptRecursiveMoveAfter(Instruction* toMoveAfter, Instruction* toCheck, DominatorTree& DT) {
  std::vector<Instruction*> candidates;
  bool b = attemptRecursiveMoveHelper(toMoveAfter, toCheck, DT, candidates);
  if (!b) return false;

  auto last = toMoveAfter;
  for (Instruction* cand : candidates) {
    cand->moveBefore(last);
    last = cand;
  }
  if (last != toMoveAfter) toMoveAfter->moveBefore(last);

  if (!DT.dominates(toMoveAfter, toCheck)) {
    toMoveAfter->dump();
    toCheck->dump();
    toMoveAfter->getParent()->getParent()->dump();
    return false;
  }
  assert(DT.dominates(toMoveAfter, toCheck));
  return true;
}

bool recursiveMoveBefore(Instruction* toMoveBefore, Value* toMoveVal, DominatorTree& DT, std::string nm) {
  Instruction* toMoveI = dyn_cast<Instruction>(toMoveVal);
  if (!toMoveI) return true;

  std::vector<Value*> toMove;
  toMove.push_back(toMoveI);
  Instruction* pi = toMoveBefore;

  while (!toMove.empty()) {
    auto b = toMove.back();
    toMove.pop_back();
    if( Instruction* inst = dyn_cast<Instruction>(b) ) {
      if( !DT.dominates(inst, toMoveBefore) ) {
        for (User::op_iterator i = inst->op_begin(), e = inst->op_end(); i != e; ++i) {
          Value *v = *i;
          toMove.push_back(v);
          //if (isa<Instruction>(v)) { llvm::errs()<<"for "; inst->dump(); v->dump(); }
        }
        if (inst->mayHaveSideEffects()) {
          errs() << "something side fx\n";
          return false;
        }
        if (isa<PHINode>(inst)) {
          errs() << "some weird phi stuff trying to move, move before:" << nm << "\n";
          inst->dump();
          toMoveBefore->dump();
          return false;
        }
        inst->moveBefore(pi);
        pi = inst;
      }
    }
  }
  return true;
}

/* Returns ind var / number of iterations */
std::pair<PHINode*,Value*> getIndVar(Loop *L, BasicBlock* detacher, DominatorTree& DT, bool actualFix=true) {
  BasicBlock *H = L->getHeader();

  BasicBlock *Incoming = nullptr, *Backedge = nullptr;
  pred_iterator PI = pred_begin(H);
  assert(PI != pred_end(H) && "Loop must have at least one backedge!");
  Backedge = *PI++;
  if (PI == pred_end(H)) return make_pair(nullptr,nullptr);  // dead loop
  Incoming = *PI++;
  if (PI != pred_end(H)) return make_pair(nullptr,nullptr);  // multiple backedges?

  if (L->contains(Incoming)) {
    if (L->contains(Backedge)) return make_pair(nullptr,nullptr);
    std::swap(Incoming, Backedge);
  } else if (!L->contains(Backedge)) return make_pair(nullptr,nullptr);

  assert( L->contains(Backedge) );
  assert( !L->contains(Incoming) );
  llvm::CmpInst* cmp = 0;
  int cmpIdx = -1;
  llvm::Value* opc = 0;

  BasicBlock* cmpNode = Backedge;
  if (H != detacher) {
    cmpNode = detacher->getUniquePredecessor();
    if(cmpNode==nullptr) return make_pair(nullptr,nullptr);
  }

  if (BranchInst* brnch = dyn_cast<BranchInst>(cmpNode->getTerminator()) ) {
    if (!brnch->isConditional()) goto cmp_error;
    cmp = dyn_cast<CmpInst>(brnch->getCondition());
    if (cmp == nullptr) {
      errs() << "no comparison inst from backedge\n";
      cmpNode->getTerminator()->dump();
      return make_pair(nullptr,nullptr);
    }
    if (!L->contains(brnch->getSuccessor(0))) {
      cmp->setPredicate(CmpInst::getInversePredicate(cmp->getPredicate()));
      brnch->swapSuccessors();
    }
    if (!cmp->isIntPredicate() || cmp->getPredicate() == CmpInst::ICMP_EQ ) {
      cmpNode->getParent()->dump();
      cmpNode->dump();
      cmp->dump();
      brnch->dump();
      return make_pair(nullptr,nullptr);
    }
  } else {
    cmp_error:
    errs() << "<no comparison from backedge>\n";
    cmpNode->getTerminator()->dump();
    cmpNode->getParent()->dump();
    errs() << "</no comparison from backedge>\n";
    return make_pair(nullptr,nullptr);
  }

  for (unsigned i=0; i<2; i++) {
    LoadInst* inst = dyn_cast<LoadInst>(uncast(cmp->getOperand(i)));
    if (!inst) continue;
    AllocaInst* alloca = dyn_cast<AllocaInst>(inst->getOperand(0));
    if (!alloca) continue;
    if (isAllocaPromotable(alloca, DT)) {
      PromoteMemToReg({alloca}, DT, nullptr, nullptr);
    }
  }
  if (!actualFix) return make_pair(nullptr,nullptr);

  // Loop over all of the PHI nodes, looking for a canonical indvar.
  PHINode* RPN = nullptr;
  Instruction* INCR = nullptr;
  Value* amt = nullptr;
  std::vector<std::tuple<PHINode*,Instruction*,Value*>> others;
  for (BasicBlock::iterator I = H->begin(); isa<PHINode>(I); ) {
    assert( isa<PHINode>(I) );
    PHINode *PN = cast<PHINode>(I);
    if (LoadInst* ld = dyn_cast<LoadInst>(uncast(PN->getIncomingValueForBlock(Incoming)))) {
      if (LoadInst* ld2 = dyn_cast<LoadInst>(uncast(PN->getIncomingValueForBlock(Backedge)))) {
        LoadInst *t1 = ld, *t2 = ld2;
        bool valid = false;
        while (t1 && t2) {
          if(t1->getPointerOperand() == t1->getPointerOperand()) { valid = true; break; }
          uncast(t1->getPointerOperand())->dump();
          uncast(t2->getPointerOperand())->dump();

          /// TODO GEP inst
          ///if (LoadInst* ld = dyn_cast<LoadInst>(uncast(PN->getIncomingValueForBlock(Incoming)))) {
          ///  if (LoadInst* ld2 = dyn_cast<LoadInst>(uncast(PN->getIncomingValueForBlock(Backedge)))) {


          t1 = dyn_cast<LoadInst>(uncast(t1->getPointerOperand()));
          t2 = dyn_cast<LoadInst>(uncast(t2->getPointerOperand()));
        }
        if (valid) {
          ++I;
          ld2->replaceAllUsesWith(ld);
          PN->replaceAllUsesWith(PN->getIncomingValueForBlock(Incoming));
          PN->eraseFromParent();
          ld2->eraseFromParent();
          continue;
        } else {
          llvm::errs() << "phinode cmp uses odd load with diff values\n";
          ld->dump();
          ld2->dump();
          H->getParent()->dump();
        }
      }
    }

    if( !PN->getType()->isIntegerTy() ) {
      errs() << "phinode uses non-int type\n";
      PN->dump();
      H->getParent()->dump();
      return make_pair(nullptr,nullptr);
    }
    if (BinaryOperator* Inc = dyn_cast<BinaryOperator>(PN->getIncomingValueForBlock(Backedge))) {
      if (Inc->getOpcode() == Instruction::Sub && Inc->getOperand(0) == PN) {
        IRBuilder<> build(Inc);
        auto val = build.CreateNeg(Inc->getOperand(1));
        auto tmp = build.CreateAdd(PN, val);
        assert( isa<BinaryOperator>(tmp) );
        auto newI = cast<BinaryOperator>(tmp);
        Inc->replaceAllUsesWith(newI);
        for (auto& tup : others) {
          if (std::get<1>(tup) == Inc) std::get<1>(tup) = newI;
          if (std::get<2>(tup) == Inc) std::get<2>(tup) = newI;
        }
        Inc->eraseFromParent();
        Inc = newI;
      }
      if (Inc->getOpcode() == Instruction::Add && (uncast(Inc->getOperand(0)) == PN || uncast(Inc->getOperand(1)) == PN) ) {
        if ( uncast(Inc->getOperand(1)) == PN ) Inc->swapOperands();
        assert( uncast(Inc->getOperand(0)) == PN);
        bool rpnr = false;
        bool incr = false;
        for(unsigned i = 0; i < cmp->getNumOperands(); i++) {
          bool hadr = uncast(cmp->getOperand(i)) == PN;
          rpnr |= hadr;
          bool hadi = uncast(cmp->getOperand(i)) == Inc;
          incr |= hadi;
          if (hadr | hadi) { assert(cmpIdx == -1); cmpIdx = i; }
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
        assert( !isa<PHINode>(Inc->getOperand(1)) );
        if (!recursiveMoveBefore(Incoming->getTerminator(), Inc->getOperand(1), DT, "1")) return make_pair(nullptr, nullptr);
        //assert( !isa<PHINode>(PN->getIncomingValueForBlock(Incoming)) );
        if (!recursiveMoveBefore(Incoming->getTerminator(), PN->getIncomingValueForBlock(Incoming), DT, "2")) return make_pair(nullptr, nullptr);
      } else {
        errs() << "no add found for:\n"; PN->dump(); Inc->dump();
        H->getParent()->dump();
        return make_pair(nullptr,nullptr);
      }
    } else {
      errs() << "no inc found for:\n"; PN->dump(); PN->getParent()->getParent()->dump();
      return make_pair(nullptr, nullptr);
    }
    ++I;
  }

  if (RPN == 0) {
    errs() << "<no RPN>\n";
    cmp->dump();
    errs() << "<---->\n";
    H->dump();
    errs() << "<---->\n";
    for( auto a : others ) { std::get<0>(a)->dump(); }
    errs() << "</no RPN>\n";
    return make_pair(nullptr,nullptr);
  }

  llvm::Value* mul;
  llvm::Value* newV;

  SmallPtrSet<llvm::Value*, 4> toIgnore;
  {
    BasicBlock* Spawned = detacher->getTerminator()->getSuccessor(0);

    if (cilk::getNumPred(Spawned) > 1) {
      BasicBlock* ts = BasicBlock::Create(Spawned->getContext(), Spawned->getName()+".fx", Spawned->getParent(), detacher);
      IRBuilder<> b(ts);
      b.CreateBr(Spawned);
      detacher->getTerminator()->setSuccessor(0,ts);
      llvm::BasicBlock::iterator i = Spawned->begin();
      while (auto phi = llvm::dyn_cast<llvm::PHINode>(i)) {
        int idx = phi->getBasicBlockIndex(detacher);
        phi->setIncomingBlock(idx, ts);
        ++i;
      }
      Spawned = ts;
    }

    IRBuilder<> builder(Spawned->getFirstNonPHIOrDbgOrLifetime());
    if( isOne(amt) ) mul = RPN;
    else toIgnore.insert(mul = builder.CreateMul(RPN, amt, "indmul"));
    if( isZero(RPN->getIncomingValueForBlock(Incoming) )) newV = mul;
    else toIgnore.insert(newV = builder.CreateAdd(mul, RPN->getIncomingValueForBlock(Incoming), "indadd"));

    for( auto a : others ) {
      llvm::Value* val = builder.CreateSExtOrTrunc(RPN, std::get<0>(a)->getType());
      if (val != RPN) toIgnore.insert(val);
      llvm::Value* amt0 = std::get<2>(a);
      if( !isOne(amt0) ) val = builder.CreateMul(val,amt0, "vmul");
      if (val != RPN) toIgnore.insert(val);
      llvm::Value* add0 = std::get<0>(a)->getIncomingValueForBlock(Incoming);
      if( !isZero(add0) ) val = builder.CreateAdd(val,add0, "vadd");
      if (val != RPN) toIgnore.insert(val);
      assert( isa<Instruction>(val) );
      Instruction* ival = cast<Instruction>(val);

      for (auto& u : std::get<0>(a)->uses()) {
        assert( isa<Instruction>(u.getUser()) );
        Instruction *user = cast<Instruction>(u.getUser());

        //No need to override use in PHINode itself
        if (user == std::get<0>(a)) continue;
        //No need to override use in increment
        if (user == std::get<1>(a)) continue;

        if (!attemptRecursiveMoveAfter(ival, user, DT)) {
          val->dump();
          user->dump();
          std::get<0>(a)->dump();
          H->getParent()->dump();
          llvm::errs() << "FAILED TO MOVE\n";
          return make_pair(nullptr, nullptr);
        }
        assert(DT.dominates(ival, user));
      }
      {
        auto tmp = std::get<0>(a);
        tmp->replaceAllUsesWith(val);
        for (auto& tup : others) {
          if (std::get<1>(tup) == tmp) std::get<1>(tup) = tmp;
          if (std::get<2>(tup) == tmp) std::get<2>(tup) = tmp;
        }
        tmp->eraseFromParent();
      }
      if(std::get<1>(a)->getNumUses() == 0) {
        auto tmp = std::get<1>(a);
        tmp->eraseFromParent();
      }
    }
  }

  std::vector<Use*> uses;
  for( Use& U : RPN->uses() ) uses.push_back(&U);
  for( Use* Up : uses ) {
    Use &U = *Up;
    assert( isa<Instruction>(U.getUser()) );
    Instruction *I = cast<Instruction>(U.getUser());
    if( I == INCR ) INCR->setOperand(1, ConstantInt::get( RPN->getType(), 1 ) );
    else if( toIgnore.count(I) > 0 && I != RPN ) continue;
    else if( uncast(I) == cmp || I == cmp->getOperand(0) || I == cmp->getOperand(1) || uncast(I) == cmp || I == RPN || I->getParent() == cmp->getParent() || I->getParent() == detacher) continue;
    else {
      assert( isa<Instruction>(newV) );
      Instruction* ival = cast<Instruction>(newV);
      assert( isa<Instruction>(U.getUser()) );
      if (!attemptRecursiveMoveAfter(ival, cast<Instruction>(U.getUser()), DT)) {
        llvm::errs() << "newV: ";
        newV->dump();
        llvm::errs() << "U: ";
        U->dump();
        llvm::errs() << "I: ";
        I->dump();
        llvm::errs() << "uncast(I): ";
        uncast(I)->dump();
        llvm::errs() << "errs: ";
        cmp->dump();
        llvm::errs() << "RPN: ";
        RPN->dump();
        H->getParent()->dump();
        llvm::errs() << "FAILED TO MOVE2\n";
        return make_pair(nullptr, nullptr);
      }
      assert( DT.dominates((Instruction*) newV, U) );
      U.set( newV );
    }
  }

  IRBuilder<> build(cmp);
  llvm::Value* val = build.CreateSExtOrTrunc(cmp->getOperand(1-cmpIdx),RPN->getType());
  llvm::Value* adder = RPN->getIncomingValueForBlock(Incoming);
  llvm::Value* amt0  = amt;

  int cast_type = 0;
  if (isa<TruncInst>(RPN)) cast_type = 1;
  if (isa<SExtInst>(RPN))  cast_type = 2;
  if (isa<ZExtInst>(RPN))  cast_type = 3;

  switch(cast_type) {
    default:;
    case 1: amt0 = build.CreateTrunc(amt0,RPN->getType());
    case 2: amt0 = build.CreateSExt( amt0,RPN->getType());
    case 3: amt0 = build.CreateZExt( amt0,RPN->getType());
  }
  switch(cast_type){
    default:;
    case 1: adder = build.CreateTrunc(adder,RPN->getType());
    case 2: adder = build.CreateSExt( adder,RPN->getType());
    case 3: adder = build.CreateZExt( adder,RPN->getType());
  }

  {
    Value *bottom = adder, *top = val;
    if (opc == RPN && DT.dominates(detacher->getTerminator(), cmp)) {
      cmp->setOperand(cmpIdx, INCR);
      top = build.CreateAdd(top, amt0, "toplen");
    }
    int dir = 0;
    switch (cmp->getPredicate()) {
      case CmpInst::ICMP_UGE:
      case CmpInst::ICMP_UGT:
      case CmpInst::ICMP_SGE:
      case CmpInst::ICMP_SGT:
        dir = -1; break;
      case CmpInst::ICMP_ULE:
      case CmpInst::ICMP_ULT:
      case CmpInst::ICMP_SLE:
      case CmpInst::ICMP_SLT:
        dir = +1;break;
      default:
        dir = 0;break;
    }
    if ( (dir < 0 && cmpIdx == 0) || (dir > 0 && cmpIdx != 0))
      std::swap(bottom, top);

    if (!isZero(bottom)) val = build.CreateSub(top, bottom, "sublen");
    else val = top;

    switch (cmp->getPredicate() ) {
      case CmpInst::ICMP_UGT:
      case CmpInst::ICMP_SGT:
      case CmpInst::ICMP_ULT:
      case CmpInst::ICMP_SLT:
        val = subOne(val, "subineq");
        break;
      case CmpInst::ICMP_SLE:
      case CmpInst::ICMP_ULE:
      case CmpInst::ICMP_SGE:
      case CmpInst::ICMP_UGE:
      default:
        break;
    }
  }
  {
    switch (cmp->getPredicate() ) {
      case CmpInst::ICMP_SLE:
      case CmpInst::ICMP_ULE:
      case CmpInst::ICMP_ULT:
      case CmpInst::ICMP_SLT:
        if (cmpIdx == 1) amt0 = neg(amt0); break;
      case CmpInst::ICMP_SGE:
      case CmpInst::ICMP_UGE:
      case CmpInst::ICMP_UGT:
      case CmpInst::ICMP_SGT:
        if (cmpIdx == 0) amt0 = neg(amt0); break;
      case CmpInst::ICMP_NE:
        //amt0 = build.CreateSelect(build.CreateICmpSGT(amt0,ConstantInt::get(val->getType(), 0)),amt0,neg(amt0));
      default:
        break;
    }
    if (!isOne(amt0)) val = build.CreateSDiv(val, amt0, "divlen");
    if (cmp->getPredicate()!=CmpInst::ICMP_NE) val = addOne(val, "nepred");
  }

  cmp->setPredicate(CmpInst::ICMP_NE);
  cmp->setOperand(cmpIdx, RPN);
  cmp->setOperand(1-cmpIdx, val);

  RPN->setIncomingValue(RPN->getBasicBlockIndex(Incoming),  ConstantInt::get(RPN->getType(), 0));

  return make_pair(RPN, val);
}

void removeFromAll(Loop* L, BasicBlock* B){
  if( !L ) return;
  if( L->contains(B) ) L->removeBlockFromLoop(B);
  removeFromAll(L->getParentLoop(), B);
}

template<typename A, typename B> bool contains(const A& a, const B& b) {
  return std::find(a.begin(), a.end(), b) != a.end();
}

BasicBlock* getTrueExit(Loop *L){
  SmallVector< BasicBlock *, 32> exitBlocks;
  L->getExitBlocks(exitBlocks);
  SmallPtrSet<BasicBlock *, 32> exits(exitBlocks.begin(), exitBlocks.end());
  SmallPtrSet<BasicBlock *, 32> alsoLoop;

  bool toRemove = true;
  while (toRemove) {
    toRemove = false;
    if (exits.size() >= 2) {
      for (auto tempExit : exits) {
        SmallPtrSet<BasicBlock *, 32> reachable;
        std::vector<BasicBlock*> Q = { tempExit };
        bool valid = true;
        while(!Q.empty() && valid) {
          auto m = Q.back();
          Q.pop_back();
          if( isa<UnreachableInst>(m->getTerminator()) ) { reachable.insert(m); continue; }
          else if( auto b = dyn_cast<BranchInst>(m->getTerminator()) ) {
            reachable.insert(m);
            for( unsigned i=0; i<b->getNumSuccessors(); i++ ) {
               auto suc = b->getSuccessor(i);
               if( L->contains(suc) || contains(exitBlocks,suc) || contains(alsoLoop, suc) || contains(reachable, suc) ) {

               } else{
                Q.push_back(suc);
                break;
              }
            }
          }
          else valid = false;
        }
        if (valid && reachable.size() > 0) {
          for( auto b : reachable){
            exits.erase(b);
            alsoLoop.insert(b);
          }
          toRemove = true;
        }
      }
    }
  }

  if (exits.size() == 1) return *exits.begin();
  return nullptr;
}

BasicBlock* continueToFindSync(BasicBlock* endL) {
  //TODO consider lifetime intrinsics
  while (endL && !isa<SyncInst>(endL->getTerminator())) {
    if (getNonPhiSize(endL) == 1 && isa<BranchInst>(endL->getTerminator()) && endL->getTerminator()->getNumSuccessors() == 1) {
      endL = endL->getTerminator()->getSuccessor(0);
    }
    else
      endL = nullptr;
  }

  if (endL)
    assert(endL && isa<SyncInst>(endL->getTerminator()));

  return endL;
}


/*
cilk_for_recursive(count_t low, count_t high, ...) {
tail_recurse:
  count_t count = high - low;
  if (count > grain)
  {
      // Invariant: count >= 2
      count_t mid = low + count / 2;
      spawn cilk_for_recursive(low, mid, grain, ...);
      low = mid;
      goto tail_recurse;
  }

  for(int i=low; i<high; i++) {
    body(i, data);
  }
  sync;
*/
bool createDACOnExtractedFunction(Function* extracted, LLVMContext &Ctx, std::vector<Value*>& ext_args) {
  Function::arg_iterator args = extracted->arg_begin();
  Argument *low0 = &*args;
  args++;
  Argument *high0 = &*args;
  args++;
  Argument *grain = &*args;

  BasicBlock *entry = &extracted->getEntryBlock();
  BasicBlock *body = entry->getTerminator()->getSuccessor(0);
  BasicBlock *tail_recurse = entry->splitBasicBlock(entry->getTerminator(), "tail_recurse");
  BasicBlock *recur = BasicBlock::Create(Ctx, "recur", extracted);
  recur->moveAfter(tail_recurse);
  tail_recurse->getTerminator()->eraseFromParent();

  IRBuilder<> trbuilder(tail_recurse);
  PHINode* low = trbuilder.CreatePHI(low0->getType(), 2, "low");
  low0->replaceAllUsesWith(low);
  Value* count = trbuilder.CreateSub(high0, low, "count");
  Value* cond = trbuilder.CreateICmpUGT(count, grain);
  trbuilder.CreateCondBr(cond, recur, body);
  low->addIncoming(low0, entry);

  IRBuilder<> rbuilder(recur);
  Value *mid = rbuilder.CreateAdd(low, rbuilder.CreateUDiv(count, ConstantInt::get(count->getType(), 2)), "mid");
  BasicBlock *detached = BasicBlock::Create(Ctx, "detached", extracted);
  detached->moveAfter(recur);
  BasicBlock *reattached = BasicBlock::Create(Ctx, "reattached", extracted);
  reattached->moveAfter(detached);
  rbuilder.CreateDetach(detached, reattached);

  IRBuilder<> dbuilder(detached);

  //Fill in closure arguments
  std::vector<Value*> next_args;
  args = extracted->arg_begin();
  for (unsigned i=0, len = ext_args.size(); i<len; i++) {
    next_args.push_back(&*args);
    args++;
  }

  //Replace the bounds arguments
  next_args[0] = low;
  next_args[1] = mid;
  next_args[2] = grain;

  dbuilder.CreateCall(extracted, next_args);
  dbuilder.CreateReattach(reattached);

  IRBuilder<> rebuilder(reattached);
  rebuilder.CreateBr(tail_recurse);
  low->addIncoming(mid, reattached);

  SmallVector<BasicBlock *, 32> blocks;
  for (BasicBlock& BB : *extracted) { blocks.push_back(&BB); }

  for (BasicBlock* BB : blocks) {
    if (ReturnInst *Ret = dyn_cast<ReturnInst>(BB->getTerminator())) {
      auto tret = BB->splitBasicBlock(Ret);
      BB->getTerminator()->eraseFromParent();
      IRBuilder<> build(BB);
      build.CreateSync(tret);
    }
  }
  return true;
}

bool Loop2Cilk::performDAC(Loop *L, LPPassManager &LPM) {

  BasicBlock* Header = L->getHeader();
  Module* M = Header->getParent()->getParent();
  LLVMContext &Ctx = M->getContext();
  assert(Header);

  Loop* parentL = L->getParentLoop();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  TerminatorInst* T = Header->getTerminator();
  if (!isa<BranchInst>(T)) {
    BasicBlock *Preheader = L->getLoopPreheader();
    if (isa<BranchInst>(Preheader->getTerminator())) {
      T = Preheader->getTerminator();
      Header = Preheader;
    } else if (isa<SyncInst>(Preheader->getTerminator())) {
      // BasicBlock *ph = BasicBlock::Create(Ctx, "sync.br", Header->getParent());
      // DT.addNewBlock(ph, Preheader);
      // ph->moveAfter(Preheader);
      // IRBuilder <> b(ph);
      // T = Preheader->getTerminator();
      // assert(T->getSuccessor(0) == Header);
      // b.CreateBr(T->getSuccessor(0));
      // T->setSuccessor(0,ph);
      // Header->dump();
      // T = ph->getTerminator(); Header = ph;
      BasicBlock *NewPreheader = SplitEdge(Preheader, Header, &DT, &LI);
      SyncInst::Create(NewPreheader, Preheader->getTerminator());
      Preheader->getTerminator()->eraseFromParent();
      T = BranchInst::Create(Header, NewPreheader->getTerminator());
      NewPreheader->getTerminator()->eraseFromParent();
      Header = NewPreheader;
    } else {
      llvm::errs() << "Loop not entered via branch instance\n";
      T->dump();
      Preheader->dump();
      Header->dump();
      return false;
    }
  }

  assert (isa<BranchInst>(T));
  BranchInst* B = cast<BranchInst>(T);

  BasicBlock *detacher = nullptr, *syncer = nullptr;
  /////!!< BEGIN ESTABLISH DETACH/SYNC BLOCKS
  if (B->getNumSuccessors() != 2) {
    detacher = B->getSuccessor(0);
    if (!isa<DetachInst>(detacher->getTerminator())) {
      return false;
    }
    assert(detacher && isa<DetachInst>(detacher->getTerminator()));

    BasicBlock* endL = getTrueExit(L);
    endL = continueToFindSync(endL);

    if (endL) {
      syncer = endL;
      assert(syncer && isa<SyncInst>(syncer->getTerminator()));
    } else {
      return false;
    }
  } else {

    if (isa<DetachInst>(B->getSuccessor(0)->getTerminator())) {
      detacher = B->getSuccessor(0);
      syncer   = B->getSuccessor(1);
    } else if (isa<DetachInst>(B->getSuccessor(1)->getTerminator())) {
      detacher = B->getSuccessor(1);
      syncer   = B->getSuccessor(0);
    } else {
      //errs() << "No detach found" << "\n";
      //detacher->dump();
      return false;
    }

    syncer = continueToFindSync(syncer);
    if (!syncer) {
      //errs() << "No sync found" << "\n";
      return false;
    }

    BasicBlock* done = getTrueExit(L);
    if (!done) {
      //errs() << "no unique exit block\n";
      return false;
    }

    if (BranchInst* BI = dyn_cast<BranchInst>(done->getTerminator())) {
      if( BI->getNumSuccessors() == 2 ) {
        if( BI->getSuccessor(0) == detacher && BI->getSuccessor(1) == syncer )
          done = syncer;
        if( BI->getSuccessor(1) == detacher && BI->getSuccessor(0) == syncer )
          done = syncer;
      }
    }

    if (done->getUniquePredecessor() == syncer) {
      auto term = done->getTerminator();
      bool good = true;
      for (unsigned i=0; i<term->getNumSuccessors(); i++)
        if (L->contains( term->getSuccessor(i))) {
          good = false;
          break;
        }
      if (good) done = syncer;
    }
    if (done != syncer) {
      errs() << "exit != sync\n";
      return false;
    }
  }
  /////!!< END ESTABLISH DETACH/SYNC BLOCKS
  assert(syncer && isa<SyncInst>(syncer->getTerminator()));
  assert(detacher && isa<DetachInst>(detacher->getTerminator()));

  DetachInst* det = cast<DetachInst>(detacher->getTerminator());

   {
    SmallPtrSet<BasicBlock *, 32> functionPieces;
    SmallVector<BasicBlock*, 32 > reattachB;
    if (!llvm::cilk::populateDetachedCFG(*det, DT, functionPieces, reattachB, false)) return false;
    for (BasicBlock* BB : functionPieces) {
      for (Instruction &I : *BB) {
        if (CallInst* ca = dyn_cast<CallInst>(&I)) {
          if (ca->getCalledFunction() == Header->getParent()) {
            //errs() << "Selecting successive spawn in place of DAC for recursive cilk_for in function " << Header->getParent()->getName() << "|" << Header->getName() << "\n";
            return false;
          }
        }
      }
    }
  }

  /////!!< REQUIRE DETACHER BLOCK IS EMPTY EXCEPT FOR BRANCH
  while (getNonPhiSize(detacher)!=1) {
    Instruction* badInst = getLastNonTerm(detacher);
    if (!badInst->mayWriteToMemory()) {
      bool dominated = true;
      for (const Use &U : badInst->uses()) {
        if (!DT.dominates(BasicBlockEdge(detacher, det->getSuccessor(0) ), U) ) {
          errs() << "use not dominated:\n";
          U->dump();
          dominated = false;
          break;
        }
      }
      if (dominated) {
        badInst->moveBefore( getFirstPostPHI(det->getSuccessor(0)) );
        continue;
      }
    } else
      errs() << "mayWrite:\n";
    errs() << "invalid detach size of " << getNonPhiSize(detacher) << "|" << detacher->size() << "\n";
    detacher->dump();
    return false;
  }

  /////!!< REQUIRE SYNC BLOCK HAS ONLY PHI's / EXIT
  while (getNonPhiSize(syncer)!=1) {
    Instruction* badInst = getLastNonTerm(syncer);
    if (!badInst->mayWriteToMemory()) {
      badInst->moveBefore( getFirstPostPHI(syncer->getTerminator()->getSuccessor(0)) );
    } else {
      errs() << "invalid sync size" << "\n";
      syncer->dump();
      return false;
    }
  }

  /////!!< REMOVE ANY SYNC BLOCK PHI's
  while (syncer->size() != 1) {
    assert( isa<PHINode>(&syncer->front()) );
    PHINode* pn = cast<PHINode>(&syncer->front());
    if (pn->getNumIncomingValues() != 1 ) {
      errs() << "invalid phi for sync\n";
      return false;
    }
    pn->replaceAllUsesWith(pn->getIncomingValue(0));
    pn->eraseFromParent();
  }

  std::pair<PHINode*,Value*> indVarResult = getIndVar(L, detacher, DT);
  PHINode* oldvar = indVarResult.first;
  Value* cmp = indVarResult.second;

  //oldvar guarenteed to be canonical (start at 0, inc by 1, end at ...)
  if (!oldvar) {
      errs() << "no induction var\n";
      return false;
  }

  assert( ( L->getHeader()->size() == getNonPhiSize(L->getHeader()) + 1 ) && "Can only cilk_for loops with only 1 phi node " );

  bool simplified = false;
  while (!simplified) {
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

  DT.recalculate(*L->getHeader()->getParent());

  llvm::CallInst* call = 0;

  if (!recursiveMoveBefore(Header->getTerminator(), cmp, DT, "3")) {
    errs() << "cmp not moved\n"; cmp->dump();
    L->getHeader()->getParent()->dump();
    return false;
  }

  std::vector<Value*> ext_args;
  Function* extracted = llvm::cilk::extractDetachBodyToFunction(*det, DT, &call, /*closure*/ oldvar, &ext_args);

  if (!extracted) {
    errs() << "not extracted\n";
    return false;
  }

  // FIX FOR THE DAC recur
  createDACOnExtractedFunction(extracted, Ctx, ext_args);

  for (BasicBlock& b : extracted->getBasicBlockList()) {
      removeFromAll(parentL, &b);
      LI.changeLoopFor(&b, nullptr);
      LI.removeBlock(&b);
  }

  auto a1 = det->getSuccessor(0);
  auto a2 = det->getSuccessor(1);

  oldvar->removeIncomingValue( 1U );
  oldvar->removeIncomingValue( 0U );
  assert( oldvar->getNumUses() == 0 );
  assert( det->use_empty() );

  det->eraseFromParent();
  if (cilk::getNumPred(a2) == 0) {
    auto tmp = a1;
    a1 = a2;
    a2 = tmp;
  }

  if (parentL)
    parentL->removeChildLoop(std::find(parentL->getSubLoops().begin(), parentL->getSubLoops().end(), L));

  if (auto term = a1->getTerminator())
    term->eraseFromParent();
  if (auto term = a2->getTerminator())
    term->eraseFromParent();

  LI.removeBlock(a1);
  removeFromAll(parentL, a1);
  DeleteDeadBlock(a1);
  if (a1 != a2) {
    LI.removeBlock(a2);
    removeFromAll(parentL, a2);
    DeleteDeadBlock(a2);
  }

  if (auto term = Header->getTerminator())
    term->eraseFromParent();

  IRBuilder<> b2(Header);
  b2.CreateBr(detacher);

   {
    IRBuilder<> b(detacher);

    LLVMContext &Ctx = M->getContext();

    IRBuilder<> builder2(syncer);
    if (!syncer->empty())
      builder2.SetInsertPoint(&*syncer->begin());

    auto count = cmp;
    Value* cond = b.CreateICmpSLT(count, ConstantInt::get(count->getType(), 1));
    BasicBlock *graint = BasicBlock::Create(Ctx, "graint", detacher->getParent());
    graint->moveBefore(syncer);
    if (parentL) {
      parentL->addBasicBlockToLoop(graint, LI);
    }

    b.CreateCondBr(cond, syncer, graint);

    b.SetInsertPoint(graint);
    Value* P8 = b.CreateIntCast(cilk::GetOrCreateWorker8(*detacher->getParent()), count->getType(), false);
    Value* n = b.CreateUDiv(b.CreateSub(b.CreateAdd(count, P8), ConstantInt::get(count->getType(), 1)), P8);
    Value* cutoff = ConstantInt::get(count->getType(), 2048);
    Value* c2 = b.CreateICmpUGT(n, cutoff);
    Value* pn = b.CreateSelect(c2, cutoff, n);

    ext_args[0] = ConstantInt::get(count->getType(), 0);
    ext_args[1] = count;
    ext_args[2] = pn;

    b.CreateCall(extracted, ext_args);

    assert (syncer->size() == 1);
    b.CreateBr(syncer);
  }

  DT.recalculate(*Header->getParent());
  L->invalidate();

  if (parentL) parentL->verifyLoop();

  return true;
}


bool Loop2Cilk::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (llvm::verifyFunction(*L->getHeader()->getParent(), &llvm::errs())) {
    L->getHeader()->getParent()->dump();
    assert(0);
  }
  if (skipLoop(L)) {
    return false;
  }
  bool ans = performDAC(L, LPM);
  if (llvm::verifyFunction(*L->getHeader()->getParent(), &llvm::errs())) {
    L->getHeader()->getParent()->dump();
    assert(0);
  }
  return ans;
}
