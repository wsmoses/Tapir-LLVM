//===- TapirUtils.cpp - Utility functions for handling Tapir --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements several utility functions for operating with Tapir.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Tapir/CilkABI.h"
#include "llvm/Transforms/Tapir/OpenMPABI.h"
#include "llvm/Transforms/Tapir/PTXABI.h"
#include "llvm/Transforms/Tapir/QthreadsABI.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/LoopSpawning.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "tapir"

TapirTarget *llvm::getTapirTargetFromType(TapirTargetType Type) {
  switch(Type) {
  case TapirTargetType::Cilk:
    return new CilkABI(/*useRuntimeForLoop=*/false);
  case TapirTargetType::CilkLegacy:
    return new CilkABI(/*useRuntimeForLoop=*/true);
  case TapirTargetType::OpenMP:
    return new OpenMPABI();
  case TapirTargetType::PTX:
    return new PTXABI();
  case TapirTargetType::Qthreads:
    return new QthreadsABI();
  case TapirTargetType::None:
    return nullptr;
  case TapirTargetType::Serial:
  default:
    assert(0 && "Tapir target not implemented");
    return nullptr;
  }
}

bool llvm::verifyDetachedCFG(const DetachInst &Detach, DominatorTree &DT,
                             bool error) {
  BasicBlock *Spawned  = Detach.getDetached();
  BasicBlock *Continue = Detach.getContinue();
  BasicBlockEdge DetachEdge(Detach.getParent(), Spawned);

  SmallVector<BasicBlock *, 32> Todo;
  SmallPtrSet<BasicBlock *, 32> functionPieces;
  SmallVector<BasicBlock *, 4> WorkListEH;
  Todo.push_back(Spawned);

  while (!Todo.empty()) {
    BasicBlock *BB = Todo.pop_back_val();

    if (!functionPieces.insert(BB).second)
      continue;

    TerminatorInst* Term = BB->getTerminator();
    if (Term == nullptr) return false;
    if (ReattachInst* Inst = dyn_cast<ReattachInst>(Term)) {
      //only analyze reattaches going to the same continuation
      if (Inst->getSuccessor(0) != Continue) continue;
      continue;
    } else if (DetachInst* Inst = dyn_cast<DetachInst>(Term)) {
      assert(Inst != &Detach && "Found recursive Detach!");
      Todo.push_back(Inst->getSuccessor(0));
      Todo.push_back(Inst->getSuccessor(1));
      continue;
    } else if (SyncInst* Inst = dyn_cast<SyncInst>(Term)) {
      //only sync inner elements, consider as branch
      Todo.push_back(Inst->getSuccessor(0));
      continue;
    } else if (isa<BranchInst>(Term) || isa<SwitchInst>(Term) ||
               isa<InvokeInst>(Term)) {
      for (BasicBlock *Succ : successors(BB)) {
        if (!DT.dominates(DetachEdge, Succ))
          // We assume that this block is an exception-handling block and save
          // it for later processing.
          WorkListEH.push_back(Succ);
        else
          Todo.push_back(Succ);
      }
      continue;
    } else if (isa<UnreachableInst>(Term) || isa<ResumeInst>(Term)) {
      continue;
    } else {
      assert(!error && "Detached block did not absolutely terminate in reattach");
      return false;
    }
  }
  {
    SmallPtrSet<BasicBlock *, 4> Visited;
    while (!WorkListEH.empty()) {
      BasicBlock *BB = WorkListEH.pop_back_val();
      if (!Visited.insert(BB).second)
        continue;

      // Make sure that the control flow through these exception-handling blocks
      // cannot re-enter the blocks being outlined.
      assert(!functionPieces.count(BB) &&
             "EH blocks for a detached region reenter that region.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't perform an ordinary return.
      assert(!isa<ReturnInst>(BB->getTerminator()) &&
             "EH block terminated by return.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't reattach to the detached CFG's continuation.
      if (ReattachInst *RI = dyn_cast<ReattachInst>(BB->getTerminator()))
        assert(RI->getSuccessor(0) != Continue &&
               "Exit block reaches a reattach to the continuation.");

      for (BasicBlock *Succ : successors(BB))
        WorkListEH.push_back(Succ);
    }
  }
  return true;
}

bool llvm::populateDetachedCFG(
    const DetachInst &Detach, DominatorTree &DT,
    SmallPtrSetImpl<BasicBlock *> &functionPieces,
    SmallVectorImpl<ReattachInst*> &reattachB,
    SmallPtrSetImpl<BasicBlock *> &ExitBlocks,
    bool error) {
  return llvm::populateDetachedCFG(Detach.getDetached(),
    Detach, DT, functionPieces, reattachB, ExitBlocks, error);
}

bool llvm::populateDetachedCFG(
    BasicBlock* startSearch, const DetachInst& Detach,
    DominatorTree &DT,
    SmallPtrSetImpl<BasicBlock *> &functionPieces,
    SmallVectorImpl<ReattachInst*> &reattachB,
    SmallPtrSetImpl<BasicBlock *> &ExitBlocks,
    bool error) {
  SmallVector<BasicBlock *, 32> Todo;
  SmallVector<BasicBlock *, 4> WorkListEH;

  BasicBlock *Spawned  = Detach.getDetached();
  BasicBlock *Continue = Detach.getContinue();
  BasicBlockEdge DetachEdge(Detach.getParent(), Spawned);
  Todo.push_back(startSearch);

  while (!Todo.empty()) {
    BasicBlock *BB = Todo.pop_back_val();

    if (!functionPieces.insert(BB).second)
      continue;

    TerminatorInst *Term = BB->getTerminator();
    if (Term == nullptr) return false;
    if (auto reattach = dyn_cast<ReattachInst>(Term)) {
      // only analyze reattaches going to the same continuation
      if (Term->getSuccessor(0) != Continue) continue;
      reattachB.push_back(reattach);
      continue;
    } else if (isa<DetachInst>(Term)) {
      assert(Term != &Detach && "Found recursive detach!");
      Todo.push_back(Term->getSuccessor(0));
      Todo.push_back(Term->getSuccessor(1));
      continue;
    } else if (isa<SyncInst>(Term)) {
      //only sync inner elements, consider as branch
      Todo.push_back(Term->getSuccessor(0));
      continue;
    } else if (isa<BranchInst>(Term) || isa<SwitchInst>(Term) ||
               isa<InvokeInst>(Term)) {
      for (BasicBlock *Succ : successors(BB)) {
        if (!DT.dominates(DetachEdge, Succ)) {
          // We assume that this block is an exception-handling block and save
          // it for later processing.
          ExitBlocks.insert(Succ);
          WorkListEH.push_back(Succ);
        } else {
          Todo.push_back(Succ);
        }
      }
      // We don't bother cloning unreachable exits from the detached CFG at this
      // point.  We're cloning the entire detached CFG anyway when we outline
      // the function.
      continue;
    } else if (isa<UnreachableInst>(Term) || isa<ResumeInst>(Term)) {
      continue;
    } else {
      assert(!error && "Detached block did not absolutely terminate in reattach");
      return false;
    }
  }

  // Find the exit-handling blocks.
  {
    SmallPtrSet<BasicBlock *, 4> Visited;
    while (!WorkListEH.empty()) {
      BasicBlock *BB = WorkListEH.pop_back_val();
      if (!Visited.insert(BB).second)
        continue;

      // Make sure that the control flow through these exception-handling blocks
      // cannot re-enter the blocks being outlined.
      assert(!functionPieces.count(BB) &&
             "EH blocks for a detached region reenter that region.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't perform an ordinary return.
      assert(!isa<ReturnInst>(BB->getTerminator()) &&
             "EH block terminated by return.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't reattach to the detached CFG's continuation.
      if (ReattachInst *RI = dyn_cast<ReattachInst>(BB->getTerminator()))
        assert(RI->getSuccessor(0) != Continue &&
               "Exit block reaches a reattach to the continuation.");

      // if (isa<ResumeInst>(BB-getTerminator()))
      //   ResumeBlocks.push_back(BB);

      for (BasicBlock *Succ : successors(BB)) {
        ExitBlocks.insert(Succ);
        WorkListEH.push_back(Succ);
      }
    }

    // Visited now contains exception-handling blocks that we want to clone as
    // part of outlining.
    for (BasicBlock *EHBlock : Visited)
      functionPieces.insert(EHBlock);
  }

  return true;
}

//Returns true if success
Function *llvm::extractDetachBodyToFunction(DetachInst &detach,
                                            DominatorTree &DT,
                                            AssumptionCache &AC,
                                            CallInst **call) {
  BasicBlock *Detacher = detach.getParent();
  Function &F = *(Detacher->getParent());

  BasicBlock *Spawned  = detach.getDetached();
  BasicBlock *Continue = detach.getContinue();

  SmallPtrSet<BasicBlock *, 32> functionPieces;
  SmallVector<BranchInst*, 32> branchB;
  SmallPtrSet<BasicBlock *, 4> ExitBlocks;

  assert(Spawned->getUniquePredecessor() &&
         "Entry block of detached CFG has multiple predecessors.");
  assert(Spawned->getUniquePredecessor() == Detacher &&
         "Broken CFG.");

  {
    SmallVector<ReattachInst*, 32> reattachB;
    if (!populateDetachedCFG(detach, DT, functionPieces, reattachB,
                             ExitBlocks))
      return nullptr;

    /*change reattach to branch*/
    for(auto reattach: reattachB) {
      BranchInst* toReplace = BranchInst::Create(reattach->getSuccessor(0));
      ReplaceInstWithInst(reattach, toReplace);
      branchB.push_back(toReplace);
    }
  }

  // Check the spawned block's predecessors.
  for (BasicBlock *BB : functionPieces) {
    int detached_count = 0;
    if (ExitBlocks.count(BB))
      continue;
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      BasicBlock *Pred = *PI;
      if (detached_count == 0 && BB == Spawned && Pred == detach.getParent()) {
        detached_count = 1;
        continue;
      }
      assert(functionPieces.count(Pred) &&
             "Block inside of detached context branched into from outside branch context");
    }
  }

  // Get the inputs and outputs for the detached CFG.
  SetVector<Value *> Inputs, Outputs;
  SetVector<Value *> BodyInputs;
  findInputsOutputs(functionPieces, BodyInputs, Outputs, DT, &ExitBlocks);
  assert(Outputs.empty() &&
         "All results from detached CFG should be passed by memory already.");
  {
    // Scan for any sret parameters in BodyInputs and add them first.
    Value *SRetInput = nullptr;
    if (F.hasStructRetAttr()) {
      Function::arg_iterator ArgIter = F.arg_begin();
      if (F.hasParamAttribute(0, Attribute::StructRet))
	if (BodyInputs.count(&*ArgIter))
	  SRetInput = &*ArgIter;
      if (F.hasParamAttribute(1, Attribute::StructRet)) {
	++ArgIter;
	if (BodyInputs.count(&*ArgIter))
	  SRetInput = &*ArgIter;
      }
    }
    if (SRetInput) {
      DEBUG(dbgs() << "sret input " << *SRetInput << "\n");
      Inputs.insert(SRetInput);
    }
    // Add the remaining inputs.
    for (Value *V : BodyInputs)
      if (!Inputs.count(V))
	Inputs.insert(V);
  }

  // Clone the detached CFG into a helper function.
  ValueToValueMapTy VMap;
  Function *extracted;
  {
    SmallVector<ReturnInst *, 4> Returns;  // Ignore returns cloned.
    std::vector<BasicBlock *> blocks(functionPieces.begin(), functionPieces.end());

    extracted = CreateHelper(Inputs, Outputs, blocks,
                             Spawned, Detacher, Continue,
                             VMap, F.getParent(),
                             F.getSubprogram() != nullptr, Returns, ".cilk",
                             &ExitBlocks, nullptr, nullptr, nullptr, nullptr);

    assert(Returns.empty() && "Returns cloned when cloning detached CFG.");

    // Use a fast calling convention for the helper.
    extracted->setCallingConv(CallingConv::Fast);
    extracted->addFnAttr(Attribute::NoInline);
  }

  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  AddAlignmentAssumptions(&F, Inputs, VMap, &detach, &AC, &DT);

  // Add call to new helper function in original function.
  CallInst *TopCall;
  {
    // Create call instruction.
    IRBuilder<> Builder(&detach);
    TopCall = Builder.CreateCall(extracted, Inputs.getArrayRef());
    // Use a fast calling convention for the helper.
    TopCall->setCallingConv(CallingConv::Fast);
    TopCall->setDebugLoc(detach.getDebugLoc());
  }
  if (call)
    *call = TopCall;

  // Move allocas in the newly cloned detached CFG to the entry block of the
  // helper.
  {
    // Collect reattach instructions.
    SmallVector<Instruction *, 4> ReattachPoints;
    for (pred_iterator PI = pred_begin(Continue), PE = pred_end(Continue);
         PI != PE; ++PI) {
      BasicBlock *Pred = *PI;
      if (!isa<ReattachInst>(Pred->getTerminator())) continue;
      if (functionPieces.count(Pred))
        ReattachPoints.push_back(cast<BasicBlock>(VMap[Pred])->getTerminator());
    }

    // Move allocas in cloned detached block to entry of helper function.
    BasicBlock *ClonedDetachedBlock = cast<BasicBlock>(VMap[Spawned]);
    MoveStaticAllocasInBlock(&extracted->getEntryBlock(), ClonedDetachedBlock,
                             ReattachPoints);

    // We do not need to add new llvm.stacksave/llvm.stackrestore intrinsics,
    // because calling and returning from the helper will automatically manage
    // the stack appropriately.
  }

  for(auto branch : branchB) {
    auto BB = branch->getParent();
    BasicBlock::iterator BI = branch->getSuccessor(0)->begin();
    while (PHINode *P = dyn_cast<PHINode>(BI)) {
      P->removeIncomingValue(BB);
      ++BI;
    }
    while (BB->size()) {
      (&*--BB->end())->eraseFromParent();
    }
    IRBuilder<> b(BB);
    b.CreateUnreachable();
  }
  return extracted;
}

bool TapirTarget::shouldProcessFunction(const Function &F) {
  if (canDetach(&F))
    return true;
  return false;
}

bool llvm::isConstantMemoryFreeOperation(Instruction* I, bool allowsyncregion) {
  if (auto call = dyn_cast<CallInst>(I)) {
    auto id = call->getCalledFunction()->getIntrinsicID();
    return (id == Intrinsic::lifetime_start ||
            id == Intrinsic::lifetime_end ||
        (allowsyncregion && (id == Intrinsic::syncregion_start)));
  }
  return isa<BinaryOperator>(I) ||
      isa<CmpInst>(I) ||
      isa<ExtractElementInst>(I) ||
      isa<CatchPadInst>(I) || isa<CleanupPadInst>(I) ||
      isa<GetElementPtrInst>(I) ||
      isa<InsertElementInst>(I) ||
      isa<InsertValueInst>(I) ||
      isa<LandingPadInst>(I) ||
      isa<PHINode>(I) ||
      isa<SelectInst>(I) ||
      isa<ShuffleVectorInst>(I) ||
      // Unary
        isa<AllocaInst>(I) ||
        isa<CastInst>(I) ||
        isa<ExtractValueInst>(I);
}

bool llvm::isConstantOperation(Instruction* I, bool allowsyncregion) {
  if (auto call = dyn_cast<CallInst>(I)) {
    auto id = call->getCalledFunction()->getIntrinsicID();
    return (id == Intrinsic::lifetime_start ||
            id == Intrinsic::lifetime_end ||
        (allowsyncregion && (id == Intrinsic::syncregion_start)));
  }
  return
      isa<AtomicCmpXchgInst>(I) ||
      isa<AtomicRMWInst>(I) ||
      isa<BinaryOperator>(I) ||
      isa<CmpInst>(I) ||
      isa<ExtractElementInst>(I) ||
      isa<CatchPadInst>(I) || isa<CleanupPadInst>(I) ||
      isa<GetElementPtrInst>(I) ||
      isa<InsertElementInst>(I) ||
      isa<InsertValueInst>(I) ||
      isa<LandingPadInst>(I) ||
      isa<PHINode>(I) ||
      isa<SelectInst>(I) ||
      isa<ShuffleVectorInst>(I) ||
      isa<StoreInst>(I) ||
      // Unary
        isa<AllocaInst>(I) ||
        isa<CastInst>(I) ||
        isa<ExtractValueInst>(I) ||
        isa<LoadInst>(I) ||
        isa<VAArgInst>(I)
        ;
}

/*
spawn {
  A()
  spawn B();
}

A write | B write => can't move
A write | B read => can't move

A read | B read => can move
A read | B write => can't move
*/
bool llvm::doesDetachedInstructionAlias(AliasSetTracker &CurAST, const Instruction& I, bool FoundMod, bool FoundRef) {
  // Loads have extra constraints we have to verify before we can hoist them.
  if (const auto *LI = dyn_cast<LoadInst>(&I)) {
    if (!LI->isUnordered())
      return true; // Don't touch volatile/atomic loads!

    // Don't hoist loads which have may-aliased stores in predecessors.
    uint64_t Size = 0;
    if (LI->getType()->isSized())
      Size = I.getModule()->getDataLayout().getTypeStoreSize(LI->getType());

    AAMDNodes AAInfo;
    LI->getAAMetadata(AAInfo);

    auto ps = CurAST.getAliasSetForPointerIfExists(LI->getOperand(0), Size, AAInfo);
    if (ps == nullptr) return false;
    return ps->isMod();
  } else if(const auto *LI = dyn_cast<IndirectBrInst>(&I)) {
    // Don't hoist loads which have may-aliased stores in predecessors.
    AAMDNodes AAInfo;
    LI->getAAMetadata(AAInfo);
    auto ps = CurAST.getAliasSetForPointerIfExists(LI->getOperand(0), 0, AAInfo);
    if (ps == nullptr) return false;
    return ps->isMod();
  } else if (const auto *SI = dyn_cast<StoreInst>(&I)) {
    if (!SI->isUnordered())
      return true; // Don't touch volatile/atomic stores!

    // Don't hoist stores which have may-aliased loads in predecessors.
    uint64_t Size = 0;
    if (SI->getType()->isSized())
      Size = I.getModule()->getDataLayout().getTypeStoreSize(SI->getType());

    AAMDNodes AAInfo;
    SI->getAAMetadata(AAInfo);

    auto ps = CurAST.getAliasSetForPointerIfExists(LI->getOperand(0), Size, AAInfo);
    if (ps == nullptr) return false;
    return ps->isMod() | ps->isRef();
  } else if (const auto *CI = dyn_cast<CallInst>(&I)) {
    // Dbg info always legal.
    if (isa<DbgInfoIntrinsic>(I))
      return false;

    // Handle simple cases by querying alias analysis.
    FunctionModRefBehavior Behavior = CurAST.getAliasAnalysis().getModRefBehavior(CI);
    if (Behavior == FMRB_DoesNotAccessMemory)
      return false;

    if (AliasAnalysis::onlyReadsMemory(Behavior)) {
      if (!FoundMod)
        return false;
      // A readonly argmemonly function only reads from memory pointed to by
      // it's arguments with arbitrary offsets.  If we can prove there are no
      // writes to this memory in the loop, we can hoist or sink.
      if (AliasAnalysis::onlyAccessesArgPointees(Behavior)) {
        for (Value *Op : CI->arg_operands())
          if (Op->getType()->isPointerTy()) {
            auto ps = CurAST.getAliasSetForPointerIfExists(Op, MemoryLocation::UnknownSize, AAMDNodes());
            if (ps == nullptr) continue;
            if (ps->isMod()) return true;
          }
        return false;
      }
    }
  } else if (const auto *CI = dyn_cast<InvokeInst>(&I)) {
    // Dbg info always legal.
    if (isa<DbgInfoIntrinsic>(I))
      return false;

    // Handle simple cases by querying alias analysis.
    FunctionModRefBehavior Behavior = CurAST.getAliasAnalysis().getModRefBehavior(CI);
    if (Behavior == FMRB_DoesNotAccessMemory)
      return false;

    if (AliasAnalysis::onlyReadsMemory(Behavior)) {
      if (!FoundMod)
        return false;
      // A readonly argmemonly function only reads from memory pointed to by
      // it's arguments with arbitrary offsets.  If we can prove there are no
      // writes to this memory in the loop, we can hoist or sink.
      if (AliasAnalysis::onlyAccessesArgPointees(Behavior)) {
        for (Value *Op : CI->arg_operands())
          if (Op->getType()->isPointerTy() &&
              CurAST.getAliasSetForPointer(Op, MemoryLocation::UnknownSize, AAMDNodes()).isMod())
              return true;
        return false;
      }
    }
  }

  // Only these instructions are hoistable/sinkable.
  if (isa<BinaryOperator>(I) ||
      isa<CmpInst>(I) ||
      isa<ExtractElementInst>(I) ||
      isa<CatchPadInst>(I) || isa<CleanupPadInst>(I) ||
      isa<GetElementPtrInst>(I) ||
      isa<InsertElementInst>(I) ||
      isa<InsertValueInst>(I) ||
      isa<LandingPadInst>(I) ||
      isa<PHINode>(I) ||
      isa<SelectInst>(I) ||
      isa<ShuffleVectorInst>(I) ||
      // Terminators
        isa<BranchInst>(I) ||
        isa<CatchReturnInst>(I) ||
        isa<CatchSwitchInst>(I) ||
        isa<CleanupReturnInst>(I) ||
        isa<ResumeInst>(I) ||
        isa<DetachInst>(I) ||
        isa<ReattachInst>(I) ||
        isa<SyncInst>(I) ||
      // Unary
        isa<AllocaInst>(I) ||
        isa<CastInst>(I) ||
        isa<ExtractValueInst>(I)
      )
    return false;

  return true;
}

// Any reads/writes done in must be done in CurAST
// cannot have any writes/reads, in detached region, respectively
bool llvm::doesDetachedRegionAlias(AliasSetTracker &CurAST, const SmallPtrSetImpl<BasicBlock*>& functionPieces) {
  // If this call only reads from memory and there are no writes to memory
  // above, we can hoist or sink the call as appropriate.
  bool FoundMod = false;
  bool FoundRef = false;
  for (const AliasSet &AS : CurAST) {
    if (!AS.isForwardingAliasSet() && AS.isMod()) {
      FoundMod = true;
      break;
    }
    if (!AS.isForwardingAliasSet() && AS.isRef()) {
      FoundRef = true;
      break;
    }
  }
  if (!FoundMod && !FoundRef)
    return false;

  for(const auto BB : functionPieces) {
    for(const auto& I : *BB) {
      if (doesDetachedInstructionAlias(CurAST, I, FoundMod, FoundRef))
        return true;
    }
  }
  return false;
}

void llvm::moveDetachInstBefore(Instruction* moveBefore, DetachInst& det,
                          const SmallVectorImpl<ReattachInst*>& reattaches,
                          DominatorTree* DT, Value* newSyncRegion) {
  if (newSyncRegion==nullptr) {
    newSyncRegion = det.getSyncRegion();
  }

  auto oldSyncRegion = det.getSyncRegion();
  det.setSyncRegion(newSyncRegion);

  auto oldDetachingParent = det.getParent();
  auto oldContinuation    = det.getContinue();

  auto insertBB = moveBefore->getParent();
  auto newContinuation = insertBB->splitBasicBlock(moveBefore);

  for(auto reattach : reattaches) {
    assert(reattach->getSuccessor(0) == oldContinuation);
    reattach->setSyncRegion(newSyncRegion);
    reattach->setSuccessor(0, newContinuation);

    //there should be no phi's after a reattach
    // assert(oldContinuation->phis().size() == 0);
  }

  if (oldSyncRegion != newSyncRegion) {
    attemptSyncRegionElimination(dyn_cast<Instruction>(oldSyncRegion));
  }

  det.setSuccessor(1, newContinuation);

  det.removeFromParent();
  {
    IRBuilder<> b(oldDetachingParent);
    b.CreateBr(oldContinuation);
  }

  auto term = insertBB->getTerminator();
  det.insertAfter(term);
  term->eraseFromParent();

  if (DT) {
    DT->recalculate(*det.getParent()->getParent());
    DT->verify();
  }
}

bool llvm::attemptSyncRegionElimination(Instruction *SyncRegion) {
  assert(SyncRegion);
  SmallVector<SyncInst*, 4> syncs;
  for (User *U : SyncRegion->users()) {
    if (Instruction *Inst = dyn_cast<Instruction>(U)) {
      if (auto sync = dyn_cast<SyncInst>(Inst)) {
        syncs.push_back(sync);
      } else if (isa<DetachInst>(Inst) || isa<ReattachInst>(Inst)) {
        return false;
      } else {
        llvm_unreachable("Attempting to use sync region elimination on non sync region");
      }
    }
  }
  for(auto sync : syncs) {
    BranchInst* toReplace = BranchInst::Create(sync->getSuccessor(0));
    ReplaceInstWithInst(sync, toReplace);
  }
  SyncRegion->eraseFromParent();
  return true;
}

llvm::LoopSpawningHints::LoopSpawningHints(Loop *L)
    : Strategy(StrategyPrefix(), ST_SEQ, HK_STRATEGY),
      Grainsize(GrainsizePrefix(), 0, HK_GRAINSIZE),
      TheLoop(L) {
  // Populate values with existing loop metadata.
  for(auto& pair: getHintsFromMetadata(TheLoop)) {
    setHint(pair.first, pair.second);
  }
}

LoopSpawningHints::SpawningStrategy
llvm::LoopSpawningHints::getStrategy() const {
  return (SpawningStrategy)Strategy.Value;
}

unsigned llvm::LoopSpawningHints::getGrainsize() const {
  return Grainsize.Value;
}

/// Checks string hint with one operand and set value if valid.
void llvm::LoopSpawningHints::setHint(StringRef Name, Metadata *Arg) {
  if (!Name.startswith(Prefix()))
    return;
  Name = Name.substr(Prefix().size(), StringRef::npos);

  const ConstantInt *C = mdconst::dyn_extract<ConstantInt>(Arg);
  if (!C)
    return;
  unsigned Val = C->getZExtValue();

  Hint *Hints[] = {&Strategy, &Grainsize};
  for (auto H : Hints) {
    if (Name == H->Name) {
      if (H->validate(Val))
        H->Value = Val;
      else
        DEBUG(dbgs() << " ignoring invalid hint '" <<
              Name << "'\n");
      break;
    }
  }
}

/// Create a new hint from name / value pair.
MDNode *llvm::LoopSpawningHints::createHintMetadata(StringRef Name,
                                                    unsigned V) const {
  LLVMContext &Context = TheLoop->getHeader()->getContext();
  Metadata *MDs[] = {MDString::get(Context, Name),
                     ConstantAsMetadata::get(
                         ConstantInt::get(Type::getInt32Ty(Context), V))};
  return MDNode::get(Context, MDs);
}

/// Matches metadata with hint name.
bool llvm::LoopSpawningHints::matchesHintMetadataName(
    MDNode *Node, ArrayRef<Hint> HintTypes) {
  MDString *Name = dyn_cast<MDString>(Node->getOperand(0));
  if (!Name)
    return false;

  for (auto H : HintTypes)
    if (Name->getString().endswith(H.Name))
      return true;
  return false;
}

/// Sets current hints into loop metadata, keeping other values intact.
void llvm::LoopSpawningHints::writeHintsToMetadata(ArrayRef<Hint> HintTypes) {
  if (HintTypes.size() == 0)
    return;

  // Reserve the first element to LoopID (see below).
  SmallVector<Metadata *, 4> MDs(1);
  // If the loop already has metadata, then ignore the existing operands.
  MDNode *LoopID = TheLoop->getLoopID();
  if (LoopID) {
    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
      // If node in update list, ignore old value.
      if (!matchesHintMetadataName(Node, HintTypes))
        MDs.push_back(Node);
    }
  }

  // Now, add the missing hints.
  for (auto H : HintTypes)
    MDs.push_back(createHintMetadata(Twine(Prefix(), H.Name).str(), H.Value));

  // Replace current metadata node with new one.
  LLVMContext &Context = TheLoop->getHeader()->getContext();
  MDNode *NewLoopID = MDNode::get(Context, MDs);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);

  TheLoop->setLoopID(NewLoopID);
}

bool llvm::LoopSpawningHints::Hint::validate(unsigned Val) {
  switch (Kind) {
  case HK_STRATEGY:
    return (Val < ST_END);
  case HK_GRAINSIZE:
    return true;
  }
  return false;
}

STATISTIC(LoopsConvertedToDAC,
          "Number of Tapir loops converted to divide-and-conquer iteration spawning");

/// DACLoopSpawning implements the transformation to spawn the iterations of a
/// Tapir loop in a recursive divide-and-conquer fashion.
class DACLoopSpawning : public LoopOutline {
public:
  TapirTarget* tapirTarget;
  unsigned SpecifiedGrainsize;
  DACLoopSpawning(Loop *OrigLoop, unsigned Grainsize,
                  ScalarEvolution &SE,
                  LoopInfo &LI, DominatorTree &DT,
                  AssumptionCache &AC,
                  OptimizationRemarkEmitter &ORE, TapirTarget* tapirTarget)
      : LoopOutline(OrigLoop, SE, LI, DT, AC, ORE),
        tapirTarget(tapirTarget),
        SpecifiedGrainsize(Grainsize)
  {}

    /// Top-level call to convert loop to spawn its iterations in a
    /// divide-and-conquer fashion.
    bool processLoop() {
      Loop *L = OrigLoop;

      BasicBlock *Header = L->getHeader();
      BasicBlock *Preheader = L->getLoopPreheader();
      BasicBlock *Latch = L->getLoopLatch();

      using namespace ore;

      SmallPtrSet<BasicBlock *, 4> HandledExits;
      if (!getHandledExits(Header, HandledExits))
        return false;

      Module* M = OrigFunction->getParent();

      DEBUG(dbgs() << "LS loop header:" << *Header);
      DEBUG(dbgs() << "LS loop latch:" << *Latch);
      DEBUG(dbgs() << "LS SE exit count: " << *(SE.getExitCount(L, Latch)) << "\n");

      /// Get loop limit.
      const SCEV *Limit = getLimit();
      if (!Limit) return false;

      /// Clean up the loop's induction variable.
      PHINode *CanonicalIV = canonicalizeIVs(Limit->getType());
      if (!CanonicalIV) return false;

      // Remove the IV's (other than CanonicalIV) and replace them with
      // their stronger forms.
      //
      // TODO?: We can probably adapt this loop->DAC process such that we
      // don't require all IV's to be canonical.
      SmallVector<PHINode*, 8> IVs;
      if (!removeNonCanonicalIVs(Header, Preheader, CanonicalIV, IVs))
        return false;

     const SCEVAddRecExpr *CanonicalSCEV =
        cast<const SCEVAddRecExpr>(SE.getSCEV(CanonicalIV));

      // Insert the computation for the loop limit into the Preheader.
      SCEVExpander Exp(SE, M->getDataLayout(), "ls");
      Value *LimitVar = Exp.expandCodeFor(Limit, Limit->getType(),
                                          Preheader->getTerminator());
      DEBUG(dbgs() << "LimitVar: " << *LimitVar << "\n");

      // Canonicalize the loop latch.
      assert(SE.isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT,
                                            CanonicalSCEV, Limit) &&
             "Loop backedge is not guarded by canonical comparison with limit.");
      Value *NewCond = canonicalizeLoopLatch(CanonicalIV, LimitVar);

      // Insert computation of grainsize into the Preheader.
      Value *GrainVar;
      if (!SpecifiedGrainsize)
        GrainVar = computeGrainsize(LimitVar, tapirTarget);
      else
        GrainVar = ConstantInt::get(LimitVar->getType(), SpecifiedGrainsize);

      DEBUG(dbgs() << "GrainVar: " << *GrainVar << "\n");

      /// Clone the loop into a new function.

      // Get the inputs and outputs for the Loop blocks.
      SetVector<Value *> Inputs, Outputs;
      SetVector<Value *> BodyInputs, BodyOutputs;
      ValueToValueMapTy VMap;
      SmallPtrSet<BasicBlock *, 4> ExitsToSplit;
      Value *SRetInput = nullptr;

      // Get the sync region containing this Tapir loop.
      Instruction *InputSyncRegion;
      {
        const DetachInst *DI = cast<DetachInst>(Header->getTerminator());
        InputSyncRegion = cast<Instruction>(DI->getSyncRegion());
      }

      // Add start iteration, end iteration, and grainsize to inputs.

      // Blocks to clone are all those in loop and unreachable / exception-handling exits
      std::vector<BasicBlock *> LoopBlocks(L->getBlocks());
      LoopBlocks.insert(LoopBlocks.end(), HandledExits.begin(), HandledExits.end());

        {
          const DetachInst *DI = cast<DetachInst>(Header->getTerminator());
          BasicBlockEdge DetachEdge(Header, DI->getDetached());
          for (BasicBlock *HE : HandledExits) {
            if (!DT.dominates(DetachEdge, HE))
              ExitsToSplit.insert(HE);
          }
          DEBUG({
              dbgs() << "Loop exits to split:";
              for (BasicBlock *ETS : ExitsToSplit)
                dbgs() << *ETS;
              dbgs() << "\n";
            });
        }

        // Get the inputs and outputs for the loop body.
        findInputsOutputs(LoopBlocks, BodyInputs, BodyOutputs, DT, &ExitsToSplit);
        BodyInputs.remove(InputSyncRegion);

        Value *CanonicalIVInput = CanonicalIV->getIncomingValueForBlock(Preheader);

        // CanonicalIVInput should be the constant 0.
        assert(isa<Constant>(CanonicalIVInput) &&
               "Input to canonical IV from preheader is not constant.");

        // Add explicit argument for loop start, removing from inputs if didn't make new var
        Value* startArg = ensureDistinctArgument(LoopBlocks, CanonicalIVInput, "start");
        BodyInputs.remove(startArg);

        // Add explicit argument for loop end, removing from inputs if didn't make new var
        Value* limitArg = ensureDistinctArgument(LoopBlocks, LimitVar, "end");
        BodyInputs.remove(limitArg);

        // Add explicit argument for grainsize, removing from inputs if didn't make new var
        Value* grainArg = ensureDistinctArgument(LoopBlocks, GrainVar, "grainsize");
        BodyInputs.remove(grainArg);

        // Scan for any sret parameters in BodyInputs and add them first.
        if (OrigFunction->hasStructRetAttr()) {
          Function::arg_iterator ArgIter = OrigFunction->arg_begin();
          if (OrigFunction->hasParamAttribute(0, Attribute::StructRet))
            if (BodyInputs.count(&*ArgIter))
              SRetInput = &*ArgIter;
          if (OrigFunction->hasParamAttribute(1, Attribute::StructRet)) {
            ++ArgIter;
            if (BodyInputs.count(&*ArgIter))
              SRetInput = &*ArgIter;
          }
          if (SRetInput) BodyInputs.remove(SRetInput);
        }

        // Put all of the inputs together
        if (SRetInput) {
          DEBUG(dbgs() << "sret input " << *SRetInput << "\n");
          Inputs.insert(SRetInput);
        }

        Inputs.insert(startArg);
        Inputs.insert(limitArg);
        Inputs.insert(grainArg);
        for (Value *V : BodyInputs) {
            Inputs.insert(V);
            DEBUG({ dbgs() << "Remaining body input: " << *V << "\n"; });
        }

        DEBUG({
            for (Value *V : BodyOutputs)
               dbgs() << "EL output: " << *V << "\n";
        });
        assert(0 == BodyOutputs.size() &&
               "All results from parallel loop should be passed by memory already.");

      DEBUG({
          for (Value *V : Inputs)
            dbgs() << "EL input: " << *V << "\n";
          for (Value *V : Outputs)
            dbgs() << "EL output: " << *V << "\n";
      });

      // Clone the loop blocks into a new helper function.
      Function *Helper;
      {
        SmallVector<ReturnInst *, 0> Returns;  // Ignore returns cloned.
        Helper = CreateHelper(Inputs, Outputs, LoopBlocks,
                              Header, Preheader, ExitBlock,
                              VMap, M,
                              OrigFunction->getSubprogram() != nullptr, Returns, ".ls",
                              &ExitsToSplit, InputSyncRegion,
                              nullptr, nullptr, nullptr);

        assert(Returns.empty() && "Returns cloned when cloning loop.");

        // Use a fast calling convention for the helper.
        Helper->setCallingConv(CallingConv::Fast);
        // Helper->setCallingConv(Header->getParent()->getCallingConv());
      }

      // Add a sync to the helper's return.
      BasicBlock *HelperHeader = cast<BasicBlock>(VMap[Header]);
      {
        BasicBlock *HelperExit = cast<BasicBlock>(VMap[ExitBlock]);
        assert(isa<ReturnInst>(HelperExit->getTerminator()));
        BasicBlock *NewHelperExit = SplitBlock(HelperExit,
                                               HelperExit->getTerminator(),
                                               &DT, &LI);
        IRBuilder<> Builder(&(HelperExit->front()));
        SyncInst *NewSync = Builder.CreateSync(
            NewHelperExit,
            cast<Instruction>(VMap[InputSyncRegion]));
        // Set debug info of new sync to match that of terminator of the header of
        // the cloned loop.
        NewSync->setDebugLoc(HelperHeader->getTerminator()->getDebugLoc());
        HelperExit->getTerminator()->eraseFromParent();
      }

      BasicBlock *NewPreheader = cast<BasicBlock>(VMap[Preheader]);
      PHINode *NewCanonicalIV = cast<PHINode>(VMap[CanonicalIV]);

      // Rewrite the cloned IV's to start at the start iteration argument.
      Argument *NewCanonicalIVStart = cast<Argument>(VMap[startArg]);
      setIVStartingValues(NewCanonicalIVStart, CanonicalIV, IVs, NewPreheader, VMap);

      // The loop has been outlined by this point.  To handle the special cases
      // where the loop limit was constant or used elsewhere within the loop, this
      // pass rewrites the outlined loop-latch condition to use the explicit
      // end-iteration argument.
      if (isa<Constant>(LimitVar) || countUseInRegion(LoopBlocks, LimitVar) != 1) {
        CmpInst *HelperCond = cast<CmpInst>(VMap[NewCond]);
        assert(((isa<Constant>(LimitVar) &&
                 HelperCond->getOperand(1) == LimitVar) ||
                (countUseInRegion(LoopBlocks, LimitVar) != 1 &&
                 HelperCond->getOperand(1) == VMap[LimitVar] )) &&
               "Unexpected condition in loop latch.");
        IRBuilder<> Builder(HelperCond);
        Value *NewHelperCond = Builder.CreateICmpULT(HelperCond->getOperand(0),
                                                     VMap[limitArg]);
        HelperCond->replaceAllUsesWith(NewHelperCond);
        HelperCond->eraseFromParent();
        DEBUG(dbgs() << "Rewritten Latch: " <<
              *(cast<Instruction>(NewHelperCond)->getParent()));
      } else {
        CmpInst *HelperCond = cast<CmpInst>(VMap[NewCond]);
        assert(HelperCond->getOperand(1) == VMap[limitArg]);
      }

      // DEBUGGING: Simply serialize the cloned loop.
      // BasicBlock *NewHeader = cast<BasicBlock>(VMap[Header]);
      // SerializeDetachedCFG(cast<DetachInst>(NewHeader->getTerminator()), nullptr);
      implementDACIterSpawnOnHelper(Helper, NewPreheader,
                                    cast<BasicBlock>(VMap[Header]),
                                    cast<PHINode>(VMap[CanonicalIV]),
                                    cast<Argument>(VMap[limitArg]),
                                    cast<Argument>(VMap[grainArg]),
                                    cast<Instruction>(VMap[InputSyncRegion]),
                                    /*DT=*/nullptr, /*LI=*/nullptr,
                                    CanonicalSCEV->getNoWrapFlags(SCEV::FlagNUW),
                                    CanonicalSCEV->getNoWrapFlags(SCEV::FlagNSW));

      if (verifyFunction(*Helper, &dbgs()))
        return false;

      // Update allocas in cloned loop body.
      {
        // Collect reattach instructions.
        SmallVector<Instruction *, 4> ReattachPoints;
        for (pred_iterator PI = pred_begin(Latch), PE = pred_end(Latch);
             PI != PE; ++PI) {
          BasicBlock *Pred = *PI;
          if (!isa<ReattachInst>(Pred->getTerminator())) continue;
          if (L->contains(Pred))
            ReattachPoints.push_back(cast<BasicBlock>(VMap[Pred])->getTerminator());
        }
        // The cloned loop should be serialized by this point.
        BasicBlock *ClonedLoopBodyEntry =
          cast<BasicBlock>(VMap[Header])->getSingleSuccessor();
        assert(ClonedLoopBodyEntry &&
               "Head of cloned loop body has multiple successors.");
        bool ContainsDynamicAllocas =
          MoveStaticAllocasInBlock(&Helper->getEntryBlock(), ClonedLoopBodyEntry,
                                   ReattachPoints);

        // If the cloned loop contained dynamic alloca instructions, wrap the cloned
        // loop with llvm.stacksave/llvm.stackrestore intrinsics.
        if (ContainsDynamicAllocas) {
          Module *M = Helper->getParent();
          // Get the two intrinsics we care about.
          Function *StackSave = Intrinsic::getDeclaration(M, Intrinsic::stacksave);
          Function *StackRestore =
            Intrinsic::getDeclaration(M,Intrinsic::stackrestore);

          // Insert the llvm.stacksave.
          CallInst *SavedPtr = IRBuilder<>(&*ClonedLoopBodyEntry,
                                           ClonedLoopBodyEntry->begin())
                                 .CreateCall(StackSave, {}, "savedstack");

          // Insert a call to llvm.stackrestore before the reattaches in the
          // original Tapir loop.
          for (Instruction *ExitPoint : ReattachPoints)
            IRBuilder<>(ExitPoint).CreateCall(StackRestore, SavedPtr);
        }
      }

      if (verifyFunction(*Helper, &dbgs()))
        return false;

      // Add alignment assumptions to arguments of helper, based on alignment of
      // values in old function.
      AddAlignmentAssumptions(OrigFunction, Inputs, VMap,
                              Preheader->getTerminator(), &AC, &DT);

      // Add call to new helper function in original function.
      {
        // Setup arguments for call.
        SmallVector<Value *, 4> TopCallArgs;
        // Add sret input, if it exists.
        if (SRetInput)
          TopCallArgs.push_back(SRetInput);
        // Add start iteration 0.
        assert(CanonicalSCEV->getStart()->isZero() &&
               "Canonical IV does not start at zero.");
        TopCallArgs.push_back(ConstantInt::get(CanonicalIV->getType(), 0));
        // Add loop limit.
        TopCallArgs.push_back(LimitVar);
        // Add grainsize.
        TopCallArgs.push_back(GrainVar);
        // Add the rest of the arguments.
        for (Value *V : BodyInputs) {
          TopCallArgs.push_back(V);
        }
        DEBUG({
            for (Value *TCArg : TopCallArgs)
              dbgs() << "Top call arg: " << *TCArg << "\n";
          });

        // Create call instruction.
        IRBuilder<> Builder(Preheader->getTerminator());
        CallInst *TopCall = Builder.CreateCall(Helper,
                                               ArrayRef<Value *>(TopCallArgs));

        // Use a fast calling convention for the helper.
        TopCall->setCallingConv(CallingConv::Fast);
        TopCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
        // // Update CG graph with the call we just added.
        // CG[F]->addCalledFunction(TopCall, CG[Helper]);
      }

      // Remove sync of loop in parent.
      {
        // Get the sync region for this loop's detached iterations.
        DetachInst *HeadDetach = cast<DetachInst>(Header->getTerminator());
        Value *SyncRegion = HeadDetach->getSyncRegion();
        // Check the Tapir instructions contained in this sync region.  Look for a
        // single sync instruction among those Tapir instructions.  Meanwhile,
        // verify that the only detach instruction in this sync region is the detach
        // in theloop header.  If these conditions are met, then we assume that the
        // sync applies to this loop.  Otherwise, something more complicated is
        // going on, and we give up.
        SyncInst *LoopSync = nullptr;
        bool SingleSyncJustForLoop = true;
        for (User *U : SyncRegion->users()) {
          // Skip the detach in the loop header.
          if (HeadDetach == U) continue;
          // Remember the first sync instruction we find.  If we find multiple sync
          // instructions, then something nontrivial is going on.
          if (SyncInst *SI = dyn_cast<SyncInst>(U)) {
            if (!LoopSync)
              LoopSync = SI;
            else
              SingleSyncJustForLoop = false;
          }
          // If we find a detach instruction that is not the loop header's, then
          // something nontrivial is going on.
          if (isa<DetachInst>(U))
            SingleSyncJustForLoop = false;
        }
        if (LoopSync && SingleSyncJustForLoop)
          // Replace the sync with a branch.
          ReplaceInstWithInst(LoopSync,
                              BranchInst::Create(LoopSync->getSuccessor(0)));
        else if (!LoopSync)
          DEBUG(dbgs() << "No sync found for this loop.");
        else
          DEBUG(dbgs() << "No single sync found that only affects this loop.");
      }

      ++LoopsConvertedToDAC;

      unlinkLoop();

      return Helper;
    }

  virtual ~DACLoopSpawning() {}

protected:


/// \brief Method to help convertLoopToDACIterSpawn convert the Tapir
/// loop cloned into function Helper to spawn its iterations in a
/// parallel divide-and-conquer fashion.
///
/// Example: Suppose that Helper contains the following Tapir loop:
///
/// Helper(iter_t start, iter_t end, iter_t grain, ...) {
///   iter_t i = start;
///   ... Other loop setup ...
///   do {
///     spawn { ... loop body ... };
///   } while (i++ < end);
///   sync;
/// }
///
/// Then this method transforms Helper into the following form:
///
/// Helper(iter_t start, iter_t end, iter_t grain, ...) {
/// recur:
///   iter_t itercount = end - start;
///   if (itercount > grain) {
///     // Invariant: itercount >= 2
///     count_t miditer = start + itercount / 2;
///     spawn Helper(start, miditer, grain, ...);
///     start = miditer + 1;
///     goto recur;
///   }
///
///   iter_t i = start;
///   ... Other loop setup ...
///   do {
///     ... Loop Body ...
///   } while (i++ < end);
///   sync;
/// }
///
void implementDACIterSpawnOnHelper(Function *Helper,
                                                    BasicBlock *Preheader,
                                                    BasicBlock *Header,
                                                    PHINode *CanonicalIV,
                                                    Argument *Limit,
                                                    Argument *Grainsize,
                                                    Instruction *SyncRegion,
                                                    DominatorTree *DT,
                                                    LoopInfo *LI,
                                                    bool CanonicalIVFlagNUW = false,
                                                    bool CanonicalIVFlagNSW = false) {
  // Serialize the cloned copy of the loop.
  assert(Preheader->getParent() == Helper &&
         "Preheader does not belong to helper function.");
  assert(Header->getParent() == Helper &&
         "Header does not belong to helper function.");
  assert(CanonicalIV->getParent() == Header &&
         "CanonicalIV does not belong to header");
  assert(isa<DetachInst>(Header->getTerminator()) &&
         "Cloned header is not terminated by a detach.");
  DetachInst *DI = dyn_cast<DetachInst>(Header->getTerminator());
  SerializeDetachedCFG(DI, DT);

  // Convert the cloned loop into the strip-mined loop body.

  BasicBlock *DACHead = Preheader;
  if (&(Helper->getEntryBlock()) == Preheader)
    // Split the entry block.  We'll want to create a backedge into
    // the split block later.
    DACHead = SplitBlock(Preheader, Preheader->getTerminator(), DT, LI);

  BasicBlock *RecurHead, *RecurDet, *RecurCont;
  Value *IterCount;
  Value *CanonicalIVInput;
  PHINode *CanonicalIVStart;
  {
    Instruction *PreheaderOrigFront = &(DACHead->front());
    IRBuilder<> Builder(PreheaderOrigFront);
    // Create branch based on grainsize.
    DEBUG(dbgs() << "LS CanonicalIV: " << *CanonicalIV << "\n");
    CanonicalIVInput = CanonicalIV->getIncomingValueForBlock(DACHead);
    CanonicalIVStart = Builder.CreatePHI(CanonicalIV->getType(), 2,
                                         CanonicalIV->getName()+".dac");
    CanonicalIVInput->replaceAllUsesWith(CanonicalIVStart);
    IterCount = Builder.CreateSub(Limit, CanonicalIVStart,
                                  "itercount");
    Value *IterCountCmp = Builder.CreateICmpUGT(IterCount, Grainsize);
    TerminatorInst *RecurTerm =
      SplitBlockAndInsertIfThen(IterCountCmp, PreheaderOrigFront,
                                /*Unreachable=*/false,
                                /*BranchWeights=*/nullptr,
                                DT);
    RecurHead = RecurTerm->getParent();
    // Create skeleton of divide-and-conquer recursion:
    // DACHead -> RecurHead -> RecurDet -> RecurCont -> DACHead
    RecurDet = SplitBlock(RecurHead, RecurHead->getTerminator(),
                          DT, LI);
    RecurCont = SplitBlock(RecurDet, RecurDet->getTerminator(),
                           DT, LI);
    RecurCont->getTerminator()->replaceUsesOfWith(RecurTerm->getSuccessor(0),
                                                  DACHead);
  }

  // Compute mid iteration in RecurHead.
  Value *MidIter, *MidIterPlusOne;
  {
    IRBuilder<> Builder(&(RecurHead->front()));
    MidIter = Builder.CreateAdd(CanonicalIVStart,
                                Builder.CreateLShr(IterCount, 1,
                                                   "halfcount"),
                                "miditer",
                                CanonicalIVFlagNUW, CanonicalIVFlagNSW);
  }

  // Create recursive call in RecurDet.
  {
    // Create input array for recursive call.
    IRBuilder<> Builder(&(RecurDet->front()));
    SetVector<Value*> RecurInputs;
    Function::arg_iterator AI = Helper->arg_begin();
    // Handle an initial sret argument, if necessary.  Based on how
    // the Helper function is created, any sret parameter will be the
    // first parameter.
    if (Helper->hasParamAttribute(0, Attribute::StructRet))
      RecurInputs.insert(&*AI++);
    assert(cast<Argument>(CanonicalIVInput) == &*AI &&
           "First non-sret argument does not match original input to canonical IV.");
    RecurInputs.insert(CanonicalIVStart);
    ++AI;
    assert(Limit == &*AI &&
           "Second non-sret argument does not match original input to the loop limit.");
    RecurInputs.insert(MidIter);
    ++AI;
    for (Function::arg_iterator AE = Helper->arg_end();
         AI != AE;  ++AI)
        RecurInputs.insert(&*AI);
    DEBUG({
        dbgs() << "RecurInputs: ";
        for (Value *Input : RecurInputs)
          dbgs() << *Input << ", ";
        dbgs() << "\n";
      });

    // Create call instruction.
    CallInst *RecurCall = Builder.CreateCall(Helper, RecurInputs.getArrayRef());
    RecurCall->setDebugLoc(Header->getTerminator()->getDebugLoc());
    // Use a fast calling convention for the helper.
    RecurCall->setCallingConv(CallingConv::Fast);
    // RecurCall->setCallingConv(Helper->getCallingConv());
    // // Update CG graph with the recursive call we just added.
    // CG[Helper]->addCalledFunction(RecurCall, CG[Helper]);
  }

  // Set up continuation of detached recursive call.  We effectively
  // inline this tail call automatically.
  {
    IRBuilder<> Builder(&(RecurCont->front()));
    MidIterPlusOne = Builder.CreateAdd(MidIter,
                                       ConstantInt::get(Limit->getType(), 1),
                                       "miditerplusone",
                                       CanonicalIVFlagNUW,
                                       CanonicalIVFlagNSW);
  }

  // Finish setup of new phi node for canonical IV.
  {
    CanonicalIVStart->addIncoming(CanonicalIVInput, Preheader);
    CanonicalIVStart->addIncoming(MidIterPlusOne, RecurCont);
  }

  /// Make the recursive DAC parallel.
  {
    IRBuilder<> Builder(RecurHead->getTerminator());
    // Create the detach.
    DetachInst *DI = Builder.CreateDetach(RecurDet, RecurCont, SyncRegion);
    DI->setDebugLoc(Header->getTerminator()->getDebugLoc());
    RecurHead->getTerminator()->eraseFromParent();
    // Create the reattach.
    Builder.SetInsertPoint(RecurDet->getTerminator());
    ReattachInst *RI = Builder.CreateReattach(RecurCont, SyncRegion);
    RI->setDebugLoc(Header->getTerminator()->getDebugLoc());
    RecurDet->getTerminator()->eraseFromParent();
  }
}

};

bool llvm::TapirTarget::processDACLoop(LoopSpawningHints LSH, LoopInfo &LI, ScalarEvolution &SE, DominatorTree &DT,
                                       AssumptionCache &AC, OptimizationRemarkEmitter &ORE) {

    DEBUG(dbgs() << "LS: Hints dictate DAC spawning.\n");

    Loop* L = LSH.TheLoop;

    DebugLoc DLoc = L->getStartLoc();
    BasicBlock *Header = L->getHeader();
    DACLoopSpawning DLS(L, LSH.getGrainsize(), SE, LI, DT, AC, ORE, this);
      if (DLS.processLoop()) {
        DEBUG({
            if (verifyFunction(*L->getHeader()->getParent())) {
              dbgs() << "Transformed function is invalid.\n";
              return false;
            }
          });
        // Report success.
        ORE.emit(OptimizationRemark(LS_NAME, "DACSpawning", DLoc, Header)
                 << "spawning iterations using divide-and-conquer");
        return true;
      } else {
        // Report failure.
        ORE.emit(OptimizationRemarkMissed(LS_NAME, "NoDACSpawning", DLoc,
                                          Header)
                 << "cannot spawn iterations using divide-and-conquer");
        ORE.emit(DiagnosticInfoOptimizationFailure(
              DEBUG_TYPE, "FailedRequestedSpawning",
              L->getStartLoc(), L->getHeader())
          << "Tapir loop not transformed: "
          << "failed to use divide-and-conquer loop spawning");
        return false;
      }

  return false;
}
