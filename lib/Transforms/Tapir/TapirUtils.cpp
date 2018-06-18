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

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Transforms/Tapir/CilkABI.h"
#include "llvm/Transforms/Tapir/OpenMPABI.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "tapir"

TapirTarget *llvm::getTapirTargetFromType(TapirTargetType Type) {
  switch(Type) {
  case TapirTargetType::Cilk:
    return new CilkABI();
  case TapirTargetType::OpenMP:
    return new OpenMPABI();
  case TapirTargetType::None:
  case TapirTargetType::Serial:
  default:
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
  findInputsOutputs(functionPieces, BodyInputs, Outputs, &ExitBlocks, &DT);
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
        allowsyncregion && (id == Intrinsic::syncregion_start));
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
        allowsyncregion && (id == Intrinsic::syncregion_start));
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
