//===- TapirOutline.cpp - Outlining for Tapir -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements helper functions for outlining portions of code
// containing Tapir instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "outlining"

/// definedInRegion - Return true if the specified value is defined in the
/// extracted region.
static bool definedInRegion(const SmallPtrSetImpl<BasicBlock *> &Blocks,
                            Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (Blocks.count(I->getParent()))
      return true;
  return false;
}

/// definedInCaller - Return true if the specified value is defined in the
/// function being code extracted, but not in the region being extracted.
/// These values must be passed in as live-ins to the function.
static bool definedInCaller(const SmallPtrSetImpl<BasicBlock *> &Blocks,
                            Value *V) {
  if (isa<Argument>(V)) return true;
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (!Blocks.count(I->getParent()))
      return true;
  return false;
}

void llvm::findInputsOutputs(const SmallPtrSetImpl<BasicBlock *> &Blocks,
                             ValueSet &Inputs,
                             ValueSet &Outputs,
                             const SmallPtrSetImpl<BasicBlock *> *ExitBlocks) {
  for (BasicBlock *BB : Blocks) {
    // If a used value is defined outside the region, it's an input.  If an
    // instruction is used outside the region, it's an output.
    for (Instruction &II : *BB) {
      for (User::op_iterator OI = II.op_begin(), OE = II.op_end(); OI != OE;
           ++OI) {
        // The PHI nodes each exit block will be updated after the exit block is
        // cloned, so we don't want to count their uses of values defined
        // outside the region.
        if (ExitBlocks->count(BB))
          if (PHINode *PN = dyn_cast<PHINode>(&II))
            if (!Blocks.count(PN->getIncomingBlock(*OI)))
              continue;
        if (definedInCaller(Blocks, *OI))
          Inputs.insert(*OI);
      }

      for (User *U : II.users())
        if (!definedInRegion(Blocks, U)) {
          Outputs.insert(&II);
          break;
        }
    }
  }
}

// Clone Blocks into NewFunc, transforming the old arguments into references to
// VMap values.
//
/// TODO: Fix the std::vector part of the type of this function.
void llvm::CloneIntoFunction(Function *NewFunc, const Function *OldFunc,
                             std::vector<BasicBlock *> Blocks,
                             ValueToValueMapTy &VMap,
                             bool ModuleLevelChanges,
                             SmallVectorImpl<ReturnInst *> &Returns,
                             const StringRef NameSuffix,
                             SmallPtrSetImpl<BasicBlock *> *ExitBlocks,
                             ClonedCodeInfo *CodeInfo,
                             ValueMapTypeRemapper *TypeMapper,
                             ValueMaterializer *Materializer) {
  // Get the predecessors of the exit blocks
  SmallPtrSet<const BasicBlock *, 4> ExitBlockPreds, ClonedEBPreds;
  for (BasicBlock *EB : *ExitBlocks)
    for (BasicBlock *Pred : predecessors(EB))
      ExitBlockPreds.insert(Pred);

  // Loop over all of the basic blocks in the function, cloning them as
  // appropriate.
  for (const BasicBlock *BB : Blocks) {
    // Record all exit block predecessors that are cloned.
    if (ExitBlockPreds.count(BB))
      ClonedEBPreds.insert(BB);

    // Create a new basic block and copy instructions into it!
    BasicBlock *CBB = CloneBasicBlock(BB, VMap, NameSuffix, NewFunc, CodeInfo);

    // Add basic block mapping.
    VMap[BB] = CBB;

    // It is only legal to clone a function if a block address within that
    // function is never referenced outside of the function.  Given that, we
    // want to map block addresses from the old function to block addresses in
    // the clone. (This is different from the generic ValueMapper
    // implementation, which generates an invalid blockaddress when
    // cloning a function.)
    if (BB->hasAddressTaken()) {
      Constant *OldBBAddr = BlockAddress::get(const_cast<Function*>(OldFunc),
                                              const_cast<BasicBlock*>(BB));
      VMap[OldBBAddr] = BlockAddress::get(NewFunc, CBB);
    }

    // Note return instructions for the caller.
    if (ReturnInst *RI = dyn_cast<ReturnInst>(CBB->getTerminator()))
      Returns.push_back(RI);
  }

  // For each exit block, clean up its phi nodes to exclude predecessors that
  // were not cloned.
  if (ExitBlocks) {
    for (BasicBlock *EB : *ExitBlocks) {
      // dbgs() << "Exit block:" << *EB;
      // Get the predecessors of this exit block that were not cloned.
      SmallVector<BasicBlock *, 4> PredNotCloned;
      for (BasicBlock *Pred : predecessors(EB)) {
        if (!ClonedEBPreds.count(Pred)) {
          // dbgs() << "\tpred not cloned: " << Pred->getName() << "\n";
          PredNotCloned.push_back(Pred);
        // } else {
        //   dbgs() << "Pred " << Pred->getName() << " maps to " << VMap[Pred]->getName() << "\n";
        }
      }
      // Iterate over the phi nodes in the cloned exit block and remove incoming
      // values from predecessors that were not cloned.
      BasicBlock *ClonedEB = cast<BasicBlock>(VMap[EB]);
      BasicBlock::iterator BI = ClonedEB->begin();
      while (PHINode *PN = dyn_cast<PHINode>(BI)) {
        for (BasicBlock *DeadPred : PredNotCloned)
          if (PN->getBasicBlockIndex(DeadPred) > -1)
            PN->removeIncomingValue(DeadPred);
        ++BI;
      }
    }
  }

  // Loop over all of the instructions in the function, fixing up operand
  // references as we go.  This uses VMap to do all the hard work.
  for (const BasicBlock *BB : Blocks) {
    BasicBlock *CBB = cast<BasicBlock>(VMap[BB]);
    // Loop over all instructions, fixing each one as we find it...
    for (Instruction &II : *CBB) {
      RemapInstruction(&II, VMap,
                       ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges,
                       TypeMapper, Materializer);
      
    }
  }
}

/// Create a helper function whose signature is based on Inputs and
/// Outputs as follows: f(in0, ..., inN, out0, ..., outN)
///
/// TODO: Fix the std::vector part of the type of this function.
Function *llvm::CreateHelper(const ValueSet &Inputs,
                             const ValueSet &Outputs,
                             std::vector<BasicBlock *> Blocks,
                             BasicBlock *Header,
                             const BasicBlock *OldEntry,
                             const BasicBlock *OldExit,
                             ValueToValueMapTy &VMap,
                             Module *DestM,
                             bool ModuleLevelChanges,
                             SmallVectorImpl<ReturnInst *> &Returns,
                             const StringRef NameSuffix,
                             SmallPtrSetImpl<BasicBlock *> *ExitBlocks,
                             ClonedCodeInfo *CodeInfo,
                             ValueMapTypeRemapper *TypeMapper,
                             ValueMaterializer *Materializer) {
  DEBUG(dbgs() << "inputs: " << Inputs.size() << "\n");
  DEBUG(dbgs() << "outputs: " << Outputs.size() << "\n");

  Function *OldFunc = Header->getParent();
  Type *RetTy = Type::getVoidTy(Header->getContext());

  std::vector<Type*> paramTy;

  // Add the types of the input values to the function's argument list
  for (Value *value : Inputs) {
    DEBUG(dbgs() << "value used in func: " << *value << "\n");
    paramTy.push_back(value->getType());
  }

  // Add the types of the output values to the function's argument list.
  for (Value *output : Outputs) {
    DEBUG(dbgs() << "instr used in func: " << *output << "\n");
    paramTy.push_back(PointerType::getUnqual(output->getType()));
  }

  DEBUG({
      dbgs() << "Function type: " << *RetTy << " f(";
      for (Type *i : paramTy)
	dbgs() << *i << ", ";
      dbgs() << ")\n";
    });

  FunctionType *FTy = FunctionType::get(RetTy, paramTy, false);

  // Create the new function
  Function *NewFunc = Function::Create(FTy,
				       GlobalValue::InternalLinkage,
				       OldFunc->getName() + "_" +
				       Header->getName(), DestM);
  
  // Set names for input and output arguments.
  Function::arg_iterator DestI = NewFunc->arg_begin();
  for (Value *I : Inputs)
    if (VMap.count(I) == 0) {       // Is this argument preserved?
      DestI->setName(I->getName()+NameSuffix); // Copy the name over...
      VMap[I] = &*DestI++;          // Add mapping to VMap
    }
  for (Value *I : Outputs)
    if (VMap.count(I) == 0) {              // Is this argument preserved?
      DestI->setName(I->getName()+NameSuffix); // Copy the name over...
      VMap[I] = &*DestI++;                 // Add mapping to VMap
    }
  
  // Copy all attributes other than those stored in the AttributeSet.  We need
  // to remap the parameter indices of the AttributeSet.
  AttributeSet NewAttrs = NewFunc->getAttributes();
  NewFunc->copyAttributesFrom(OldFunc);
  NewFunc->setAttributes(NewAttrs);

  // Fix up the personality function that got copied over.
  if (OldFunc->hasPersonalityFn())
    NewFunc->setPersonalityFn(
        MapValue(OldFunc->getPersonalityFn(), VMap,
                 ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges,
                 TypeMapper, Materializer));

  AttributeSet OldAttrs = OldFunc->getAttributes();
  // Clone any argument attributes
  for (Argument &OldArg : OldFunc->args()) {
    // Check if we're passing this argument to the helper.  We check Inputs here
    // instead of the VMap to avoid potentially populating the VMap with a null
    // entry for the old argument.
    if (Inputs.count(&OldArg) || Outputs.count(&OldArg)) {
      Argument *NewArg = dyn_cast_or_null<Argument>(VMap[&OldArg]);
      AttributeSet attrs =
        OldAttrs.getParamAttributes(OldArg.getArgNo() + 1);
      if (attrs.getNumSlots() > 0)
        NewArg->addAttr(attrs);
    }
  }
  NewFunc->setAttributes(
      NewFunc->getAttributes()
      // Ignore the return attributes of the old function.
      // .addAttributes(NewFunc->getContext(), AttributeSet::ReturnIndex,
      //                OldAttrs.getRetAttributes())
      .addAttributes(NewFunc->getContext(), AttributeSet::FunctionIndex,
                     OldAttrs.getFnAttributes()));

  SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
  OldFunc->getAllMetadata(MDs);
  for (auto MD : MDs) {
    NewFunc->addMetadata(
        MD.first,
        *MapMetadata(MD.second, VMap,
                     ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges,
                     TypeMapper, Materializer));
  }
  // We assume that the Helper reads and writes its arguments.  If the parent
  // function had stronger attributes on memory access -- specifically, if the
  // parent is marked as only reading memory -- we must replace this attribute
  // with an appropriate weaker form.
  if (OldFunc->onlyReadsMemory()) {
    NewFunc->removeFnAttr(Attribute::ReadNone);
    NewFunc->removeFnAttr(Attribute::ReadOnly);
    NewFunc->setOnlyAccessesArgMemory();
  }

  // Inherit the calling convention from the parent.
  NewFunc->setCallingConv(OldFunc->getCallingConv());

  // The new function needs a root node because other nodes can branch to the
  // head of the region, but the entry node of a function cannot have preds.
  BasicBlock *NewEntry = BasicBlock::Create(Header->getContext(),
					    OldEntry->getName()+NameSuffix,
                                            NewFunc);
  // The new function also needs an exit node.
  BasicBlock *NewExit = BasicBlock::Create(Header->getContext(),
					   OldExit->getName()+NameSuffix,
                                           NewFunc);

  // Add mappings to the NewEntry and NewExit.
  VMap[OldEntry] = NewEntry;
  VMap[OldExit] = NewExit;

  // Clone Blocks into the new function.
  CloneIntoFunction(NewFunc, OldFunc, Blocks, VMap, ModuleLevelChanges,
		    Returns, NameSuffix, ExitBlocks, CodeInfo, TypeMapper,
                    Materializer);

  // Add a branch in the new function to the cloned Header.
  BranchInst::Create(cast<BasicBlock>(VMap[Header]), NewEntry);
  // Add a return in the new function.
  ReturnInst::Create(Header->getContext(), NewExit);
  
  return NewFunc;
}

/// Return the result of AI->isStaticAlloca() if AI were moved to the entry
/// block. Allocas used in inalloca calls and allocas of dynamic array size
/// cannot be static.
/// (Borrowed from Transforms/Utils/InlineFunction.cpp)
static bool allocaWouldBeStaticInEntry(const AllocaInst *AI) {
  return isa<Constant>(AI->getArraySize()) && !AI->isUsedWithInAlloca();
}

// Check whether this Value is used by a lifetime intrinsic.
static bool isUsedByLifetimeMarker(Value *V) {
  for (User *U : V->users()) {
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(U)) {
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
        return true;
      }
    }
  }
  return false;
}

// Check whether the given alloca already has
// lifetime.start or lifetime.end intrinsics.
static bool hasLifetimeMarkers(AllocaInst *AI) {
  Type *Ty = AI->getType();
  Type *Int8PtrTy = Type::getInt8PtrTy(Ty->getContext(),
                                       Ty->getPointerAddressSpace());
  if (Ty == Int8PtrTy)
    return isUsedByLifetimeMarker(AI);

  // Do a scan to find all the casts to i8*.
  for (User *U : AI->users()) {
    if (U->getType() != Int8PtrTy) continue;
    if (U->stripPointerCasts() != AI) continue;
    if (isUsedByLifetimeMarker(U))
      return true;
  }
  return false;
}

// Move static allocas in a cloned block into the entry block of helper.  Leave
// lifetime markers behind for those static allocas.  Returns true if the cloned
// block still contains dynamic allocas, which cannot be moved.
bool llvm::MoveStaticAllocasInClonedBlock(
    Function *Helper,
    BasicBlock *ClonedBlock,
    SmallVectorImpl<Instruction *> &ClonedExitPoints) {
  SmallVector<AllocaInst *, 4> StaticAllocas;
  bool ContainsDynamicAllocas = false;
  BasicBlock::iterator InsertPoint = Helper->begin()->begin();
  for (BasicBlock::iterator I = ClonedBlock->begin(),
         E = ClonedBlock->end(); I != E; ) {
    AllocaInst *AI = dyn_cast<AllocaInst>(I++);
    if (!AI) continue;

    if (!allocaWouldBeStaticInEntry(AI)) {
      ContainsDynamicAllocas = true;
      continue;
    }

    StaticAllocas.push_back(AI);

    // Scan for the block of allocas that we can move over, and move them
    // all at once.
    while (isa<AllocaInst>(I) &&
           allocaWouldBeStaticInEntry(cast<AllocaInst>(I))) {
      StaticAllocas.push_back(cast<AllocaInst>(I));
      ++I;
    }

    // Transfer all of the allocas over in a block.  Using splice means
    // that the instructions aren't removed from the symbol table, then
    // reinserted.
    Helper->getEntryBlock().getInstList().splice(
        InsertPoint, ClonedBlock->getInstList(), AI->getIterator(), I);
  }
  // Move any dbg.declares describing the allocas into the entry basic block.
  DIBuilder DIB(*Helper->getParent());
  for (auto &AI : StaticAllocas)
    replaceDbgDeclareForAlloca(AI, AI, DIB, /*Deref=*/false);

  // Leave lifetime markers for the static alloca's, scoping them to the
  // from cloned block to cloned exit.
  if (!StaticAllocas.empty()) {
    IRBuilder<> Builder(&ClonedBlock->front());
    for (unsigned ai = 0, ae = StaticAllocas.size(); ai != ae; ++ai) {
      AllocaInst *AI = StaticAllocas[ai];
      // Don't mark swifterror allocas. They can't have bitcast uses.
      if (AI->isSwiftError())
        continue;

      // If the alloca is already scoped to something smaller than the whole
      // function then there's no need to add redundant, less accurate markers.
      if (hasLifetimeMarkers(AI))
        continue;

      // Try to determine the size of the allocation.
      ConstantInt *AllocaSize = nullptr;
      if (ConstantInt *AIArraySize =
          dyn_cast<ConstantInt>(AI->getArraySize())) {
        auto &DL = Helper->getParent()->getDataLayout();
        Type *AllocaType = AI->getAllocatedType();
        uint64_t AllocaTypeSize = DL.getTypeAllocSize(AllocaType);
        uint64_t AllocaArraySize = AIArraySize->getLimitedValue();

        // Don't add markers for zero-sized allocas.
        if (AllocaArraySize == 0)
          continue;

        // Check that array size doesn't saturate uint64_t and doesn't
        // overflow when it's multiplied by type size.
        if (AllocaArraySize != ~0ULL &&
            UINT64_MAX / AllocaArraySize >= AllocaTypeSize) {
          AllocaSize = ConstantInt::get(Type::getInt64Ty(AI->getContext()),
                                        AllocaArraySize * AllocaTypeSize);
        }
      }

      Builder.CreateLifetimeStart(AI, AllocaSize);
      for (Instruction *ExitPoint : ClonedExitPoints) {
        IRBuilder<>(ExitPoint).CreateLifetimeEnd(AI, AllocaSize);
      }
    }
  }

  return ContainsDynamicAllocas;
}

// Add alignment assumptions to parameters of outlined function, based on known
// alignment data in the caller.
void llvm::AddAlignmentAssumptions(const Function *Caller,
                                   const ValueSet &Inputs,
                                   ValueToValueMapTy &VMap,
                                   const Instruction *CallSite,
                                   AssumptionCache *AC,
                                   DominatorTree *DT) {
  auto &DL = Caller->getParent()->getDataLayout();
  for (Value *ArgVal : Inputs) {
    // Ignore arguments to non-pointer types
    if (!ArgVal->getType()->isPointerTy()) continue;
    Argument *Arg = cast<Argument>(VMap[ArgVal]);
    // Ignore arguments to non-pointer types
    if (!Arg->getType()->isPointerTy()) continue;
    // If the argument already has an alignment attribute, skip it.
    if (Arg->getParamAlignment()) continue;
    // Get any known alignment information for this argument's value.
    unsigned Align = getKnownAlignment(ArgVal, DL, CallSite, AC, DT);
    // If we have alignment data, add it as an attribute to the outlined
    // function's parameter.
    if (Align) {
      AttrBuilder B;
      B.addAlignmentAttr(Align);
      Arg->addAttr(AttributeSet::get(Arg->getContext(), Arg->getArgNo() + 1,
                                     B));
    }
  }
}
