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
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/Cloning.h"
using namespace llvm;

#define DEBUG_TYPE "outlining"

// Clone Blocks into NewFunc, transforming the old arguments into references to
// VMap values.
//
/// TODO: Fix the std::vector part of the type of this function.
void llvm::CloneIntoFunction(Function *NewFunc, const Function *OldFunc,
                             std::vector<BasicBlock *> Blocks,
                             ValueToValueMapTy &VMap,
                             bool ModuleLevelChanges,
                             SmallVectorImpl<ReturnInst *> &Returns,
                             const char *NameSuffix,
                             ClonedCodeInfo *CodeInfo,
                             ValueMapTypeRemapper *TypeMapper,
                             ValueMaterializer *Materializer) {
  assert(NameSuffix && "NameSuffix cannot be null!");

  // Loop over all of the basic blocks in the function, cloning them as
  // appropriate.
  //
  for (const BasicBlock *BB : Blocks) {
    // dbgs() << "Cloning basic block " << BB->getName() << "\n";
    // Create a new basic block and copy instructions into it!
    BasicBlock *CBB = CloneBasicBlock(BB, VMap, NameSuffix, NewFunc, CodeInfo);

    // Add basic block mapping.
    // dbgs() << "Mapping " << BB->getName() << " to " << CBB->getName() << "\n";
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
Function *llvm::CreateHelper(const SetVector<Value *> &Inputs,
                             const SetVector<Value *> &Outputs,
                             std::vector<BasicBlock *> Blocks,
                             BasicBlock *Header,
                             const BasicBlock *OldEntry,
                             const BasicBlock *OldExit,
                             ValueToValueMapTy &VMap,
                             Module *DestM,
                             bool ModuleLevelChanges,
                             SmallVectorImpl<ReturnInst *> &Returns,
                             const char *NameSuffix,
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

  // // If the old function is no-throw, so is the new one.
  // if (OldFunc->doesNotThrow())
  //   NewFunc->setDoesNotThrow();

  // // Inherit the uwtable attribute if we need to.
  // if (OldFunc->hasUWTable())
  //   NewFunc->setHasUWTable();
  
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

  // // Inherit the calling convention from the parent.
  // NewFunc->setCallingConv(OldFunc->getCallingConv());

  // The new function needs a root node because other nodes can branch to the
  // head of the region, but the entry node of a function cannot have preds.
  BasicBlock *NewEntry = BasicBlock::Create(Header->getContext(),
					    OldEntry->getName()+NameSuffix);
  NewFunc->getBasicBlockList().push_back(NewEntry);
  
  BasicBlock *NewExit = BasicBlock::Create(Header->getContext(),
					   OldExit->getName()+NameSuffix);
  NewFunc->getBasicBlockList().push_back(NewExit);

  // Add a mapping to the NewFuncRoot.
  VMap[OldEntry] = NewEntry;
  VMap[OldExit] = NewExit;

  // Clone Blocks into the new function.
  CloneIntoFunction(NewFunc, OldFunc, Blocks, VMap, ModuleLevelChanges,
		    Returns, NameSuffix, CodeInfo, TypeMapper, Materializer);

  // Add a branch in the new function to the cloned Header.
  NewEntry->getInstList().push_back(BranchInst::Create(cast<BasicBlock>(VMap[Header])));
  NewExit->getInstList().push_back(ReturnInst::Create(Header->getContext()));
  
  return NewFunc;
}
