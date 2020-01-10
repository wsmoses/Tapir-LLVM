//===- CilkRTSCilkFor.cpp - Interface to __cilkrts_cilk_for ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a loop-outline processor to lower Tapir loops to a call
// to a Cilk runtime method, __cilkrts_cilk_for.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/CilkRTSCilkFor.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "cilkrtscilkfor"

cl::opt<bool> llvm::UseRuntimeCilkFor(
    "cilk-use-runtime-cilkfor", cl::init(false), cl::Hidden,
    cl::desc("Insert a call into the Cilk runtime to handle cilk_for loops"));

using cilk32_t = uint32_t;
using cilk64_t = uint64_t;
using __cilk_abi_f32_t = void (*)(void *data, cilk32_t low, cilk32_t high);
using __cilk_abi_f64_t = void (*)(void *data, cilk64_t low, cilk64_t high);

using __cilkrts_cilk_for_32 = void (__cilk_abi_f32_t body, void *data,
                                    cilk32_t count, int grain);
using __cilkrts_cilk_for_64 = void (__cilk_abi_f64_t body, void *data,
                                    cilk64_t count, int grain);

#define CILKRTS_FUNC(name, CGF) Get__cilkrts_##name(CGF)

#define DEFAULT_GET_CILKRTS_FUNC(name)                                  \
  static Function *Get__cilkrts_##name(Module& M) {                     \
    return cast<Function>(M.getOrInsertFunction(                        \
                              "__cilkrts_"#name,                        \
                              TypeBuilder<__cilkrts_##name, false>::get(M.getContext()) \
                                                                        )); \
  }

DEFAULT_GET_CILKRTS_FUNC(cilk_for_32)
DEFAULT_GET_CILKRTS_FUNC(cilk_for_64)

void RuntimeCilkFor::setupLoopOutlineArgs(
    Function &F, ValueSet &HelperArgs, SmallVectorImpl<Value *> &HelperInputs,
    ValueSet &InputSet, const SmallVectorImpl<Value *> &LCArgs,
    const SmallVectorImpl<Value *> &LCInputs, const ValueSet &TLInputsFixed) {
  // Add the argument structure
  HelperArgs.insert(TLInputsFixed[0]);
  HelperInputs.push_back(TLInputsFixed[0]);

  // Add the loop-control inputs.
  auto LCArgsIter = LCArgs.begin();
  auto LCInputsIter = LCInputs.begin();
  // First, add the start iteration.
  HelperArgs.insert(*LCArgsIter);
  HelperInputs.push_back(*LCInputsIter);
  if (!isa<Constant>(*LCInputsIter))
    InputSet.insert(*LCInputsIter);
  // Next, add the end iteration.
  ++LCArgsIter;
  ++LCInputsIter;
  HelperArgs.insert(*LCArgsIter);
  HelperInputs.push_back(*LCInputsIter);
  if (!isa<Constant>(*LCInputsIter))
    InputSet.insert(*LCInputsIter);

  // Save the third loop-control input -- the grainsize -- for use later.
  ++LCArgsIter;
  ++LCInputsIter;
  HelperArgs.insert(*LCArgsIter);
  HelperInputs.push_back(*LCInputsIter);
  if (!isa<Constant>(*LCInputsIter))
    InputSet.insert(*LCInputsIter);
}

unsigned RuntimeCilkFor::getIVArgIndex(const Function &F, const ValueSet &Args)
  const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

void RuntimeCilkFor::postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                                        ValueToValueMapTy &VMap) {
  Function *Helper = Out.Outline;
  // If the helper uses an argument structure, then it is not a write-only
  // function.
  if (getArgStructMode() != ArgStructMode::None) {
    Helper->removeFnAttr(Attribute::WriteOnly);
    Helper->removeFnAttr(Attribute::ArgMemOnly);
    Helper->removeFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
  }
}

void RuntimeCilkFor::processOutlinedLoopCall(TapirLoopInfo &TL,
                                             TaskOutlineInfo &TOI,
                                             DominatorTree &DT) {
  Function *Outlined = TOI.Outline;
  Instruction *ReplStart = TOI.ReplStart;
  Instruction *ReplCall = TOI.ReplCall;
  CallSite CS(ReplCall);
  BasicBlock *CallCont = TOI.ReplRet;
  BasicBlock *UnwindDest = TOI.ReplUnwind;
  Function *Parent = ReplCall->getFunction();
  Module &M = *Parent->getParent();
  unsigned IVArgIndex = getIVArgIndex(*Parent, TOI.InputSet);
  Type *PrimaryIVTy =
      CS.getArgOperand(IVArgIndex)->getType();
  Value *TripCount = CS.getArgOperand(IVArgIndex + 1);
  Value *GrainsizeVal = CS.getArgOperand(IVArgIndex + 2);

  // Get the correct CilkForABI call.
  Function *CilkForABI;
  if (PrimaryIVTy->isIntegerTy(32))
    CilkForABI = CILKRTS_FUNC(cilk_for_32, M);
  else if (PrimaryIVTy->isIntegerTy(64))
    CilkForABI = CILKRTS_FUNC(cilk_for_64, M);
  else
    llvm_unreachable("No CilkForABI call matches IV type for Tapir loop.");

  // Get the grainsize input
  Value *GrainsizeInput;
  {
    IRBuilder<> B(ReplCall);
    GrainsizeInput = B.CreateIntCast(GrainsizeVal, GrainsizeType,
                                     /*isSigned*/ false);
  }

  // Split the basic block containing the detach replacement just before the
  // start of the detach-replacement instructions.
  BasicBlock *DetBlock = ReplStart->getParent();
  BasicBlock *CallBlock = SplitBlock(DetBlock, ReplStart);

  LLVMContext &C = M.getContext();

  // Insert a call or invoke to the cilk_for ABI method.
  LLVM_DEBUG(dbgs() << "RuntimeCilkFor: Adding call to __cilkrts_cilk_for\n");
  IRBuilder<> B(ReplCall);
  Type *FPtrTy = PointerType::getUnqual(
      FunctionType::get(Type::getVoidTy(C),
                        { Type::getInt8PtrTy(C), PrimaryIVTy, PrimaryIVTy },
                        false));
  Value *OutlinedFnPtr = B.CreatePointerBitCastOrAddrSpaceCast(Outlined,
                                                               FPtrTy);
  AllocaInst *ArgStruct = cast<AllocaInst>(CS.getArgOperand(0));
  Value *ArgStructPtr = B.CreateBitCast(ArgStruct, Type::getInt8PtrTy(C));
  if (UnwindDest) {
    InvokeInst *Invoke = InvokeInst::Create(CilkForABI, CallCont, UnwindDest,
                                            { OutlinedFnPtr, ArgStructPtr,
                                              TripCount, GrainsizeInput });
    Invoke->setDebugLoc(ReplCall->getDebugLoc());
    ReplaceInstWithInst(ReplCall, Invoke);
    TOI.replaceReplCall(Invoke);
  } else {
    CallInst *Call = B.CreateCall(CilkForABI,
                                  { OutlinedFnPtr, ArgStructPtr,
                                    TripCount, GrainsizeInput });
    Call->setDebugLoc(ReplCall->getDebugLoc());
    Call->setDoesNotThrow();
    TOI.replaceReplCall(Call);
    ReplCall->eraseFromParent();
  }

  // If we're not using dynamic argument structs, then no further processing is
  // needed.
  if (ArgStructMode::Dynamic != getArgStructMode())
    return;

  // N.B. The following code to outline the invocation of the __cilkrts_cilk_for
  // call, is primarily included for debugging purposes.  In practice, this code
  // should not run, because the __cilkrts_cilk_for ABI should work with a
  // static structure.
  LLVM_DEBUG(dbgs() << "RuntimeCilkFor: Adding additional spawn helper to "
             << "manage dynamic argument-struct allocation.\n");

  // Update the value of ReplCall.
  ReplCall = TOI.ReplCall;
  // Create a separate spawn-helper function to allocate and populate the
  // argument struct.
  // Inputs to the spawn helper
  ValueSet SHInputSet = TOI.InputSet;
  SHInputSet.insert(GrainsizeVal);
  ValueSet SHInputs;
  fixupInputSet(*Parent, SHInputSet, SHInputs);
  LLVM_DEBUG({
      dbgs() << "SHInputSet:\n";
      for (Value *V : SHInputSet)
        dbgs() << "\t" << *V << "\n";
      dbgs() << "SHInputs:\n";
      for (Value *V : SHInputs)
        dbgs() << "\t" << *V << "\n";
    });

  ValueSet Outputs;  // Should be empty.
  // Only one block needs to be cloned into the spawn helper
  std::vector<BasicBlock *> BlocksToClone;
  BlocksToClone.push_back(CallBlock);
  SmallVector<ReturnInst *, 1> Returns;  // Ignore returns cloned.
  ValueToValueMapTy VMap;
  Twine NameSuffix = ".shelper";
  Function *SpawnHelper =
      CreateHelper(SHInputs, Outputs, BlocksToClone, CallBlock, DetBlock,
                   CallCont, VMap, &M, Parent->getSubprogram() != nullptr,
                   Returns, NameSuffix.str(), nullptr, nullptr, nullptr,
                   UnwindDest);

  assert(Returns.empty() && "Returns cloned when creating SpawnHelper.");

  // If there is no unwind destination, then the SpawnHelper cannot throw.
  if (!UnwindDest)
    SpawnHelper->setDoesNotThrow();

  // Add attributes to new helper function.
  //
  // Use a fast calling convention for the helper.
  SpawnHelper->setCallingConv(CallingConv::Fast);
  // Note that the address of the helper is unimportant.
  SpawnHelper->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  // The helper is private to this module.
  SpawnHelper->setLinkage(GlobalValue::PrivateLinkage);

  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  AddAlignmentAssumptions(Parent, SHInputs, VMap, ReplCall, nullptr, nullptr);

  // Move allocas in the newly cloned block to the entry block of the helper.
  {
    // Collect the end instructions of the task.
    SmallVector<Instruction *, 4> Ends;
    Ends.push_back(cast<BasicBlock>(VMap[CallCont])->getTerminator());
    if (isa<InvokeInst>(ReplCall))
      Ends.push_back(cast<BasicBlock>(VMap[UnwindDest])->getTerminator());

    // Move allocas in cloned detached block to entry of helper function.
    BasicBlock *ClonedBlock = cast<BasicBlock>(VMap[CallBlock]);
    MoveStaticAllocasInBlock(&SpawnHelper->getEntryBlock(), ClonedBlock, Ends);

    // We do not need to add new llvm.stacksave/llvm.stackrestore intrinsics,
    // because calling and returning from the helper will automatically manage
    // the stack appropriately.
  }

  // Insert a call to the spawn helper.
  SmallVector<Value *, 8> SHInputVec;
  for (Value *V : SHInputs)
    SHInputVec.push_back(V);
  SplitEdge(DetBlock, CallBlock);
  B.SetInsertPoint(CallBlock->getTerminator());
  if (isa<InvokeInst>(ReplCall)) {
    InvokeInst *SpawnHelperCall = InvokeInst::Create(SpawnHelper, CallCont,
                                                     UnwindDest, SHInputVec);
    SpawnHelperCall->setDebugLoc(ReplCall->getDebugLoc());
    SpawnHelperCall->setCallingConv(SpawnHelper->getCallingConv());
    ReplaceInstWithInst(CallBlock->getTerminator(), SpawnHelperCall);
  } else {
    CallInst *SpawnHelperCall = B.CreateCall(SpawnHelper, SHInputVec);
    SpawnHelperCall->setDebugLoc(ReplCall->getDebugLoc());
    SpawnHelperCall->setCallingConv(SpawnHelper->getCallingConv());
    SpawnHelperCall->setDoesNotThrow();
    // Branch around CallBlock.  Its contents are now dead.
    ReplaceInstWithInst(CallBlock->getTerminator(),
                        BranchInst::Create(CallCont));
  }
}
