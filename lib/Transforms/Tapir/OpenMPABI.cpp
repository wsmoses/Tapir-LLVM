//===- OpenMPABI.cpp - Lower Tapir into Cilk runtime system calls -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CilkABI interface, which is used to convert Tapir
// instructions -- detach, reattach, and sync -- to calls into the Cilk
// runtime system.  This interface does the low-level dirty work of passes
// such as LowerToCilk.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/OpenMPABI.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "ompabi"

StructType *IdentTy = nullptr;
FunctionType *Kmpc_MicroTy = nullptr;
Constant *DefaultOpenMPPSource = nullptr;
Constant *DefaultOpenMPLocation = nullptr;
PointerType *KmpRoutineEntryPtrTy = nullptr;

// instruction of its thread id.
typedef DenseMap<Function *, Value *> OpenMPThreadIDAllocaMapTy;
OpenMPThreadIDAllocaMapTy OpenMPThreadIDAllocaMap;

// Maps a funtion to the instruction where we loaded the thread id addrs
typedef DenseMap<Function *, Value *> OpenMPThreadIDLoadMapTy;
OpenMPThreadIDLoadMapTy OpenMPThreadIDLoadMap;

// Maps an extracted forked function (Using CodeExtractor) to its
// corresponding task outlined function as required by OMP runtime.
typedef DenseMap<Function *, Function *> ExtractedToOutlinedMapTy;
ExtractedToOutlinedMapTy ExtractedToOutlinedMap;

// Maps an outlined task function to its corresponding task entry function.
typedef DenseMap<Function *, Function *> OutlinedToEntryMapTy;
OutlinedToEntryMapTy OutlinedToEntryMap;

Value *emitTaskInit(Function *Caller,
                                     IRBuilder<> &CallerIRBuilder,
                                     IRBuilder<> &CallerAllocaIRBuilder,
                                     Function *ForkedFn,
                                     ArrayRef<Value *> LoadedCapturedArgs);

void emitBranch(BasicBlock *Target, IRBuilder<> &IRBuilder);

void emitKmpRoutineEntryT(Module *M);

Type *createKmpTaskTTy(Module *M, PointerType *KmpRoutineEntryPtrTy);

Type *createKmpTaskTWithPrivatesTy(Type *KmpTaskTTy);
Function *emitTaskOutlinedFunction(Module *M,
                                                    Type *SharedsPtrTy,
                                                    Function *ForkedFn);

DenseMap<Argument *, Value *> startFunction(Function *Fn);

Function *emitProxyTaskFunction(
    Type *KmpTaskTWithPrivatesPtrTy, Type *SharedsPtrTy, Function *TaskFunction,
    Value *TaskPrivatesMap);

void emitTaskwaitCall(Function *Caller,
                                       IRBuilder<> &CallerIRBuilder,
                                       const DataLayout &DL);

PointerType *getIdentTyPointerTy() {
  return PointerType::getUnqual(IdentTy);
}

FunctionType *getOrCreateKmpc_MicroTy(LLVMContext &Context) {
  if (Kmpc_MicroTy == nullptr) {
    auto *Int32PtrTy = PointerType::getUnqual(Type::getInt32Ty(Context));
    Type *MicroParams[] = {Int32PtrTy, Int32PtrTy};
    Kmpc_MicroTy =
        FunctionType::get(Type::getVoidTy(Context), MicroParams, true);
  }

  return Kmpc_MicroTy;
}

PointerType *getKmpc_MicroPointerTy(LLVMContext &Context) {
  return PointerType::getUnqual(getOrCreateKmpc_MicroTy(Context));
}

/*
/// Extracts the region and tasks contained in \p PR and starts code OMP
/// generation.
void startRegionEmission(const ParallelRegion &PR,
                                          LoopInfo &LI, DominatorTree &DT,
                                          ScalarEvolution &SE) {
  // A utility function used to find calls that replace extracted parallel
  // region function.
  auto FindCallToExtractedFn = [](Function *SpawningFn,
                                  Function *ExtractedFn) {
    // Find the call instruction to the extracted region function: RegionFn.
    CallInst *ExtractedFnCI = nullptr;
    for (auto &BB : *SpawningFn) {
      if (CallInst *CI = dyn_cast<CallInst>(BB.begin())) {
        // NOTE: I use pointer equality here, is that fine?
        if (ExtractedFn == CI->getCalledFunction()) {
          ExtractedFnCI = CI;
          break;
        }
      }
    }

    assert(ExtractedFnCI != nullptr &&
           "Couldn't find the call to the extracted region function!");

    return ExtractedFnCI;
  };

  // Check if \p PR is a parallel loop
  Loop *L = LI.getLoopFor(PR.getFork().getParent());
  bool IsParallelLoop = (L && PR.isParallelLoopRegion(*L, DT));

  auto &ForkInst = PR.getFork();
  Function *SpawningFn = ForkInst.getParent()->getParent();

  if (IsParallelLoop) {
    auto *CIV = L->getCanonicalInductionVariable();
    assert(CIV && "Non-canonical loop");

    auto *PreHeader = L->getLoopPreheader();
    auto *LoopLatch = L->getLoopLatch();
    assert(LoopLatch &&
           "Only parallel loops with a single latch are supported.");

    auto &ForkedTask = PR.getForkedTask();
    auto &ContTask = PR.getContinuationTask();

    std::vector<BasicBlock *> RegionBBs;
    // Include the preheader which is required for private and firstprivate
    // support.
    RegionBBs.push_back(PreHeader);

    for (auto *BB : L->blocks()) {
      RegionBBs.push_back(BB);
    }

    assert(ContTask.getHaltsOrJoints().size() == 1 &&
           "Only 1 join per loop is currently supported\n");

    // Include the join and post join blocks which are required for lastprivate
    // support.
    auto *JoinBB = ContTask.getHaltsOrJoints()[0]->getParent();
    auto *PostJoinBB = JoinBB->getSingleSuccessor();
    RegionBBs.push_back(JoinBB);
    RegionBBs.push_back(PostJoinBB);

    // If a value is:
    //   1. Used inside the loop.
    //   2. Defined outside the loop
    // then, demote that value to memory so that it can be shared across
    // the serial/parallel region.
    // TODO this should be done for non-loop regions as well.
    // NOTE We don't need to do this for the preheader. For example, in case
    // of a firstprivate clause, we need to share the variable intentionally
    // by value.
    std::vector<AllocaInst *> DemotedAllocas;
    for (auto *BB : L->blocks()) {
      for (auto &I : *BB) {
        for (auto &U : I.operands()) {
          if (auto *I2 = dyn_cast<Instruction>(&*U)) {
            if (!L->contains(I2) &&
                I2->getParent() !=
                    PreHeader ) {
              // No need to demote what hasn't been promoted in the first place.
              if (dyn_cast<AllocaInst>(I2) &&
                  !PR.getPRI().isSafeToPromote(*dyn_cast<AllocaInst>(I2), DT)) {
                continue;
              }

              // Don't demote what has been already demoted.
              if (std::count(DemotedAllocas.begin(), DemotedAllocas.end(), I2)) {
                continue;
              }

              DemotedAllocas.push_back(DemoteRegToStack(*I2));
            }
          }
        }
      }
    }

    removePIRInstructions(ForkInst, ForkedTask, ContTask);

    CodeExtractor LoopExtractor(RegionBBs, &DT);
    Function *LoopFn = LoopExtractor.extractCodeRegion();
    bool MergeRes =
        MergeBlockIntoPredecessor(LoopFn->getEntryBlock().getSingleSuccessor());
    assert(MergeRes && "Couldn't merge the preheader");

    // assert(verifyExtractedFn(LoopFn));

    // Create the outlined function that contains the OpenMP calls required for
    // outermost regions. This corresponds to the "if (omp_get_num_threads() ==
    // 1)" part as indicated by the example in the header file.
    ValueToValueMapTy VMap;
    auto *OMPLoopFn = declareOMPRegionFn(LoopFn, false, VMap);
    // Create the outlined function that contains the OpenMP calls required for
    // nested regions. This corresponds to the "else" part as indicated by the
    // example in the header file.
    ValueToValueMapTy NestedVMap;
    auto *OMPNestedLoopFn = declareOMPRegionFn(LoopFn, true, NestedVMap);

    auto *ExtractedFnCI = FindCallToExtractedFn(SpawningFn, LoopFn);

    auto ArgIt = ExtractedFnCI->arg_begin();
    auto &LoopFnEntryBB = LoopFn->getEntryBlock();

    IRBuilder<> IRBuilder(&LoopFnEntryBB,
                          LoopFnEntryBB.getTerminator()->getIterator());

    while (ArgIt != ExtractedFnCI->arg_end()) {
      if (std::find(DemotedAllocas.begin(), DemotedAllocas.end(), *ArgIt) !=
          DemotedAllocas.end()) {
        auto ParamIt =
            std::next(LoopFn->arg_begin(),
                      std::distance(ExtractedFnCI->arg_begin(), ArgIt));

        auto *ParamVal = IRBuilder.CreateLoad(&*ParamIt);
        auto *ParamType = (PointerType*)ParamIt->getType();

        for (auto &U : ParamIt->uses()) {
          if (auto *UI = dyn_cast<Instruction>(U.getUser())) {
            if (UI != ParamVal) {
              UI->replaceAllUsesWith(ParamVal);
              UI->eraseFromParent();
            }
          }
        }

      }

      ++ArgIt;
    }


    // Replace ExtractedFnCI with an if-else region that calls the outermost and
    // nested functions.
    replaceExtractedRegionFnCall(ExtractedFnCI, OMPLoopFn, OMPNestedLoopFn);

    emitOMPRegionFn(OMPLoopFn, LoopFn, VMap, false);

    emitOMPRegionFn(OMPNestedLoopFn, LoopFn, NestedVMap, true);
  } else {
    // errs() << "Parallel Region:\n";
    // Split the fork instruction parent into 2 BBs to satisfy the assumption
    // of CodeExtractor (i.e. single-entry region and the head BB is the entry).
    auto *OldForkInstBB = ForkInst.getParent();
    auto *NewForkInstBB = SplitBlock(OldForkInstBB, &ForkInst);

    auto &ForkedTask = PR.getForkedTask();
    auto &ContTask = PR.getContinuationTask();

    // Collect all the BBs of forked and continuation tasks for extraction.
    std::vector<BasicBlock *> RegionBBs;
    // Keep track of which blocks belong to forked and continuation tasks
    // because we are about to replace fork-join instructions by regular
    // branches.
    std::vector<BasicBlock *> ForkedBBs;
    std::vector<BasicBlock *> ContBBs;
    RegionBBs.push_back(NewForkInstBB);

    ParallelTask::VisitorTy ForkedVisitor = [&RegionBBs, &ForkedBBs](
        BasicBlock &BB, const ParallelTask &PT) -> bool {
      RegionBBs.push_back(&BB);
      ForkedBBs.push_back(&BB);

      return true;
    };

    ParallelTask::VisitorTy ContVisitor =
        [&RegionBBs, &ContBBs](BasicBlock &BB, const ParallelTask &PT) -> bool {
      RegionBBs.push_back(&BB);
      ContBBs.push_back(&BB);

      return true;
    };

    // Collect all the BBs of the forked task.
    ForkedTask.visit(ForkedVisitor, true);
    // Collect all the BBs of the continuation task.
    ContTask.visit(ContVisitor, true);

    removePIRInstructions(ForkInst, ForkedTask, ContTask);

    CodeExtractor RegionExtractor(RegionBBs);
    Function *RegionFn = RegionExtractor.extractCodeRegion();
    assert(verifyExtractedFn(RegionFn));

    CodeExtractor ForkedExtractor(ForkedBBs);
    Function *ForkedFn = ForkedExtractor.extractCodeRegion();
    // The sub-set of RegionFn args that is passed to ForkedFn.
    std::vector<Argument *> ForkedFnArgs;
    std::vector<Argument *> ForkedFnNestedArgs;

    CodeExtractor ContExtractor(ContBBs);
    Function *ContFn = ContExtractor.extractCodeRegion();
    // The sub-set of RegionFn args that is passed to ContFn
    std::vector<Argument *> ContFnArgs;
    std::vector<Argument *> ContFnNestedArgs;

    // Create the outlined function that contains the OpenMP calls required for
    // outermost regions. This corresponds to the "if (omp_get_num_threads() ==
    // 1)" part as indicated by the example in the header file.
    ValueToValueMapTy VMap;
    auto *OMPRegionFn = declareOMPRegionFn(RegionFn, false, VMap);
    // Create the outlined function that contains the OpenMP calls required for
    // nested regions. This corresponds to the "else" part as indicated by the
    // example in the header file.
    ValueToValueMapTy NestedVMap;
    auto *OMPNestedRegionFn = declareOMPRegionFn(RegionFn, true, NestedVMap);

    auto *ExtractedFnCI = FindCallToExtractedFn(SpawningFn, RegionFn);

    // Replace ExtractedFnCI with an if-else region that calls the outermost and
    // nested functions.
    replaceExtractedRegionFnCall(ExtractedFnCI, OMPRegionFn, OMPNestedRegionFn);

    // Calculates the sub-list of \p VMap's values that should be passed to \p
    // TaskFn.
    auto filterCalledFnArgs = [](Function *ExtractedRegionFn, Function *TaskFn,
                                 ValueToValueMapTy &VMap) {
      std::vector<Argument *> FilteredArgs;

      // Locate the CallInst to TaskFn
      for (auto &BB : *ExtractedRegionFn) {
        if (CallInst *CI = dyn_cast<CallInst>(BB.begin())) {
          if (TaskFn == CI->getCalledFunction()) {
            auto ChildArgIt = CI->arg_begin();

            while (ChildArgIt != CI->arg_end()) {
              for (auto &Arg : ExtractedRegionFn->args()) {
                if (ChildArgIt->get() == &Arg) {
                  FilteredArgs.push_back(dyn_cast<Argument>(VMap[&Arg]));
                  break;
                }
              }

              ++ChildArgIt;
            }

            break;
          }
        }
      }

      return FilteredArgs;
    };

    ForkedFnArgs = filterCalledFnArgs(RegionFn, ForkedFn, VMap);
    ForkedFnNestedArgs = filterCalledFnArgs(RegionFn, ForkedFn, NestedVMap);
    ContFnArgs = filterCalledFnArgs(RegionFn, ContFn, VMap);
    ContFnNestedArgs = filterCalledFnArgs(RegionFn, ContFn, NestedVMap);

    // Emit the function containing outermost logic.
    emitOMPRegionFn(OMPRegionFn, ForkedFn, ContFn, ForkedFnArgs, ContFnArgs,
                    false);

    // Emit the function containing nested logic.
    emitOMPRegionFn(OMPNestedRegionFn, ForkedFn, ContFn, ForkedFnNestedArgs,
                    ContFnNestedArgs, true);
  }
}
*/

/*
void removePIRInstructions(ForkInst &ForkInst,
                                            const ParallelTask &ForkedTask,
                                            const ParallelTask &ContTask) {
  // Replace fork, halt, and join instructions with br, otherwise, we will end
  // up in an infinite loop since the region's extracted function will contain
  // a "new" parallel region.

  // Replace fork with branch
  BranchInst::Create(&ForkedTask.getEntry(), &ForkInst);
  ForkInst.eraseFromParent();

  // Replace halts with branches
  for (auto *I : ForkedTask.getHaltsOrJoints()) {
    if (HaltInst *HI = dyn_cast<HaltInst>(I)) {
      BranchInst::Create(HI->getContinuationBB(), HI);
      HI->eraseFromParent();
    } else {
      assert(false && "A forked task is terminated by a join instruction");
    }
  }

  // Replace joins with branches
  for (auto *I : ContTask.getHaltsOrJoints()) {
    if (JoinInst *JI = dyn_cast<JoinInst>(I)) {
      BranchInst::Create(JI->getSuccessor(0), JI);
      JI->eraseFromParent();
    } else {
      assert(false &&
             "A continuation task is terminated by a halt instruction");
    }
  }
}
*/

/// Creates the declaration for a function that will contain OpenMP runtime
/// code.
///
/// \param RegionFn an extracted function of a parallel region.
///
/// \param Nested whether the requested function is for the outermost case or
/// the nested case
///
/// \param [out] VMap maps the arguments of \p RegionFn to arguments in the
/// returned function.
Function *declareOMPRegionFn(Function *RegionFn, bool Nested,
                                              ValueToValueMapTy &VMap) {
  auto *Module = RegionFn->getParent();
  auto &Context = Module->getContext();
  DataLayout DL(Module);

  /*
  std::vector<Type *> FnParams;
  std::vector<StringRef> FnArgNames;
  std::vector<AttributeSet> FnArgAttrs;

  if (!Nested) {
    // OMP parallel regions require some implicit arguments. Add them.
    auto *Int32PtrTy = PointerType::getUnqual(Type::getInt32Ty(Context));
    FnParams.push_back(Int32PtrTy);
    FnParams.push_back(Int32PtrTy);

    FnArgNames.push_back(".global_tid.");
    FnArgNames.push_back(".bound_tid.");

    //FnArgAttrs.push_back(AttributeSet::get(Context, ArrayRef<Attribute>(Attribute::NoAlias));
    //FnArgAttrs.push_back(AttributeSet::get(Context, ArrayRef<Attribute>(Attribute::NoAlias)));
  }

  auto &RegionFnArgList = RegionFn->getArgumentList();

  int ArgOffset = Nested ? 0 : 2;

  // For RegionFn argument add a corresponding argument to the new function.
  for (auto &Arg : RegionFnArgList) {
    FnParams.push_back(Arg.getType());
    FnArgNames.push_back(Arg.getName());

    // Allow speculative loading from shared data.
    if (Arg.getType()->isPointerTy()) {
      AttrBuilder B;
      B.addDereferenceableAttr(
          DL.getTypeAllocSize(Arg.getType()->getPointerElementType()));
      FnArgAttrs.push_back(AttributeSet::get(Context, ++ArgOffset, B));
    } else {
      FnArgAttrs.push_back(AttributeSet());
      ++ArgOffset;
    }
  }

  // Create the function and set its argument properties.
  auto *VoidTy = Type::getVoidTy(Context);
  auto *OMPRegionFnTy = FunctionType::get(VoidTy, FnParams, false);
  auto Name = RegionFn->getName() + (Nested ? ".Nested.OMP" : ".OMP");
  Function *OMPRegionFn = dyn_cast<Function>(
      Module->getOrInsertFunction(Name.str(), OMPRegionFnTy));
  auto &FnArgList = OMPRegionFn->getArgumentList();

  auto ArgIt = FnArgList.begin();
  auto ArgNameIt = FnArgNames.begin();

  for (auto &ArgAttr : FnArgAttrs) {
    if (!ArgAttr.isEmpty()) {
      (*ArgIt).addAttr(ArgAttr);
    }
    (*ArgIt).setName(*ArgNameIt);
    ++ArgIt;
    ++ArgNameIt;
  }

  // If this is an outermost region, skip the first 2 arguments (global_tid and
  // bound_tid) ...
  auto OMPArgIt = OMPRegionFn->arg_begin();
  if (!Nested) {
    ++OMPArgIt;
    ++OMPArgIt;
  }
  // ... then map corresponding arguments in RegionFn and OMPRegionFn
  for (auto &Arg : RegionFnArgList) {
    VMap[&Arg] = &*OMPArgIt;
    ++OMPArgIt;
  }

  return OMPRegionFn;
  */
}

Constant *createRuntimeFunction(OpenMPRuntimeFunction Function,
                                                 Module *M) {
  auto *VoidTy = Type::getVoidTy(M->getContext());
  auto *VoidPtrTy = Type::getInt8PtrTy(M->getContext());
  auto *Int32Ty = Type::getInt32Ty(M->getContext());
  auto *Int32PtrTy = Type::getInt32PtrTy(M->getContext());
  // TODO double check for how SizeTy get created. Eventually, it get emitted
  // as i64 on my machine.
  auto *SizeTy = Type::getInt64Ty(M->getContext());
  auto *IdentTyPtrTy = getIdentTyPointerTy();
  Constant *RTLFn = nullptr;

  switch (Function) {
  case OMPRTL__kmpc_fork_call: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty,
                          getKmpc_MicroPointerTy(M->getContext())};
    FunctionType *FnTy = FunctionType::get(VoidTy, TypeParams, true);
    RTLFn = M->getOrInsertFunction("__kmpc_fork_call", FnTy);
    break;
  }
  case OMPRTL__kmpc_for_static_init_4: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty,    Int32Ty,
                          Int32PtrTy,   Int32PtrTy, Int32PtrTy,
                          Int32PtrTy,   Int32Ty,    Int32Ty};
    FunctionType *FnTy =
      FunctionType::get(VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = M->getOrInsertFunction("__kmpc_for_static_init_4", FnTy);
    break;
  }
  case OMPRTL__kmpc_for_static_fini: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty};
    FunctionType *FnTy =
        FunctionType::get(VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = M->getOrInsertFunction("__kmpc_for_static_fini", FnTy);
    break;
  }
  case OMPRTL__kmpc_master: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty};
    FunctionType *FnTy =
        FunctionType::get(Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_master", FnTy);
    break;
  }
  case OMPRTL__kmpc_end_master: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty};
    FunctionType *FnTy =
        FunctionType::get(VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_end_master", FnTy);
    break;
  }
  case OMPRTL__kmpc_omp_task_alloc: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty, Int32Ty,
                          SizeTy,       SizeTy,  KmpRoutineEntryPtrTy};
    FunctionType *FnTy =
        FunctionType::get(VoidPtrTy, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_omp_task_alloc", FnTy);
    break;
  }
  case OMPRTL__kmpc_omp_task: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty, VoidPtrTy};
    FunctionType *FnTy =
        FunctionType::get(Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_omp_task", FnTy);
    break;
  }
  case OMPRTL__kmpc_omp_taskwait: {
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty};
    FunctionType *FnTy =
        FunctionType::get(Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_omp_taskwait", FnTy);
    break;
  }
  case OMPRTL__kmpc_global_thread_num: {
    Type *TypeParams[] = {IdentTyPtrTy};
    FunctionType *FnTy =
        FunctionType::get(Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_global_thread_num", FnTy);
    break;
  }
  case OMPRTL__kmpc_barrier: {
    // NOTE There is more elaborate logic to emitting barriers based on the
    // directive kind. This is just the simplified version currently needed.
    // Check: CGOpenMPRuntime::emitBarrierCall.
    Type *TypeParams[] = {IdentTyPtrTy, Int32Ty};
    FunctionType *FnTy =
      FunctionType::get(Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = M->getOrInsertFunction("__kmpc_barrier", FnTy);
    break;
  }
  }
  return RTLFn;
}

CallInst *emitRuntimeCall(Value *Callee,
                                           ArrayRef<Value *> Args,
                                           const Twine &Name,
                                           BasicBlock *Parent) {
  IRBuilder<> Builder(Parent);
  CallInst *call = Builder.CreateCall(Callee, Args, None, Name);
  return call;
}

CallInst *emitRuntimeCall(Value *Callee,
                                           ArrayRef<Value *> Args,
                                           const Twine &Name,
                                           IRBuilder<> &IRBuilder) {
  CallInst *call = IRBuilder.CreateCall(Callee, Args, None, Name);
  return call;
}

/// Replaces \p CI with an if-else region that calls \p OMPRegionFn and \p
/// OMPNestedRegionFn.
///
/// \param CI a CallInst to an extracted parallel region function.
///
/// \param OMPRegionFn an outlined function containing outermost OMP logic.
///
/// \param OMPNestedRegionFn an outlined function containing nested OMP logic.
std::pair<CallInst *, CallInst *> replaceExtractedRegionFnCall(
    CallInst *CI, Function *OMPRegionFn, Function *OMPNestedRegionFn) {
  auto *RegionFn = CI->getCalledFunction();
  auto *SpawningFn = CI->getParent()->getParent();
  auto *Module = SpawningFn->getParent();
  auto &Context = Module->getContext();

  IRBuilder<> IRBuilder(CI);
  // For outer regions, pass some parameters required by OMP runtime.
  auto *Int32Ty = Type::getInt32Ty(Context);
  std::vector<Value *> OMPRegionFnArgs = {
      DefaultOpenMPLocation,
      ConstantInt::getSigned(Int32Ty, RegionFn->arg_size()),
      IRBuilder.CreateBitCast(OMPRegionFn, getKmpc_MicroPointerTy(Context))};

  std::vector<Value *> OMPNestedRegionFnArgs;
  auto ArgIt = CI->arg_begin();

  // Append the rest of the region's arguments.
  while (ArgIt != CI->arg_end()) {
    OMPRegionFnArgs.push_back(ArgIt->get());
    OMPNestedRegionFnArgs.push_back(ArgIt->get());
    ++ArgIt;
  }

  auto *BBRemainder = CI->getParent()->splitBasicBlock(
      std::next(CI->getIterator()), "remainder");

  // Emit:
  // if (omp_get_num_threads() == 1) {
  //   call __kmpc_fork_call passing it OMPRegionFn
  // } else {
  //   call OMPNestedRegionFn
  // }
  auto *GetNumThreadsFn = Module->getOrInsertFunction(
      "omp_get_num_threads", FunctionType::get(Int32Ty, false));
  auto *NonNestedBB =
      BasicBlock::Create(Context, "non.nested", SpawningFn, nullptr);
  auto *NestedBB = BasicBlock::Create(Context, "nested", SpawningFn, nullptr);
  CI->getParent()->getTerminator()->eraseFromParent();

  auto *IsNotNested = IRBuilder.CreateICmpEQ(
      IRBuilder.CreateCall(GetNumThreadsFn), IRBuilder.getInt32(1));
  IRBuilder.CreateCondBr(IsNotNested, NonNestedBB, NestedBB);

  IRBuilder.SetInsertPoint(NonNestedBB);
  auto ForkRTFn = createRuntimeFunction(
      OpenMPRuntimeFunction::OMPRTL__kmpc_fork_call, Module);
  // Replace the old call with __kmpc_fork_call
  auto *ForkCall = emitRuntimeCall(ForkRTFn, OMPRegionFnArgs, "", IRBuilder);
  IRBuilder.CreateBr(BBRemainder);

  IRBuilder.SetInsertPoint(NestedBB);
  // Replace the old call with a call to the nested version of the parallel
  // region.
  auto *NestedCall = emitRuntimeCall(OMPNestedRegionFn, OMPNestedRegionFnArgs, "", IRBuilder);
  IRBuilder.CreateBr(BBRemainder);

  CI->eraseFromParent();
  return {ForkCall, NestedCall};
}

DenseMap<Argument *, Value *>
emitImplicitArgs(Function *OMPRegionFn,
                                  IRBuilder<> &AllocaIRBuilder,
                                  IRBuilder<> &StoreIRBuilder, bool Nested) {
  DataLayout DL(OMPRegionFn->getParent());
  DenseMap<Argument *, Value *> ParamToAllocaMap;

  auto emitArgProlog = [&](Argument &Arg) {
    auto Alloca = AllocaIRBuilder.CreateAlloca(Arg.getType(), nullptr,
                                               Arg.getName() + ".addr");
    Alloca->setAlignment(DL.getTypeAllocSize(Arg.getType()));
    StoreIRBuilder.CreateAlignedStore(&Arg, Alloca,
                                      DL.getTypeAllocSize(Arg.getType()));

    return (Value *)Alloca;
  };

  auto ArgI = OMPRegionFn->arg_begin();

  if (!Nested) {
    auto GtidAlloca = emitArgProlog(*ArgI);
    // Add an entry for the current function (representing an outlined outer
    // region) and its associated global thread id address
    auto &Elem = OpenMPThreadIDAllocaMap.FindAndConstruct(OMPRegionFn);
    Elem.second = GtidAlloca;
    ++ArgI;

    emitArgProlog(*ArgI);
    ++ArgI;
  }

  while (ArgI != OMPRegionFn->arg_end()) {
    auto *Alloca = emitArgProlog(*ArgI);
    auto &ArgToAllocaIt = ParamToAllocaMap.FindAndConstruct(&*ArgI);
    ArgToAllocaIt.second = Alloca;
    ++ArgI;
  }

  return ParamToAllocaMap;
}

Value *getThreadID(Function *F, IRBuilder<> &IRBuilder) {
  Value *ThreadID = nullptr;
  auto I = OpenMPThreadIDLoadMap.find(F);
  if (I != OpenMPThreadIDLoadMap.end()) {
    ThreadID = I->second;
    assert(ThreadID != nullptr && "A null thread ID associated to F");
    return ThreadID;
  }

  auto I2 = OpenMPThreadIDAllocaMap.find(F);

  if (I2 != OpenMPThreadIDAllocaMap.end()) {
    DataLayout DL(F->getParent());
    auto Alloca = I2->second;
    auto ThreadIDAddrs = IRBuilder.CreateLoad(Alloca);
    ThreadIDAddrs->setAlignment(DL.getTypeAllocSize(ThreadIDAddrs->getType()));
    ThreadID = IRBuilder.CreateLoad(ThreadIDAddrs);
    ((LoadInst *)ThreadID)
        ->setAlignment(DL.getTypeAllocSize(ThreadID->getType()));
    auto &Elem = OpenMPThreadIDLoadMap.FindAndConstruct(F);
    Elem.second = ThreadID;
    return ThreadID;
  }

  auto GTIDFn = createRuntimeFunction(
      OpenMPRuntimeFunction::OMPRTL__kmpc_global_thread_num, F->getParent());
  ThreadID = emitRuntimeCall(GTIDFn, {DefaultOpenMPLocation}, "", IRBuilder);
  auto &Elem = OpenMPThreadIDLoadMap.FindAndConstruct(F);
  Elem.second = ThreadID;

  return ThreadID;
}

Value *getThreadID(Function *F) {
  LoadInst *ThreadID = nullptr;
  auto I = OpenMPThreadIDLoadMap.find(F);
  if (I != OpenMPThreadIDLoadMap.end()) {
    ThreadID = (LoadInst *)I->second;
    assert(ThreadID != nullptr && "A null thread ID associated to F");
    return ThreadID;
  }

  return nullptr;
}

/*
void emitOMPRegionLogic(
    Function *OMPRegionFn, IRBuilder<> &IRBuilder,
    ::IRBuilder<> &AllocaIRBuilder, BasicBlock *LoopEntry,
    DenseMap<Argument *, Value *> ParamToAllocaMap, bool Nested) {
  Module *M = OMPRegionFn->getParent();
  LLVMContext &C = OMPRegionFn->getContext();
  DataLayout DL(OMPRegionFn->getParent());
  auto *Int32Ty = Type::getInt32Ty(C);
  auto *LoopHeader = LoopEntry->getSingleSuccessor();

  auto CapturedArgIt = ParamToAllocaMap.begin();

  while (CapturedArgIt != ParamToAllocaMap.end()) {
    auto *CapturedArgLoad = IRBuilder.CreateAlignedLoad(
        CapturedArgIt->second,
        DL.getTypeAllocSize(
            CapturedArgIt->second->getType()->getPointerElementType()));

    ++CapturedArgIt;
  }

  // Emit iteration variable
  auto *IV = AllocaIRBuilder.CreateAlloca(Int32Ty, nullptr, ".omp.iv");
  IV->setAlignment(DL.getTypeAllocSize(Int32Ty));
  // Emit alloca's to hold the thread's lower-bound, upper-bound, stride,
  // and is-last values.
  auto *LB = AllocaIRBuilder.CreateAlloca(Int32Ty, nullptr, ".omp.lb");
  LB->setAlignment(DL.getTypeAllocSize(Int32Ty));
  auto *UB = AllocaIRBuilder.CreateAlloca(Int32Ty, nullptr, ".omp.ub");
  UB->setAlignment(DL.getTypeAllocSize(Int32Ty));
  auto *Stride = AllocaIRBuilder.CreateAlloca(Int32Ty, nullptr, ".omp.stride");
  Stride->setAlignment(DL.getTypeAllocSize(Int32Ty));
  auto *IsLast = AllocaIRBuilder.CreateAlloca(Int32Ty, nullptr, ".omp.is_last");
  Stride->setAlignment(DL.getTypeAllocSize(Int32Ty));

  getThreadID(OMPRegionFn, IRBuilder);

  IRBuilder.CreateBr(LoopEntry);
  IRBuilder.SetInsertPoint(LoopEntry->getTerminator());

  //LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>(*OMPRegionFn).getLoopInfo();
  DominatorTree &DT =
    getAnalysis<DominatorTreeWrapperPass>(*OMPRegionFn).getDomTree();
  ScalarEvolution &SE =
    getAnalysis<ScalarEvolutionWrapperPass>(*OMPRegionFn).getSE();
  Loop *L = LI.getLoopFor(LoopHeader);
  auto *LoopExit = L->getExitingBlock();
  auto *PostLoopExit = *std::next(succ_begin(LoopExit));
  auto *LatchCount = SE.getExitCount(L, LoopExit);
  SCEVExpander Exp(SE, DL, "LatchExpander");
  Exp.setInsertPoint(LoopEntry->getTerminator());
  auto *LatchCountVal = Exp.expandCodeFor(LatchCount);
  auto *TripCountVal =
      IRBuilder.CreateAdd(LatchCountVal, IRBuilder.getInt32(1), "trip.cnt");
  auto *ExecLoopCmp =
      IRBuilder.CreateICmpSLT(IRBuilder.getInt32(0), TripCountVal, "exec.loop");

  auto *ExecLoopBB =
      BasicBlock::Create(C, "omp.exec.loop", OMPRegionFn, LoopHeader);

  IRBuilder.CreateCondBr(ExecLoopCmp, ExecLoopBB, PostLoopExit);
  LoopEntry->getTerminator()->eraseFromParent();

  IRBuilder.SetInsertPoint(ExecLoopBB);
  // NOTE I am not sure why the OMP code generatd from clang inits these
  // variables if the runtime is going to initialize them anyway through the
  // call to __kmpc_for_static_init_4 right after.
  IRBuilder.CreateAlignedStore(IRBuilder.getInt32(0), LB,
                               DL.getTypeAllocSize(Int32Ty));
  IRBuilder.CreateAlignedStore(LatchCountVal, UB, DL.getTypeAllocSize(Int32Ty));
  IRBuilder.CreateAlignedStore(IRBuilder.getInt32(1), Stride,
                               DL.getTypeAllocSize(Int32Ty));
  IRBuilder.CreateAlignedStore(IRBuilder.getInt32(0), IsLast,
                               DL.getTypeAllocSize(Int32Ty));

  std::vector<Value *> ForInitArgs = {DefaultOpenMPLocation,
                                      getThreadID(OMPRegionFn, IRBuilder),
                                      IRBuilder.getInt32(34),
                                      IsLast,
                                      LB,
                                      UB,
                                      Stride,
                                      IRBuilder.getInt32(1),
                                      IRBuilder.getInt32(1)};
  emitRuntimeCall(createRuntimeFunction(
                      OpenMPRuntimeFunction::OMPRTL__kmpc_for_static_init_4, M),
                  ForInitArgs, "", IRBuilder);

  auto *UBVal = IRBuilder.CreateAlignedLoad(UB, DL.getTypeAllocSize(Int32Ty));
  auto *UBCmp = IRBuilder.CreateICmpSGT(UBVal, LatchCountVal);

  auto *UBCmpT =
    BasicBlock::Create(C, "ub.cmp.true", OMPRegionFn, LoopHeader);

  auto *UBCmpF =
    BasicBlock::Create(C, "ub.cmp.false", OMPRegionFn, LoopHeader);

  auto *LoopPreHeader =
    BasicBlock::Create(C, "loop.pre.head", OMPRegionFn, LoopHeader);

  IRBuilder.CreateCondBr(UBCmp, UBCmpT, UBCmpF);

  IRBuilder.SetInsertPoint(UBCmpT);
  IRBuilder.CreateBr(LoopPreHeader);

  IRBuilder.SetInsertPoint(UBCmpF);
  UBVal = IRBuilder.CreateAlignedLoad(UB, DL.getTypeAllocSize(Int32Ty));
  IRBuilder.CreateBr(LoopPreHeader);

  IRBuilder.SetInsertPoint(LoopPreHeader);
  auto* ActualUBVal = IRBuilder.CreatePHI(Int32Ty, 2, "actual.ub");
  ActualUBVal->addIncoming(LatchCountVal, UBCmpT);
  ActualUBVal->addIncoming(UBVal, UBCmpF);

  IRBuilder.CreateAlignedStore(ActualUBVal, UB, DL.getTypeAllocSize(Int32Ty));
  auto *LBVal = IRBuilder.CreateAlignedLoad(LB, DL.getTypeAllocSize(Int32Ty));
  IRBuilder.CreateAlignedStore(LBVal, IV, DL.getTypeAllocSize(Int32Ty));
  IRBuilder.CreateBr(LoopHeader);

  // Replace the old canonical iteration variable with the .omp.iv.
  auto *IndVarPHI = dyn_cast<PHINode>(&*LoopHeader->begin());
  assert(IndVarPHI && "Couldn't find the loop's canoncial induction variable");
  IRBuilder.SetInsertPoint(&*std::next(IndVarPHI->getIterator()));
  auto *IVVal = IRBuilder.CreateAlignedLoad(IV, DL.getTypeAllocSize(Int32Ty));
  IndVarPHI->replaceAllUsesWith(IVVal);
  IndVarPHI->eraseFromParent();

  auto *LatchCmp = dyn_cast<ICmpInst>(
      dyn_cast<BranchInst>(LoopExit->getTerminator())->getCondition());
  auto *IndVarInc = dyn_cast<BinaryOperator>(LatchCmp->getOperand(0));
  assert(IndVarInc->getOpcode() == Instruction::Add);
  IRBuilder.SetInsertPoint(&*std::next(IndVarInc->getIterator()));
  IRBuilder.CreateAlignedStore(IndVarInc, IV, DL.getTypeAllocSize(Int32Ty));
  UBVal = IRBuilder.CreateAlignedLoad(UB, DL.getTypeAllocSize(Int32Ty));
  auto *NewLatchCmp = IRBuilder.CreateICmpSLE(IndVarInc, UBVal);
  LatchCmp->replaceAllUsesWith(NewLatchCmp);
  LatchCmp->eraseFromParent();

  IRBuilder.SetInsertPoint(&*PostLoopExit->getTerminator());

  std::vector<Value *> FiniArgs = {DefaultOpenMPLocation,
                                   getThreadID(OMPRegionFn, IRBuilder)};
  emitRuntimeCall(createRuntimeFunction(
                      OpenMPRuntimeFunction::OMPRTL__kmpc_for_static_fini, M),
                  FiniArgs, "", IRBuilder);
  emitRuntimeCall(
      createRuntimeFunction(OpenMPRuntimeFunction::OMPRTL__kmpc_barrier, M),
      FiniArgs, "", IRBuilder);

  auto *PostJoinBB = PostLoopExit->getSingleSuccessor();
  assert(PostJoinBB);

  auto *ExitBB =
    BasicBlock::Create(C, "exit", OMPRegionFn);

  auto *IsLastVal =
      IRBuilder.CreateAlignedLoad(IsLast, DL.getTypeAllocSize(Int32Ty));
  auto *IsLastCmp = IRBuilder.CreateICmpNE(IsLastVal, IRBuilder.getInt32(0));
  // Post-join BB is only executed by the thread executing the last iteration.
  // This is where the finalization logic required by lastprivate is found.
  IRBuilder.CreateCondBr(IsLastCmp, PostJoinBB, ExitBB);
  PostLoopExit->getTerminator()->eraseFromParent();

  IRBuilder.SetInsertPoint(ExitBB);
  IRBuilder.CreateRetVoid();
}
*/

void emitOMPRegionLogic(
    Function *OMPRegionFn, IRBuilder<> &IRBuilder,
    ::IRBuilder<> &AllocaIRBuilder, Function *ForkedFn, Function *ContFn,
    DenseMap<Argument *, Value *> ParamToAllocaMap,
    ArrayRef<Argument *> ForkedFnArgs, ArrayRef<Argument *> ContFnArgs,
    bool Nested) {
  Module *M = OMPRegionFn->getParent();
  LLVMContext &C = OMPRegionFn->getContext();
  DataLayout DL(OMPRegionFn->getParent());

  auto CapturedArgIt = ParamToAllocaMap.begin();
  // The list of LoadInst results that will be  passed as arguments to ForkedFn.
  std::vector<Value *> ForkedCapArgsLoads;
  // The list of LoadInst results that will be  passed as arguments to ContFn.
  std::vector<Value *> ContCapArgsLoads;

  // For each captured variable, emit a LoadInst and add it to
  // ForkedCapArgsLoads and/or ContCapArgsLoads as needed.
  while (CapturedArgIt != ParamToAllocaMap.end()) {
    auto *CapturedArgLoad = IRBuilder.CreateAlignedLoad(
        CapturedArgIt->second,
        DL.getTypeAllocSize(
            CapturedArgIt->second->getType()->getPointerElementType()));

    auto PrepareLoadsVec = [&CapturedArgIt, &CapturedArgLoad](
                               ArrayRef<Argument *> FnArgs,
                               std::vector<Value *> &FnArgLoads) {
      auto ArgIt =
          std::find(FnArgs.begin(), FnArgs.end(), CapturedArgIt->first);
      if (ArgIt != FnArgs.end()) {
        auto Dist = std::distance(FnArgs.begin(), ArgIt);
        if (FnArgLoads.size() <= Dist) {
          FnArgLoads.resize(Dist + 1);
        }
        FnArgLoads[Dist] = CapturedArgLoad;
      }
    };

    PrepareLoadsVec(ForkedFnArgs, ForkedCapArgsLoads);
    PrepareLoadsVec(ContFnArgs, ContCapArgsLoads);

    ++CapturedArgIt;
  }

  BasicBlock *ExitBB =
      BasicBlock::Create(C, "omp_if.end", OMPRegionFn, nullptr);
  std::vector<Value *> MasterArgs = {DefaultOpenMPLocation,
                                  getThreadID(OMPRegionFn, IRBuilder)};

  if (!Nested) {
    auto MasterRTFn =
        createRuntimeFunction(OpenMPRuntimeFunction::OMPRTL__kmpc_master, M);
    auto IsMaster = emitRuntimeCall(MasterRTFn, MasterArgs, "", IRBuilder);

    BasicBlock *BodyBB =
        BasicBlock::Create(C, "omp_if.then", OMPRegionFn, nullptr);
    auto Cond = IRBuilder.CreateICmpNE(IsMaster, IRBuilder.getInt32(0));
    IRBuilder.CreateCondBr(Cond, BodyBB, ExitBB);
    IRBuilder.SetInsertPoint(BodyBB);
  }

  auto *NewTask = emitTaskInit(OMPRegionFn, IRBuilder, AllocaIRBuilder,
                               ForkedFn, ForkedCapArgsLoads);
  std::vector<Value *> TaskArgs = {DefaultOpenMPLocation,
                                getThreadID(OMPRegionFn, IRBuilder), NewTask};
  emitRuntimeCall(
      createRuntimeFunction(OpenMPRuntimeFunction::OMPRTL__kmpc_omp_task, M),
      TaskArgs, "", IRBuilder);

  IRBuilder.CreateCall(ContFn, ContCapArgsLoads);

  emitTaskwaitCall(OMPRegionFn, IRBuilder, DL);

  if (!Nested) {
    auto EndMasterRTFn = createRuntimeFunction(
        OpenMPRuntimeFunction::OMPRTL__kmpc_end_master, M);
    emitRuntimeCall(EndMasterRTFn, MasterArgs, "", IRBuilder);
  }

  emitBranch(ExitBB, IRBuilder);
}

/*
void emitOMPRegionFn(Function *OMPRegionFn, Function *LoopFn,
                                      ValueToValueMapTy &VMap, bool Nested) {
  SmallVector<ReturnInst*, 1> Returns;
  CloneFunctionInto(OMPRegionFn, LoopFn, VMap, false, Returns);
  auto &LoopEntry = OMPRegionFn->getEntryBlock();

  auto *Module = OMPRegionFn->getParent();
  auto &Context = Module->getContext();
  auto *EntryBB = BasicBlock::Create(Context, "entry", OMPRegionFn, &LoopEntry);
  auto Int32Ty = Type::getInt32Ty(Context);
  Value *Undef = UndefValue::get(Int32Ty);
  auto *AllocaInsertPt = new BitCastInst(Undef, Int32Ty, "allocapt", EntryBB);
  IRBuilder<> AllocaIRBuilder(EntryBB,
                              ((Instruction *)AllocaInsertPt)->getIterator());

  IRBuilder<> StoreIRBuilder(EntryBB);
  auto ParamToAllocaMap =
      emitImplicitArgs(OMPRegionFn, AllocaIRBuilder, StoreIRBuilder, Nested);

  emitOMPRegionLogic(OMPRegionFn, StoreIRBuilder, AllocaIRBuilder, &LoopEntry,
                     ParamToAllocaMap, Nested);

  // StoreIRBuilder.CreateRetVoid();

  AllocaInsertPt->eraseFromParent();
}
*/

void emitOMPRegionFn(
    Function *OMPRegionFn, Function *ForkedFn, Function *ContFn,
    ArrayRef<Argument *> ForkedFnArgs,
    ArrayRef<Argument *> ContFnArgs, bool Nested) {
  auto *Module = OMPRegionFn->getParent();
  auto &Context = Module->getContext();
  auto *EntryBB = BasicBlock::Create(Context, "entry", OMPRegionFn, nullptr);
  auto Int32Ty = Type::getInt32Ty(Context);
  Value *Undef = UndefValue::get(Int32Ty);
  auto *AllocaInsertPt = new BitCastInst(Undef, Int32Ty, "allocapt", EntryBB);
  IRBuilder<> AllocaIRBuilder(EntryBB,
                              ((Instruction *)AllocaInsertPt)->getIterator());

  IRBuilder<> StoreIRBuilder(EntryBB);
  auto ParamToAllocaMap =
      emitImplicitArgs(OMPRegionFn, AllocaIRBuilder, StoreIRBuilder, Nested);

  emitOMPRegionLogic(OMPRegionFn, StoreIRBuilder, AllocaIRBuilder, ForkedFn,
                     ContFn, ParamToAllocaMap, ForkedFnArgs, ContFnArgs,
                     Nested);

  StoreIRBuilder.CreateRetVoid();

  AllocaInsertPt->eraseFromParent();
}

/// Creates a struct that contains elements corresponding to the arguments
/// of \param F.
StructType *createSharedsTy(Function *F) {
  LLVMContext &C = F->getParent()->getContext();
  auto FnParams = F->getFunctionType()->params();

  if (FnParams.size() == 0) {
    return StructType::create("anon", Type::getInt8Ty(C), (llvm::Type*)nullptr);
  }

  return StructType::create(FnParams, "anon");
}

/// Creates some data structures that are needed for the actual task work. It
/// then calls into emitProxyTaskFunction which starts  code generation for the
/// task.
///
/// \param Caller the function from which we wish to spawn the OMP task.
///
/// \param ForkedFn the function that contains the work to be done by the
/// spawned task.
///
/// \param LoadedCapturedArgs the values to be passed to the spawned task.
///
/// \returns the allocated OMP task object.
Value *emitTaskInit(Function *Caller,
                                     IRBuilder<> &CallerIRBuilder,
                                     IRBuilder<> &CallerAllocaIRBuilder,
                                     Function *ForkedFn,
                                     ArrayRef<Value *> LoadedCapturedArgs) {
  auto *M = Caller->getParent();
  DataLayout DL(M);
  LLVMContext &C = M->getContext();
  auto *SharedsTy = createSharedsTy(ForkedFn);
  auto *SharedsPtrTy = PointerType::getUnqual(SharedsTy);
  auto *SharedsTySize =
      CallerIRBuilder.getInt64(DL.getTypeAllocSize(SharedsTy));
  emitKmpRoutineEntryT(M);
  auto *KmpTaskTTy = createKmpTaskTTy(M, KmpRoutineEntryPtrTy);
  auto *KmpTaskTWithPrivatesTy = createKmpTaskTWithPrivatesTy(KmpTaskTTy);
  auto *KmpTaskTWithPrivatesPtrTy =
      PointerType::getUnqual(KmpTaskTWithPrivatesTy);
  auto *KmpTaskTWithPrivatesTySize =
      CallerIRBuilder.getInt64(DL.getTypeAllocSize(KmpTaskTWithPrivatesTy));
  auto OutlinedFn = emitTaskOutlinedFunction(M, SharedsPtrTy, ForkedFn);
  auto *TaskPrivatesMapTy = std::next(OutlinedFn->arg_begin(), 3)->getType();
  auto *TaskPrivatesMap =
      ConstantPointerNull::get(cast<PointerType>(TaskPrivatesMapTy));

  auto *TaskEntry = emitProxyTaskFunction(
      KmpTaskTWithPrivatesPtrTy, SharedsPtrTy, OutlinedFn, TaskPrivatesMap);

  // Allocate space to store the captured environment
  auto *AggCaptured =
      CallerAllocaIRBuilder.CreateAlloca(SharedsTy, nullptr, "agg.captured");

  // Store captured arguments into agg.captured
  for (unsigned i = 0; i < LoadedCapturedArgs.size(); ++i) {
    auto *AggCapturedElemPtr = CallerIRBuilder.CreateInBoundsGEP(
        SharedsTy, AggCaptured,
        {CallerIRBuilder.getInt32(0), CallerIRBuilder.getInt32(i)});
    CallerIRBuilder.CreateAlignedStore(
        LoadedCapturedArgs[i], AggCapturedElemPtr,
        DL.getTypeAllocSize(LoadedCapturedArgs[i]->getType()));
  }

  // We only need tied tasks for now and that's what the 1 value is for.
  auto *TaskFlags = CallerIRBuilder.getInt32(1);
  std::vector<Value *> AllocArgs = {
      DefaultOpenMPLocation,
      getThreadID(Caller, CallerIRBuilder),
      TaskFlags,
      KmpTaskTWithPrivatesTySize,
      SharedsTySize,
      CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
          TaskEntry, KmpRoutineEntryPtrTy)};
  auto *NewTask = emitRuntimeCall(
      createRuntimeFunction(OpenMPRuntimeFunction::OMPRTL__kmpc_omp_task_alloc,
                            Caller->getParent()),
      AllocArgs, "", CallerIRBuilder);
  auto *NewTaskNewTaskTTy = CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
      NewTask, KmpTaskTWithPrivatesPtrTy);
  auto *KmpTaskTPtr = CallerIRBuilder.CreateInBoundsGEP(
      KmpTaskTWithPrivatesTy, NewTaskNewTaskTTy,
      {CallerIRBuilder.getInt32(0), CallerIRBuilder.getInt32(0)});
  auto *KmpTaskDataPtr = CallerIRBuilder.CreateInBoundsGEP(
      KmpTaskTTy, KmpTaskTPtr,
      {CallerIRBuilder.getInt32(0), CallerIRBuilder.getInt32(0)});
  auto *KmpTaskData = CallerIRBuilder.CreateAlignedLoad(
      KmpTaskDataPtr,
      DL.getTypeAllocSize(KmpTaskDataPtr->getType()->getPointerElementType()));
  auto *AggCapturedToI8 = CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
      AggCaptured, Type::getInt8PtrTy(C));
  CallerIRBuilder.CreateMemCpy(
      KmpTaskData, AggCapturedToI8,
      CallerIRBuilder.getInt64(DL.getTypeAllocSize(SharedsTy)), 8);

  return NewTask;
}

/// Emits some boilerplate code to kick off the task work and then calls the
/// function that does the actual work.
Function *emitProxyTaskFunction(
    Type *KmpTaskTWithPrivatesPtrTy, Type *SharedsPtrTy, Function *TaskFunction,
    Value *TaskPrivatesMap) {

  auto OutlinedToEntryIt = OutlinedToEntryMap.find(TaskFunction);

  if (OutlinedToEntryIt != OutlinedToEntryMap.end()) {
    return OutlinedToEntryIt->second;
  }

  auto *M = TaskFunction->getParent();
  auto &C = M->getContext();
  auto *Int32Ty = Type::getInt32Ty(C);
  std::vector<Type *> ArgTys = {Int32Ty, KmpTaskTWithPrivatesPtrTy};
  auto *TaskEntryTy = FunctionType::get(Int32Ty, ArgTys, false);
  auto *TaskEntry = Function::Create(TaskEntryTy, GlobalValue::InternalLinkage,
                                     ".omp_task_entry.", M);
  TaskEntry->addFnAttr(Attribute::NoInline);
  TaskEntry->addFnAttr(Attribute::UWTable);
  DataLayout DL(M);
  auto *EntryBB = BasicBlock::Create(C, "entry", TaskEntry, nullptr);

  IRBuilder<> IRBuilder(EntryBB);
  auto *RetValAddr = IRBuilder.CreateAlloca(Int32Ty, nullptr, "retval");
  RetValAddr->setAlignment(DL.getTypeAllocSize(Int32Ty));

  // TODO replace this with a call to startFunction
  auto Args = TaskEntry->args();
  std::vector<Value *> ArgAllocas(TaskEntry->arg_size());
  auto ArgAllocaIt = ArgAllocas.begin();
  for (auto &Arg : Args) {
    auto *ArgAlloca = IRBuilder.CreateAlloca(Arg.getType(), nullptr, "addr");
    ArgAlloca->setAlignment(DL.getTypeAllocSize(Arg.getType()));
    *ArgAllocaIt = ArgAlloca;
    ++ArgAllocaIt;
  }

  ArgAllocaIt = ArgAllocas.begin();
  for (auto &Arg : Args) {
    IRBuilder.CreateAlignedStore(&Arg, *ArgAllocaIt,
                                 DL.getTypeAllocSize(Arg.getType()));
    ++ArgAllocaIt;
  }

  auto *Int8PtrTy = Type::getInt8PtrTy(C);
  auto GtidParam = IRBuilder.CreateAlignedLoad(ArgAllocas[0],
                                               DL.getTypeAllocSize(ArgTys[0]));

  auto TDVal = IRBuilder.CreateAlignedLoad(ArgAllocas[1],
                                           DL.getTypeAllocSize(ArgTys[1]));
  auto TaskTBase = IRBuilder.CreateInBoundsGEP(
      TDVal, {IRBuilder.getInt32(0), IRBuilder.getInt32(0)});
  auto PartIDAddr = IRBuilder.CreateInBoundsGEP(
      TaskTBase, {IRBuilder.getInt32(0), IRBuilder.getInt32(2)});
  auto *SharedsAddr = IRBuilder.CreateInBoundsGEP(
      TaskTBase, {IRBuilder.getInt32(0), IRBuilder.getInt32(0)});
  auto Shareds = IRBuilder.CreateAlignedLoad(
      SharedsAddr,
      DL.getTypeAllocSize(SharedsAddr->getType()->getPointerElementType()));
  auto SharedsParam =
      IRBuilder.CreatePointerBitCastOrAddrSpaceCast(Shareds, SharedsPtrTy);
  auto TDParam =
      IRBuilder.CreatePointerBitCastOrAddrSpaceCast(TDVal, Int8PtrTy);
  auto PrivatesParam = ConstantPointerNull::get(Int8PtrTy);

  Value *TaskParams[] = {GtidParam,       PartIDAddr, PrivatesParam,
                         TaskPrivatesMap, TDParam,    SharedsParam};

  IRBuilder.CreateCall(TaskFunction, TaskParams);
  IRBuilder.CreateRet(IRBuilder.getInt32(0));

  auto &Elem = OutlinedToEntryMap.FindAndConstruct(TaskFunction);
  Elem.second = TaskEntry;

  return TaskEntry;
}

Function *emitTaskOutlinedFunction(Module *M,
                                                    Type *SharedsPtrTy,
                                                    Function *ForkedFn) {
  auto ExtractedToOutlinedIt = ExtractedToOutlinedMap.find(ForkedFn);

  if (ExtractedToOutlinedIt != ExtractedToOutlinedMap.end()) {
    return ExtractedToOutlinedIt->second;
  }

  auto &C = M->getContext();
  DataLayout DL(M);

  auto *VoidTy = Type::getVoidTy(C);
  auto *Int8PtrTy = Type::getInt8PtrTy(C);
  auto *Int32Ty = Type::getInt32Ty(C);
  auto *Int32PtrTy = Type::getInt32PtrTy(C);

  auto *CopyFnTy = FunctionType::get(VoidTy, {Int8PtrTy}, true);
  auto *CopyFnPtrTy = PointerType::getUnqual(CopyFnTy);

  auto *OutlinedFnTy = FunctionType::get(
      VoidTy,
      {Int32Ty, Int32PtrTy, Int8PtrTy, CopyFnPtrTy, Int8PtrTy, SharedsPtrTy},
      false);
  auto *OutlinedFn = Function::Create(
      OutlinedFnTy, GlobalValue::InternalLinkage, ".omp_outlined.", M);
  StringRef ArgNames[] = {".global_tid.", ".part_id.", ".privates.",
                          ".copy_fn.",    ".task_t.",  "__context"};
  int i = 0;
  for (auto &Arg : OutlinedFn->args()) {
    Arg.setName(ArgNames[i]);
    ++i;
  }

  OutlinedFn->setLinkage(GlobalValue::InternalLinkage);
  OutlinedFn->addFnAttr(Attribute::AlwaysInline);
  OutlinedFn->addFnAttr(Attribute::NoUnwind);
  OutlinedFn->addFnAttr(Attribute::UWTable);

  auto ParamToAllocaMap = startFunction(OutlinedFn);
  auto *ContextArg = &*std::prev(OutlinedFn->args().end());
  auto It = ParamToAllocaMap.find(ContextArg);
  assert(It != ParamToAllocaMap.end() && "Argument entry wasn't found");
  auto *ContextAddr = It->second;
  auto *EntryBB = &*OutlinedFn->begin();
  IRBuilder<> IRBuilder(EntryBB->getTerminator());

  // Load the context struct so that we can access the task's accessed data
  auto *Context = IRBuilder.CreateAlignedLoad(
      ContextAddr,
      DL.getTypeAllocSize(ContextAddr->getType()->getPointerElementType()));

  std::vector<Value *> ForkedFnArgs;
  for (unsigned i = 0; i < ForkedFn->getFunctionType()->getNumParams(); ++i) {
    auto *DataAddrEP = IRBuilder.CreateInBoundsGEP(
        Context, {IRBuilder.getInt32(0), IRBuilder.getInt32(i)});
    auto *DataAddr = IRBuilder.CreateAlignedLoad(
        DataAddrEP,
        DL.getTypeAllocSize(DataAddrEP->getType()->getPointerElementType()));
    // auto *Data = IRBuilder.CreateAlignedLoad(
    //     DataAddr,
    //     DL.getTypeAllocSize(DataAddr->getType()->getPointerElementType()));
    ForkedFnArgs.push_back(DataAddr);
  }

  IRBuilder.CreateCall(ForkedFn, ForkedFnArgs);

  auto &Elem = ExtractedToOutlinedMap.FindAndConstruct(ForkedFn);
  Elem.second = OutlinedFn;

  return OutlinedFn;
}

void emitTaskwaitCall(Function *Caller,
                                       IRBuilder<> &CallerIRBuilder,
                                       const DataLayout &DL) {
  std::vector<Value *> Args = {DefaultOpenMPLocation,
                            getThreadID(Caller, CallerIRBuilder)};
  emitRuntimeCall(
      createRuntimeFunction(OpenMPRuntimeFunction::OMPRTL__kmpc_omp_taskwait,
                            Caller->getParent()),
      Args, "", CallerIRBuilder);
}

/// A helper to emit a basic block and transform the builder insertion
/// point to its start.
void emitBlock(Function *F, IRBuilder<> &IRBuilder,
                                BasicBlock *BB, bool IsFinished) {
  auto *CurBB = IRBuilder.GetInsertBlock();
  emitBranch(BB, IRBuilder);

  if (IsFinished && BB->use_empty()) {
    delete BB;
    return;
  }

  if (CurBB && CurBB->getParent())
    F->getBasicBlockList().insertAfter(CurBB->getIterator(), BB);
  else
    F->getBasicBlockList().push_back(BB);
}

void emitBranch(BasicBlock *Target, IRBuilder<> &IRBuilder) {
  auto *CurBB = IRBuilder.GetInsertBlock();

  if (!CurBB || CurBB->getTerminator()) {

  } else {
    IRBuilder.CreateBr(Target);
  }

  IRBuilder.SetInsertPoint(Target);
}

Type *getOrCreateIdentTy(Module *M) {
  if (M->getTypeByName("ident_t") == nullptr) {
    auto *Int32Ty = Type::getInt32Ty(M->getContext());
    auto *Int8PtrTy = Type::getInt8PtrTy(M->getContext());
    IdentTy = StructType::create("ident_t", Int32Ty /* reserved_1 */,
                                 Int32Ty /* flags */, Int32Ty /* reserved_2 */,
                                 Int32Ty /* reserved_3 */,
                                 Int8PtrTy /* psource */, (llvm::Type*)nullptr);
  }

  return IdentTy;
}

Type *createKmpTaskTTy(Module *M,
                                        PointerType *KmpRoutineEntryPtrTy) {
  auto &C = M->getContext();
  auto *KmpCmplrdataTy =
      StructType::create("kmp_cmplrdata_t", KmpRoutineEntryPtrTy, (llvm::Type*)nullptr);
  auto *KmpTaskTTy = StructType::create(
      "kmp_task_t", Type::getInt8PtrTy(C), KmpRoutineEntryPtrTy,
      Type::getInt32Ty(C), KmpCmplrdataTy, KmpCmplrdataTy, (llvm::Type*)nullptr);

  return KmpTaskTTy;
}

Type *createKmpTaskTWithPrivatesTy(Type *KmpTaskTTy) {
  auto *KmpTaskTWithPrivatesTy =
      StructType::create("kmp_task_t_with_privates", KmpTaskTTy, (llvm::Type*)nullptr);
  return KmpTaskTWithPrivatesTy;
}

void emitKmpRoutineEntryT(Module *M) {
  if (!KmpRoutineEntryPtrTy) {
    // Build typedef kmp_int32 (* kmp_routine_entry_t)(kmp_int32, void *); type.
    auto &C = M->getContext();
    auto *Int32Ty = Type::getInt32Ty(C);
    std::vector<Type *> KmpRoutineEntryTyArgs = {Int32Ty, Type::getInt8PtrTy(C)};
    KmpRoutineEntryPtrTy = PointerType::getUnqual(
        FunctionType::get(Int32Ty, KmpRoutineEntryTyArgs, false));
  }
}

DenseMap<Argument *, Value *> startFunction(Function *Fn) {
  auto *M = Fn->getParent();
  auto &C = M->getContext();
  DataLayout DL(M);
  DenseMap<Argument *, Value *> ParamToAllocaMap;
  auto *EntryBB = BasicBlock::Create(C, "entry", Fn, nullptr);
  IRBuilder<> IRBuilder(EntryBB);
  auto *RetTy = Fn->getReturnType();
  AllocaInst *RetValAddr = nullptr;
  if (!RetTy->isVoidTy()) {
    RetValAddr = IRBuilder.CreateAlloca(RetTy, nullptr, "retval");
    RetValAddr->setAlignment(DL.getTypeAllocSize(RetTy));
  }

  auto Args = Fn->args();
  std::vector<Value *> ArgAllocas(Fn->arg_size());
  auto ArgAllocaIt = ArgAllocas.begin();
  for (auto &Arg : Args) {
    auto *ArgAlloca =
        IRBuilder.CreateAlloca(Arg.getType(), nullptr, Arg.getName() + ".addr");
    ArgAlloca->setAlignment(DL.getTypeAllocSize(Arg.getType()));
    *ArgAllocaIt = ArgAlloca;
    auto &ArgToAllocaIt = ParamToAllocaMap.FindAndConstruct(&Arg);
    ArgToAllocaIt.second = ArgAlloca;
    ++ArgAllocaIt;
  }

  ArgAllocaIt = ArgAllocas.begin();
  for (auto &Arg : Args) {
    IRBuilder.CreateAlignedStore(&Arg, *ArgAllocaIt,
                                 DL.getTypeAllocSize(Arg.getType()));
    ++ArgAllocaIt;
  }

  if (RetTy->isVoidTy()) {
    IRBuilder.CreateRetVoid();
  } else {
    auto *RetVal =
        IRBuilder.CreateAlignedLoad(RetValAddr, DL.getTypeAllocSize(RetTy));
    IRBuilder.CreateRet(RetVal);
  }

  return ParamToAllocaMap;
}

Value *getOrCreateDefaultLocation(Module *M) {
  if (DefaultOpenMPPSource == nullptr) {
    const std::string DefaultLocStr = ";unknown;unknown;0;0;;";
    StringRef DefaultLocStrWithNull(DefaultLocStr.c_str(),
                                    DefaultLocStr.size() + 1);
    DataLayout DL(M);
    uint64_t Alignment = DL.getTypeAllocSize(Type::getInt8Ty(M->getContext()));
    Constant *C = ConstantDataArray::getString(M->getContext(),
                                               DefaultLocStrWithNull, false);
    // NOTE Are heap allocations not recommended in general or is it OK here?
    // I couldn't find a way to statically allocate an IRBuilder for a Module!
    auto *GV =
        new GlobalVariable(*M, C->getType(), true, GlobalValue::PrivateLinkage,
                           C, ".str", nullptr, GlobalValue::NotThreadLocal);
    GV->setAlignment(Alignment);
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    DefaultOpenMPPSource = cast<Constant>(GV);
    DefaultOpenMPPSource = ConstantExpr::getBitCast(
        DefaultOpenMPPSource, Type::getInt8PtrTy(M->getContext()));
  }

  if (DefaultOpenMPLocation == nullptr) {
    // Constant *C = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0,
    // true);
    auto *Int32Ty = Type::getInt32Ty(M->getContext());
    std::vector<Constant *> Members = {
        ConstantInt::get(Int32Ty, 0, true), ConstantInt::get(Int32Ty, 2, true),
        ConstantInt::get(Int32Ty, 0, true), ConstantInt::get(Int32Ty, 0, true),
        DefaultOpenMPPSource};
    Constant *C = ConstantStruct::get(IdentTy, Members);
    auto *GV =
        new GlobalVariable(*M, C->getType(), true, GlobalValue::PrivateLinkage,
                           C, "", nullptr, GlobalValue::NotThreadLocal);
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    GV->setAlignment(8);
    DefaultOpenMPLocation = GV;
  }

  return DefaultOpenMPLocation;
}

/*
/// Emits the outlined function corresponding to the parallel task (whehter
/// forked or continuation).
Function *emitTaskFunction(const ParallelRegion &PR,
                                            bool IsForked) {
  auto &F = *PR.getFork().getParent()->getParent();
  auto &M = *(Module *)F.getParent();

  // Generate the name of the outlined function for the task
  auto FName = F.getName();
  auto PRName = PR.getFork().getParent()->getName();
  auto PT = IsForked ? PR.getForkedTask() : PR.getContinuationTask();
  auto PTName = PT.getEntry().getName();
  auto PTFName = FName + "." + PRName + "." + PTName;

  auto PTFunction = (Function *)M.getOrInsertFunction(
      PTFName.str(), Type::getVoidTy(M.getContext()), NULL);

  ParallelTask::VisitorTy Visitor =
      [PTFunction](BasicBlock &BB, const ParallelTask &PT) -> bool {
    BB.removeFromParent();
    BB.insertInto(PTFunction);
    return true;
  };

  PT.visit(Visitor, true);
  auto &LastBB = PTFunction->back();
  assert((dyn_cast<RettachInst>(LastBB.getTerminator()) ||
          dyn_cast<SyncInst>(LastBB.getTerminator())) &&
         "Should have been sync or reattach");
  LastBB.getTerminator()->eraseFromParent();

  BasicBlock *PTFuncExitBB =
      BasicBlock::Create(M.getContext(), "exit", PTFunction, nullptr);
  ReturnInst::Create(M.getContext(), PTFuncExitBB);

  BranchInst::Create(PTFuncExitBB, &LastBB);

  return PTFunction;
}
*/

//##############################################################################

llvm::tapir::OpenMPABI::OpenMPABI() {}

/// \brief Get/Create the worker count for the spawning function.
Value* llvm::tapir::OpenMPABI::GetOrCreateWorker8(Function &F) {
  /*
  // Value* W8 = F.getValueSymbolTable()->lookup(worker8_name);
  // if (W8) return W8;
  IRBuilder<> B(F.getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  Value *P0 = B.CreateCall(CILKRTS_FUNC(get_nworkers, *F.getParent()));
  Value *P8 = B.CreateMul(P0, ConstantInt::get(P0->getType(), 8), worker8_name);
  return P8;
  */
  return nullptr;
}

void llvm::tapir::OpenMPABI::createSync(SyncInst &SI, ValueToValueMapTy &DetachCtxToStackFrame) {
  //TODO
}

Function *llvm::tapir::OpenMPABI::createDetach(DetachInst &detach,
                                   ValueToValueMapTy &DetachCtxToStackFrame,
                                   DominatorTree &DT, AssumptionCache &AC) {
  BasicBlock *detB = detach.getParent();
  Function &F = *(detB->getParent());

  BasicBlock *Spawned  = detach.getDetached();
  BasicBlock *Continue = detach.getContinue();

  Module *M = F.getParent();

  CallInst *cal = nullptr;
  Function *extracted = extractDetachBodyToFunction(detach, DT, AC, &cal);

    // Collect all the BBs of forked and continuation tasks for extraction.
    std::vector<BasicBlock *> RegionBBs;
    // Keep track of which blocks belong to forked and continuation tasks
    // because we are about to replace fork-join instructions by regular
    // branches.
    //std::vector<BasicBlock *> ForkedBBs;
    //std::vector<BasicBlock *> ContBBs;
    //RegionBBs.push_back(NewForkInstBB);

    // Create the outlined function that contains the OpenMP calls required for
    // outermost regions. This corresponds to the "if (omp_get_num_threads() ==
    // 1)" part as indicated by the example in the header file.
    ValueToValueMapTy VMap;
    auto *OMPRegionFn = declareOMPRegionFn(&F, false, VMap);
    // Create the outlined function that contains the OpenMP calls required for
    // nested regions. This corresponds to the "else" part as indicated by the
    // example in the header file.
    ValueToValueMapTy NestedVMap;
    auto *OMPNestedRegionFn = declareOMPRegionFn(&F, true, NestedVMap);

    // Replace ExtractedFnCI with an if-else region that calls the outermost and
    // nested functions.
    replaceExtractedRegionFnCall(cal, OMPRegionFn, OMPNestedRegionFn);

    // Calculates the sub-list of \p VMap's values that should be passed to \p
    // TaskFn.
    auto filterCalledFnArgs = [](Function *ExtractedRegionFn, Function *TaskFn,
                                 ValueToValueMapTy &VMap) {
      std::vector<Argument *> FilteredArgs;

      // Locate the CallInst to TaskFn
      for (auto &BB : *ExtractedRegionFn) {
        if (CallInst *CI = dyn_cast<CallInst>(BB.begin())) {
          if (TaskFn == CI->getCalledFunction()) {
            auto ChildArgIt = CI->arg_begin();

            while (ChildArgIt != CI->arg_end()) {
              for (auto &Arg : ExtractedRegionFn->args()) {
                if (ChildArgIt->get() == &Arg) {
                  FilteredArgs.push_back(dyn_cast<Argument>(VMap[&Arg]));
                  break;
                }
              }

              ++ChildArgIt;
            }

            break;
          }
        }
      }

      return FilteredArgs;
    };

    auto ForkedFnArgs = filterCalledFnArgs(&F, extracted, VMap);
    auto ForkedFnNestedArgs = filterCalledFnArgs(&F, extracted, NestedVMap);
    //auto ContFnArgs = filterCalledFnArgs(RegionFn, ContFn, VMap);
    //auto ContFnNestedArgs = filterCalledFnArgs(RegionFn, ContFn, NestedVMap);

    auto ContFn = nullptr;
    auto ContFnArgs = nullptr;
    auto ContFnNestedArgs = nullptr;

    // Emit the function containing outermost logic.
    emitOMPRegionFn(OMPRegionFn, extracted, ContFn, ForkedFnArgs, ContFnArgs,
                    false);

    // Emit the function containing nested logic.
    emitOMPRegionFn(OMPNestedRegionFn, extracted, ContFn, ForkedFnNestedArgs,
                    ContFnNestedArgs, true);

}

void llvm::tapir::OpenMPABI::preProcessFunction(Function &F) {
  auto M = (Module *)F.getParent();
  getOrCreateIdentTy(M);
  getOrCreateDefaultLocation(M);
}

void llvm::tapir::OpenMPABI::postProcessFunction(Function &F) {
}

void llvm::tapir::OpenMPABI::postProcessHelper(Function &F) {
}
