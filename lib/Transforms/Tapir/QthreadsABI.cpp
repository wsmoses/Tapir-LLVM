//===- QthreadsABI.cpp - Lower Tapir into Qthreads runtime system calls -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the QthreadsABI interface, which is used to convert Tapir
// instructions -- detach, reattach, and sync -- to calls into the Qthreads
// runtime system.  
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/QthreadsABI.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "qthreadsabi"

typedef uint64_t aligned_t; 
typedef aligned_t (*qthread_f)(void *arg); 
typedef int (qthread_fork_copyargs)(qthread_f f, const void *arg, size_t arg_size, aligned_t *ret); 
typedef int (qthread_readFF)(aligned_t *syncvar, aligned_t *result); 
typedef int (qthread_initialize)(); 
typedef unsigned short (qthread_num_workers)(); 

#define QTHREAD_FUNC(name, CGF) get_qthread_##name(CGF)

#define DEFAULT_GET_QTHREAD_FUNC(name)                                  \
  static Function *get_qthread_##name(Module& M) {         \
    return cast<Function>(M.getOrInsertFunction(            \
                                          "qthread_"#name,            \
                                          TypeBuilder<qthread_##name, false>::get(M.getContext()) \
							)); \
  }

// TODO: replace macros with something better
DEFAULT_GET_QTHREAD_FUNC(num_workers)
DEFAULT_GET_QTHREAD_FUNC(fork_copyargs)
DEFAULT_GET_QTHREAD_FUNC(initialize)
DEFAULT_GET_QTHREAD_FUNC(readFF)

QthreadsABI::QthreadsABI() { }
QthreadsABI::~QthreadsABI() { }

static const StringRef worker8_name = "qthread_nworker8";

/// \brief Get/Create the worker count for the spawning function.
Value *QthreadsABI::GetOrCreateWorker8(Function &F) {
  IRBuilder<> B(F.getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  Value *P0 = B.CreateCall(QTHREAD_FUNC(num_workers, *F.getParent()));
  Value *P8 = B.CreateMul(P0, ConstantInt::get(P0->getType(), 8), worker8_name);
  return P8;
}

//TODO: Make it work for multiple
void QthreadsABI::createSync(SyncInst &SI, ValueToValueMapTy &DetachCtxToStackFrame) {
  IRBuilder<> builder(&SI); 
  auto F = SI.getParent()->getParent(); 
  auto M = F->getParent();
  auto& C = M->getContext(); 
  auto null = Constant::getNullValue(Type::getInt64PtrTy(C)); 
  std::vector<Value *> args = {null, null}; //TODO: get feb pointer here
  auto readff = QTHREAD_FUNC(readFF, *M); 
  builder.CreateCall(readff, args);
  BranchInst *PostSync = BranchInst::Create(SI.getSuccessor(0));
  ReplaceInstWithInst(&SI, PostSync);
}

// Adds entry basic blocks to body of extracted, replacing extracted, and adds
// necessary code to call, i.e. storing arguments in struct
Function* formatFunctionToQthreadF(Function* extracted, CallInst* cal){
  std::vector<Value*> LoadedCapturedArgs;
  for(auto& a:cal->arg_operands()) {
    LoadedCapturedArgs.push_back(a);
  }

  Module *M = extracted->getParent(); 
  auto& C = M->getContext(); 
  DataLayout DL(M);
  IRBuilder<> CallerIRBuilder(cal);

  auto FnParams = extracted->getFunctionType()->params();
  StructType *ArgsTy = StructType::create(FnParams, "anon");
  auto *ArgsPtrTy = PointerType::getUnqual(ArgsTy);

  auto *OutlinedFnTy = FunctionType::get(
      Type::getInt64Ty(C),
      {Type::getInt8PtrTy(C)},
      false);

  auto *OutlinedFn = Function::Create(
      OutlinedFnTy, GlobalValue::InternalLinkage, ".qthreads_outlined.", M);
  OutlinedFn->addFnAttr(Attribute::AlwaysInline);
  OutlinedFn->addFnAttr(Attribute::NoUnwind);
  OutlinedFn->addFnAttr(Attribute::UWTable);

  StringRef ArgNames[] = {".args"};
  std::vector<Value*> out_args;
  for (auto &Arg : OutlinedFn->args()) {
    Arg.setName(ArgNames[out_args.size()]);
    out_args.push_back(&Arg);
  }

  // Entry Code
  auto *EntryBB = BasicBlock::Create(C, "entry", OutlinedFn, nullptr);
  IRBuilder<> EntryBuilder(EntryBB);
  auto argStructPtr = EntryBuilder.CreateBitCast(out_args[0], ArgsPtrTy); 
  ValueToValueMapTy valmap;

  unsigned int argc = 0;
  for (auto& arg : extracted->args()) {
    auto *DataAddrEP = EntryBuilder.CreateStructGEP(ArgsTy, argStructPtr, argc); 
    auto *DataAddr = EntryBuilder.CreateAlignedLoad(
        DataAddrEP,
        DL.getTypeAllocSize(DataAddrEP->getType()->getPointerElementType()));
    valmap.insert(std::pair<Value*,Value*>(&arg,DataAddr));
    argc++;
  }

  // Replace return values with return 0
  SmallVector< ReturnInst *,5> retinsts;
  CloneFunctionInto(OutlinedFn, extracted, valmap, true, retinsts);
  EntryBuilder.CreateBr(OutlinedFn->getBasicBlockList().getNextNode(*EntryBB));

  for (auto& ret : retinsts) {
    auto retzero = ReturnInst::Create(C, ConstantInt::get(Type::getInt64Ty(C), 0)); 
    ReplaceInstWithInst(ret, retzero);
  }

  // Caller code
  auto callerArgStruct = CallerIRBuilder.CreateAlloca(ArgsTy); 
  unsigned int cArgc = 0;
  for (auto& arg : LoadedCapturedArgs) {
    auto *DataAddrEP = CallerIRBuilder.CreateStructGEP(ArgsTy, callerArgStruct, cArgc); 
    auto *DataAddr = CallerIRBuilder.CreateAlignedStore(
        LoadedCapturedArgs[cArgc], DataAddrEP,
        DL.getTypeAllocSize(LoadedCapturedArgs[cArgc]->getType()));
    cArgc++;
  }

  assert(argc == cArgc && "Wrong number of arguments passed to outlined function"); 

  auto outlinedFnPtr = CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
          OutlinedFn, TypeBuilder<qthread_f, false>::get(M->getContext())); 
  auto argSize = ConstantInt::get(Type::getInt64Ty(C), DL.getTypeAllocSize(ArgsTy)); 
  auto ret = CallerIRBuilder.CreateAlloca(Type::getInt64Ty(C)); 
  auto argsStructVoidPtr = CallerIRBuilder.CreateBitCast(callerArgStruct, Type::getInt8PtrTy(C)); 
  std::vector<Value *> callerArgs = { outlinedFnPtr, argsStructVoidPtr, argSize, ret}; 
  CallerIRBuilder.CreateCall(QTHREAD_FUNC(fork_copyargs, *M), callerArgs); 

  cal->eraseFromParent();
  extracted->eraseFromParent();

  DEBUG(OutlinedFn->dump()); 

  return OutlinedFn; 
}

Function *QthreadsABI::createDetach(DetachInst &detach,
				    ValueToValueMapTy &DetachCtxToStackFrame,
				    DominatorTree &DT, AssumptionCache &AC) {
  BasicBlock *detB = detach.getParent();
  Function &F = *(detB->getParent());

  BasicBlock *Spawned  = detach.getDetached();
  BasicBlock *Continue = detach.getContinue();

  Module *M = F.getParent();

  CallInst *cal = nullptr;
  Function *extracted = extractDetachBodyToFunction(detach, DT, AC, &cal);
  extracted = formatFunctionToQthreadF(extracted, cal); 

  // Replace the detach with a branch to the continuation.
  BranchInst *ContinueBr = BranchInst::Create(Continue);
  ReplaceInstWithInst(&detach, ContinueBr);

  // Rewrite phis in the detached block.
  {
    BasicBlock::iterator BI = Spawned->begin();
    while (PHINode *P = dyn_cast<PHINode>(BI)) {
      P->removeIncomingValue(detB);
      ++BI;
    }
  }

  return extracted;
}

void QthreadsABI::preProcessFunction(Function &F) {}

void QthreadsABI::postProcessFunction(Function &F) {}

void QthreadsABI::postProcessHelper(Function &F) {}

bool QthreadsABI::processMain(Function &F) {
  IRBuilder<> start(F.getEntryBlock().getFirstNonPHIOrDbg());
  auto m = start.CreateCall(QTHREAD_FUNC(initialize, *F.getParent()));
  m->moveBefore(F.getEntryBlock().getTerminator());
  return true;
}

