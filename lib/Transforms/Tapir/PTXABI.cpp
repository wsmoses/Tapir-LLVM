/**
  ***************************************************************************
  * Copyright (c) 2017, Los Alamos National Security, LLC.
  * All rights reserved.
  *
  *  Copyright 2010. Los Alamos National Security, LLC. This software was
  *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
  *  Alamos National Laboratory (LANL), which is operated by Los Alamos
  *  National Security, LLC for the U.S. Department of Energy. The
  *  U.S. Government has rights to use, reproduce, and distribute this
  *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
  *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
  *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
  *  derivative works, such modified software should be clearly marked,
  *  so as not to confuse it with the version available from LANL.
  *
  *  Additionally, redistribution and use in source and binary forms,
  *  with or without modification, are permitted provided that the
  *  following conditions are met:
  *
  *    * Redistributions of source code must retain the above copyright
  *      notice, this list of conditions and the following disclaimer.
  *
  *    * Redistributions in binary form must reproduce the above
  *      copyright notice, this list of conditions and the following
  *      disclaimer in the documentation and/or other materials provided
  *      with the distribution.
  *
  *    * Neither the name of Los Alamos National Security, LLC, Los
  *      Alamos National Laboratory, LANL, the U.S. Government, nor the
  *      names of its contributors may be used to endorse or promote
  *      products derived from this software without specific prior
  *      written permission.
  *
  *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
  *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
  *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
  *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
  *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
  *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
  *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
  *  SUCH DAMAGE.
  *
  ***************************************************************************/

#include "llvm/Transforms/Tapir/PTXABI.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/IR/LegacyPassManager.h"

#include <iostream>
#include <set>
#include <sstream>

#define np(X)                                                            \
 std::cout << __FILE__ << ":" << __LINE__ << ": " << __PRETTY_FUNCTION__ \
           << ": " << #X << " = " << (X) << std::endl

#include <iostream>
#include <set>
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "ptxabi"

namespace {

  template<class F>
  Function* getFunction(Module& M, const char* name){
    return cast<Function>(M.getOrInsertFunction(name,
      TypeBuilder<F, false>::get(M.getContext())));
  }

  template<class B>
  Value* convertInteger(B& b, Value* from, Value* to, const std::string& name){
    auto ft = dyn_cast<IntegerType>(from->getType());
    assert(ft && "expected from type as integer type");

    auto tt = dyn_cast<IntegerType>(to->getType());
    assert(tt && "expected to type as integer type");

    if(ft->getBitWidth() > tt->getBitWidth()){
      return b.CreateTrunc(from, tt, name);
    }
    else if(ft->getBitWidth() < tt->getBitWidth()){
      return b.CreateZExt(from, tt, name);
    }

    return from;
  }

} // namespace


//##############################################################################

PTXABI::PTXABI() {}

/// \brief Get/Create the worker count for the spawning function.
Value *PTXABI::GetOrCreateWorker8(Function &F) {
  Module *M = F.getParent();
  LLVMContext& C = M->getContext();
  return ConstantInt::get(C, APInt(16, 8));
}

void PTXABI::createSync(SyncInst &SI, ValueToValueMapTy &DetachCtxToStackFrame) {
}

Function *PTXABI::createDetach(DetachInst &detach,
                               ValueToValueMapTy &DetachCtxToStackFrame,
                               DominatorTree &DT, AssumptionCache &AC) {
  //TODO nicely replace with serializeDetach
  BasicBlock *detB = detach.getParent();

  BasicBlock *Spawned  = detach.getDetached();
  BasicBlock *Continue = detach.getContinue();

  CallInst *cal = nullptr;
  Function *extracted = extractDetachBodyToFunction(detach, DT, AC, &cal);

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

void PTXABI::preProcessFunction(Function &F) {
}

void PTXABI::postProcessFunction(Function &F) {
}

void PTXABI::postProcessHelper(Function &F) {
}

bool PTXABI::processMain(Function &F) {
  return true;
}

bool PTXABILoopSpawning::processLoop(){
  Loop *L = OrigLoop;

  // L->dumpVerbose();

  //  code generation is currently limited to a simple canonical loop structure
  //  whereby we make the following assumptions and check assertions below
  //  soon we will expand this extraction mechanism to handle more complex
  //  loops

  using TypeVec = std::vector<Type*>;
  using ValueVec = std::vector<Value*>;

  LLVMContext& c = L->getHeader()->getContext();

  IRBuilder<> b(c);

  Type* voidTy = Type::getVoidTy(c);
  IntegerType* i8Ty = Type::getInt8Ty(c);
  IntegerType* i16Ty = Type::getInt16Ty(c);
  IntegerType* i32Ty = Type::getInt32Ty(c);
  IntegerType* i64Ty = Type::getInt64Ty(c);
  PointerType* voidPtrTy = Type::getInt8PtrTy(c);

  //  and LLVM transformation is able in some cases to transform the loop to
  //  contain a phi node that exists at the entry block

  PHINode* loopNode = L->getCanonicalInductionVariable();
  assert(loopNode && "expected canonical loop");

  //  only handle loops where the induction variable is initialized to a constant

  Value* loopStart = loopNode->getIncomingValue(0);
  assert(loopStart && "expected canonical loop start");

  auto cs = dyn_cast<ConstantInt>(loopStart);
  bool startsAtZero = cs && cs->isZero();

  BasicBlock* exitBlock = L->getUniqueExitBlock();
  assert(exitBlock && "expected canonical exit block");

  // and assume that a branch instruction exists here

  BasicBlock* branchBlock = exitBlock->getSinglePredecessor();
  assert(branchBlock && "expected canonical branch block");

  BranchInst* endBranch = dyn_cast<BranchInst>(branchBlock->getTerminator());
  assert(endBranch && "expected canonical end branch instruction");

  //  get the branch condition in order to extract the end loop value
  //  which we also currently assume is constant

  Value* endBranchCond = endBranch->getCondition();
  CmpInst* cmp = dyn_cast<CmpInst>(endBranchCond);
  assert(cmp && "expected canonical comparison instruction");

  Value* loopEnd = cmp->getOperand(1);
  assert(loopEnd && "expected canonical loop end");

  BasicBlock* latchBlock = L->getLoopLatch();
  Instruction* li = latchBlock->getFirstNonPHI();
  unsigned op = li->getOpcode();
  assert(op == Instruction::Add || op == Instruction::Sub &&
         "expected add or sub in loop latch");
  assert(li->getOperand(0)== loopNode);
  Value* stride = li->getOperand(1);
  cs = dyn_cast<ConstantInt>(stride);
  bool isUnitStride = cs && cs->isOne();

  BasicBlock* entryBlock = L->getBlocks()[0];

  Function* hostFunc = entryBlock->getParent();

  Module& hostModule = *hostFunc->getParent();

  // assume a detach exists here  and this basic block contains the body
  //  of the kernel function we will be generating

  DetachInst* detach = dyn_cast<DetachInst>(entryBlock->getTerminator());
  assert(detach && "expected canonical loop entry detach");

  BasicBlock* Body = detach->getDetached();

  // extract the externally defined variables
  // these will be passed in as CUDA arrays

  std::set<Value*> values;
  values.insert(loopNode);

  std::set<Value*> extValues;

  for(Instruction& ii : *Body){
    if(dyn_cast<ReattachInst>(&ii)){
      continue;
    }

    for(Use& u : ii.operands()){
      Value* v = u.get();

      if(isa<Constant>(v)){
        continue;
      }

      if(values.find(v) == values.end()){
        extValues.insert(v);
      }
    }

    values.insert(&ii);
  }

  TypeVec paramTypes;
  paramTypes.push_back(i64Ty);
  paramTypes.push_back(i64Ty);
  paramTypes.push_back(i64Ty);

  for(Value* v : extValues){
    if(auto pt = dyn_cast<PointerType>(v->getType())){
      if(auto at = dyn_cast<ArrayType>(pt->getElementType())){
        paramTypes.push_back(PointerType::get(at->getElementType(), 0));
      }
      else{
        paramTypes.push_back(pt);
      }
    }
    else{
      v->dump();
      assert(false && "expected a pointer or array type");
    }
  }

  // create the GPU function

  FunctionType* funcTy = FunctionType::get(voidTy, paramTypes, false);

  Module ptxModule("ptxModule", c);

  // each kernel function is assigned a unique ID by which the kernel
  // entry point function is named e.g. run0 for kernel ID 0

  size_t kernelRunId = nextKernelId_++;

  std::stringstream kstr;
  kstr << "run" << kernelRunId;

  Function* f = Function::Create(funcTy,
    Function::ExternalLinkage, kstr.str().c_str(), &ptxModule);

  // the first parameter defines the extent of the index space
  // i.e. number of threads to launch
  auto aitr = f->arg_begin();
  aitr->setName("runSize");
  Value* runSizeParam = aitr;
  ++aitr;

  aitr->setName("runStart");
  Value* runStartParam = aitr;
  ++aitr;

  aitr->setName("runStride");
  Value* runStrideParam = aitr;
  ++aitr;

  std::map<Value*, Value*> m;

  // set and parameter names and map values to be replaced

  size_t i = 0;

  for(Value* v : extValues){
    std::stringstream sstr;
    sstr << "arg" << i;

    m[v] = aitr;
    aitr->setName(sstr.str());
    ++aitr;
    ++i;
  }

  // create the entry block which will be used to compute the thread ID
  // and simply return if the thread ID is beyond the run size

  BasicBlock* br = BasicBlock::Create(c, "entry", f);

  b.SetInsertPoint(br);

  using SREGFunc = uint32_t();

  // calls to NVPTX intrinsics to get the thread index, block size,
  // and grid dimensions

  Value* threadIdx = b.CreateCall(getFunction<SREGFunc>(ptxModule,
    "llvm.nvvm.read.ptx.sreg.tid.x"));

  Value* blockIdx = b.CreateCall(getFunction<SREGFunc>(ptxModule,
    "llvm.nvvm.read.ptx.sreg.ctaid.x"));

  Value* blockDim = b.CreateCall(getFunction<SREGFunc>(ptxModule,
    "llvm.nvvm.read.ptx.sreg.ntid.x"));

  Value* threadId =
    b.CreateAdd(threadIdx, b.CreateMul(blockIdx, blockDim), "threadId");

  // convert the thread ID into the proper integer type of the loop variable

  threadId = convertInteger(b, threadId, loopNode, "threadId");

  if(!isUnitStride){
    threadId = b.CreateMul(threadId, runStrideParam);
  }

  if(!startsAtZero){
    threadId = b.CreateAdd(threadId, runStartParam);
  }

  // return block to exit if thread ID is greater than or equal to run size

  BasicBlock* rb = BasicBlock::Create(c, "exit", f);
  BasicBlock* bb = BasicBlock::Create(c, "body", f);

  Value* cond = b.CreateICmpUGE(threadId, runSizeParam);
  b.CreateCondBr(cond, rb, bb);

  b.SetInsertPoint(rb);
  b.CreateRetVoid();

  b.SetInsertPoint(bb);

  // map the thread ID into the new values as we clone the instructions
  // of the function

  m[loopNode] = threadId;

  BasicBlock::InstListType& il = bb->getInstList();

  // clone instructions of the body basic block,  remapping values as needed

  std::set<Value*> extReads;
  std::set<Value*> extWrites;
  std::map<Value*, Value*> extVars;

  for(Instruction& ii : *Body){
    if(dyn_cast<ReattachInst>(&ii)){
      continue;
    }

    // determine if we are reading or writing the external variables
    // i.e. those passed as CUDA arrays

    Instruction* ic = ii.clone();

    if(auto li = dyn_cast<LoadInst>(&ii)){
      Value* v = li->getPointerOperand();
      auto itr = extVars.find(v);
      if(itr != extVars.end()){
        extReads.insert(itr->second);
      }
    }
    else if(auto si = dyn_cast<StoreInst>(&ii)){
      Value* v = si->getPointerOperand();
      auto itr = extVars.find(v);
      if(itr != extVars.end()){
        extWrites.insert(itr->second);
      }
    }
    // if this is a GEP  into one of the external variables then keep track of
    // which external variable it originally came from
    else if(auto gi = dyn_cast<GetElementPtrInst>(&ii)){
      Value* v = gi->getPointerOperand();
      if(extValues.find(v) != extValues.end()){
        extVars[gi] = v;
        if(isa<ArrayType>(gi->getSourceElementType())){
          auto cgi = dyn_cast<GetElementPtrInst>(ic);
          cgi->setSourceElementType(m[v]->getType());
        }
      }
    }

    // remap values as we are cloning the instructions

    for(auto& itr : m){
      ic->replaceUsesOfWith(itr.first, itr.second);
    }

    il.push_back(ic);
    m[&ii] = ic;
  }

  b.CreateRetVoid();

  // add the necessary NVPTX to mark the global function

  NamedMDNode* annotations =
    ptxModule.getOrInsertNamedMetadata("nvvm.annotations");

  SmallVector<Metadata*, 3> av;

  av.push_back(ValueAsMetadata::get(f));
  av.push_back(MDString::get(ptxModule.getContext(), "kernel"));
  av.push_back(ValueAsMetadata::get(llvm::ConstantInt::get(i32Ty, 1)));

  annotations->addOperand(MDNode::get(ptxModule.getContext(), av));

  // remove the basic blocks corresponding to the original LLVM loop

  BasicBlock* predecessor = L->getLoopPreheader();
  entryBlock->removePredecessor(predecessor);
  BasicBlock* successor = exitBlock->getSingleSuccessor();

  BasicBlock* hostBlock = BasicBlock::Create(c, "host.block", hostFunc);

  b.SetInsertPoint(predecessor->getTerminator());
  b.CreateBr(hostBlock);
  predecessor->getTerminator()->removeFromParent();

  successor->removePredecessor(exitBlock);

  {
    std::set<BasicBlock*> visited;
    visited.insert(exitBlock);

    std::vector<BasicBlock*> next;
    next.push_back(entryBlock);

    while(!next.empty()){
      BasicBlock* b = next.back();
      next.pop_back();

      for(BasicBlock* bn : b->getTerminator()->successors()){
        if(visited.find(bn) == visited.end()){
          next.push_back(bn);
        }
      }

      b->dropAllReferences();
      b->removeFromParent();
      visited.insert(b);
    }
  }

  exitBlock->dropAllReferences();
  exitBlock->removeFromParent();

  // find the NVPTX module pass which will create the PTX code

  const Target* target = nullptr;

  for(TargetRegistry::iterator itr =  TargetRegistry::targets().begin(),
      itrEnd =  TargetRegistry::targets().end(); itr != itrEnd; ++itr){
    if(std::string(itr->getName()) == "nvptx64"){
      target = &*itr;
      break;
    }
  }

  assert(target && "failed to find NVPTX target");

  Triple triple(sys::getDefaultTargetTriple());
  triple.setArch(Triple::nvptx64);

  // TODO:  the version of LLVM that we are using currently only supports
  // up to SM_60 – we need SM_70 for Volta architectures

  TargetMachine* targetMachine =
      target->createTargetMachine(triple.getTriple(),
                                  //"sm_35",
                                  //"sm_70",
                                  "sm_60",
                                  "",
                                  TargetOptions(),
                                  Reloc::Static,
                                  CodeModel::Default,
                                  CodeGenOpt::Aggressive);

  DataLayout layout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:"
    "64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:"
    "64:64-v128:128:128-n16:32:64");

  ptxModule.setDataLayout(layout);

  legacy::PassManager* passManager = new legacy::PassManager;

  passManager->add(createVerifierPass());

  // add in our optimization passes

  passManager->add(createInstructionCombiningPass());
  passManager->add(createReassociatePass());
  passManager->add(createGVNPass());
  passManager->add(createCFGSimplificationPass());
  passManager->add(createSLPVectorizerPass());
  passManager->add(createBreakCriticalEdgesPass());
  passManager->add(createConstantPropagationPass());
  passManager->add(createDeadInstEliminationPass());
  passManager->add(createDeadStoreEliminationPass());
  passManager->add(createInstructionCombiningPass());
  passManager->add(createCFGSimplificationPass());

  SmallVector<char, 65536> buf;
  raw_svector_ostream ostr(buf);

  bool fail =
  targetMachine->addPassesToEmitFile(*passManager,
                                     ostr,
                                     TargetMachine::CGFT_AssemblyFile,
                                     false);

  assert(!fail && "failed to emit PTX");

  passManager->run(ptxModule);

  delete passManager;

  std::string ptx = ostr.str().str();

  Constant* pcs = ConstantDataArray::getString(c, ptx);

  // create a global string to hold the PTX code

  GlobalVariable* ptxGlobal =
    new GlobalVariable(hostModule,
                       pcs->getType(),
                       true,
                       GlobalValue::PrivateLinkage,
                       pcs,
                       "ptx");

  Value* kernelId = ConstantInt::get(i32Ty, kernelRunId);

  Value* ptxStr = b.CreateBitCast(ptxGlobal, voidPtrTy);

  b.SetInsertPoint(hostBlock);

  // finally, replace where the original loop was with calls to the GPU runtime

  using InitCUDAFunc = void();

  b.CreateCall(getFunction<InitCUDAFunc>(hostModule,
      "__kitsune_cuda_init"), {});

  using InitKernelFunc = void(uint32_t, const char*);

  b.CreateCall(getFunction<InitKernelFunc>(hostModule,
      "__kitsune_gpu_init_kernel"), {kernelId, ptxStr});

  for(Value* v : extValues){
    Value* elementSize;
    Value* vptr;
    Value* fieldName;
    Value* size;

    // TODO: fix
    // this is a temporary hack to get the size of the field
    // it will currently only work for a limited case

    if(auto bc = dyn_cast<BitCastInst>(v)){
      auto ci = dyn_cast<CallInst>(bc->getOperand(0));
      assert(ci && "unable to detect field size");

      Value* bytes = ci->getOperand(0);
      assert(bytes->getType()->isIntegerTy(64));

      auto pt = dyn_cast<PointerType>(v->getType());
      auto it = dyn_cast<IntegerType>(pt->getElementType());
      assert(it && "expected integer type");

      Constant* fn = ConstantDataArray::getString(c, ci->getName());

      GlobalVariable* fieldNameGlobal =
        new GlobalVariable(hostModule,
                           fn->getType(),
                           true,
                           GlobalValue::PrivateLinkage,
                           fn,
                           "field.name");

      fieldName = b.CreateBitCast(fieldNameGlobal, voidPtrTy);

      vptr = b.CreateBitCast(v, voidPtrTy);

      elementSize = ConstantInt::get(i32Ty, it->getBitWidth()/8);

      size = b.CreateUDiv(bytes, ConstantInt::get(i64Ty, it->getBitWidth()/8));
    }
    else if(auto ai = dyn_cast<AllocaInst>(v)){
      Constant* fn = ConstantDataArray::getString(c, ai->getName());

      GlobalVariable* fieldNameGlobal =
        new GlobalVariable(hostModule,
                           fn->getType(),
                           true,
                           GlobalValue::PrivateLinkage,
                           fn,
                           "field.name");

      fieldName = b.CreateBitCast(fieldNameGlobal, voidPtrTy);

      vptr = b.CreateBitCast(v, voidPtrTy);

      auto at = dyn_cast<ArrayType>(ai->getAllocatedType());
      assert(at && "expected array type");

      elementSize = ConstantInt::get(i32Ty,
        at->getElementType()->getPrimitiveSizeInBits()/8);

      size = ConstantInt::get(i64Ty, at->getNumElements());
    }

    uint8_t m = 0;
    if(extReads.find(v) != extReads.end()){
      m |= 0b01;
    }

    if(extWrites.find(v) != extWrites.end()){
      m |= 0b10;
    }

    Value* mode = ConstantInt::get(i8Ty, m);

    TypeVec params = {i32Ty, voidPtrTy, voidPtrTy, i32Ty, i64Ty, i8Ty};

    Function* initFieldFunc =
      llvm::Function::Create(FunctionType::get(voidTy, params, false),
                             llvm::Function::ExternalLinkage,
                             "__kitsune_gpu_init_field",
                             &hostModule);

    b.CreateCall(initFieldFunc,
      {kernelId, fieldName, vptr, elementSize, size, mode});
  }

  using SetRunSizeFunc = void(uint32_t, uint64_t, uint64_t, uint64_t);

  Value* runSize = b.CreateSub(loopEnd, loopStart);

  runSize = convertInteger(b, runSize, threadId, "run.size");

  Value* runStart = convertInteger(b, loopStart, threadId, "run.start");

  b.CreateCall(getFunction<SetRunSizeFunc>(hostModule,
    "__kitsune_gpu_set_run_size"), {kernelId, runSize, runStart, runStart});

  using RunKernelFunc = void(uint32_t);

  b.CreateCall(getFunction<RunKernelFunc>(hostModule,
    "__kitsune_gpu_run_kernel"), {kernelId});

  using FinishFunc = void();

  b.CreateCall(getFunction<FinishFunc>(hostModule,
    "__kitsune_gpu_finish"), {});

  b.CreateBr(successor);

  // hostModule.dump();

  // ptxModule.dump();

  return true;
}

bool llvm::PTXABI::processLoop(LoopSpawningHints LSH, LoopInfo &LI, ScalarEvolution &SE, DominatorTree &DT,
                               AssumptionCache &AC, OptimizationRemarkEmitter &ORE) {
    if (LSH.getStrategy() != LoopSpawningHints::ST_GPU)
        return false;

    Loop* L = LSH.TheLoop;
    DEBUG(dbgs() << "LS: Hints dictate GPU spawning.\n");
    {
      DebugLoc DLoc = L->getStartLoc();
      BasicBlock *Header = L->getHeader();
      PTXABILoopSpawning DLS(L, SE, LI, DT, AC, ORE);
      if (DLS.processLoop()) {
        DEBUG({
            if (verifyFunction(*L->getHeader()->getParent())) {
              dbgs() << "Transformed function is invalid.\n";
              return false;
            }
          });
        // Report success.
        ORE.emit(OptimizationRemark(LS_NAME, "GPUSpawning", DLoc, Header)
                 << "spawning iterations using direct gpu mapping");
        return true;
      } else {
        // Report failure.
        ORE.emit(OptimizationRemarkMissed(LS_NAME, "NoGPUSpawning", DLoc,
                                          Header)
                 << "cannot spawn iterations using direct gpu mapping");

        ORE.emit(DiagnosticInfoOptimizationFailure(
              DEBUG_TYPE, "FailedRequestedGPUSpawning",
              L->getStartLoc(), L->getHeader())
          << "Tapir loop not transformed: "
          << "failed to use direct gpu mapping");
        return false;
      }
    }

  return false;
}
