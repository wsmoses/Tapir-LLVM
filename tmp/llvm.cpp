#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Scalar.h"
#include <cctype>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
using namespace llvm;


static Module *TheModule;
static IRBuilder<> Builder(getGlobalContext());
static legacy::FunctionPassManager *TheFPM;
static ExecutionEngine *TheExecutionEngine;

int main() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
  LLVMContext &Context = getGlobalContext();

  // Make the module, which holds all the code.
  std::unique_ptr<Module> Owner = make_unique<Module>("my cool jit", Context);
  TheModule = Owner.get();

  // Create the JIT.  This takes ownership of the module.
  std::string ErrStr;
  TheExecutionEngine =
      EngineBuilder(std::move(Owner))
          .setErrorStr(&ErrStr)
          .setMCJITMemoryManager(llvm::make_unique<SectionMemoryManager>())
          .create();
  if (!TheExecutionEngine) {
    fprintf(stderr, "Could not create ExecutionEngine: %s\n", ErrStr.c_str());
    exit(1);
  }

  legacy::FunctionPassManager OurFPM(TheModule);

  // Set up the optimizer pipeline.  Start with registering info about how the
  // target lays out data structures.
  TheModule->setDataLayout(*TheExecutionEngine->getDataLayout());
  // Provide basic AliasAnalysis support for GVN.
  OurFPM.add(createBasicAliasAnalysisPass());
  // Promote allocas to registers.
  OurFPM.add(createPromoteMemoryToRegisterPass());
  // Do simple "peephole" optimizations and bit-twiddling optzns.
  OurFPM.add(createInstructionCombiningPass());
  // Reassociate expressions.
  OurFPM.add(createReassociatePass());
  // Eliminate Common SubExpressions.
  OurFPM.add(createGVNPass());
  // Simplify the control flow graph (deleting unreachable blocks, etc).
  OurFPM.add(createCFGSimplificationPass());

  OurFPM.doInitialization();

  // Set the global so the code gen can use this.
  TheFPM = &OurFPM;

  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(getGlobalContext()), std::vector<Type*>(), false);

  Function *FOO = Function::Create(FT, Function::ExternalLinkage, "foo", TheModule);

  Function *SYNC = Function::Create(FT, Function::ExternalLinkage, "cilk_sync", TheModule);

  
  Function *F =
      Function::Create(FT, Function::ExternalLinkage, "main", TheModule);

  auto START = llvm::BasicBlock::Create(Context,"entry", F);

  auto SPAWNED = llvm::BasicBlock::Create(Context,"spawned", F);
  auto CONT = llvm::BasicBlock::Create(Context,"cont", F);

  Builder.SetInsertPoint(START);

  Builder.CreateCall(FOO);
  
  Builder.Insert(SpawnInst::Create(CONT,SPAWNED));

  Builder.SetInsertPoint(SPAWNED);
  Builder.CreateCall(llvm::Intrinsic::getDeclaration(TheModule,llvm::Intrinsic::trap,llvm::SmallVector<llvm::Type*,0>(0) ) );


  Builder.SetInsertPoint(CONT);
  Builder.CreateCall(SYNC);
  Builder.CreateRetVoid();
  // Print out all of the generated code.
  F->dump();
  fflush(0);
  return 0;
}
