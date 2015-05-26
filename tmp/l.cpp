#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Constant.h>

#include <llvm/Support/Casting.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Support/raw_ostream.h>

#if LLVM_VERSION_MAJOR<=3 && LLVM_VERSION_MINOR>=6
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#else
#include <llvm/ExecutionEngine/JIT.h>
#endif


#include <llvm/Support/CommandLine.h>

#if LLVM_VERSION_MAJOR<=3 && LLVM_VERSION_MINOR>=5
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/FileSystem.h>
#else
#include <llvm/Analysis/Verifier.h>
#endif

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>

#if LLVM_VERSION_MAJOR<=3 && LLVM_VERSION_MINOR>=4
#include <llvm/IR/LegacyPassManager.h>
#else
#include <llvm/PassManager.h>
#endif

#if LLVM_VERSION_MAJOR<=3 && LLVM_VERSION_MINOR>=5
#include <llvm/IR/CFG.h>
#else
#include <llvm/Support/CFG.h>
#endif

#include <llvm/IR/Module.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/Scalar.h>

#include <cctype>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <memory>

using namespace llvm;

static Module *TheModule;
static IRBuilder<> Builder(getGlobalContext());
static legacy::FunctionPassManager *TheFPM;
static ExecutionEngine *TheExecutionEngine;

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

int main(){
  LLVMContext &Context = getGlobalContext();

	std::unique_ptr<Module*> lmod_owner(make_unique<llvm::Module>("Main", Context));
	TheModule = lmod_owner.get();

        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();


        std::string ErrStr;
        auto engineBuilder = new
#if LLVM_VERSION_MAJOR<=3 && LLVM_VERSION_MINOR<=5
                        llvm::EngineBuilder(TheModule)
#else
        llvm::EngineBuilder(std::move(lmod_owner))
#endif
        ;
        TheExecutionEngine = engineBuilder->setErrorStr(&ErrStr)
#if LLVM_VERSION_MAJOR<=3 && LLVM_VERSION_MINOR>=6
        .setMCJITMemoryManager(make_unique<llvm::SectionMemoryManager>())
#endif
        .create();

  if (!TheExecutionEngine) {
    fprintf(stderr, "Could not create ExecutionEngine: %s\n", ErrStr.c_str());
    exit(1);
  }

  legacy::FunctionPassManager OurFPM(TheModule);

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

  // Print out all of the generated code.
  TheModule->dump();

}
