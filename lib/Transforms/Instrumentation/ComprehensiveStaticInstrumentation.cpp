//===-- ComprehensiveStaticInstrumentation.cpp - CSI compiler pass --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of CSI, a framework that provides comprehensive static
// instrumentation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/ComprehensiveStaticInstrumentation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/CSI.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "csi"

static cl::opt<bool>
    ClInstrumentFuncEntryExit("csi-instrument-func-entry-exit", cl::init(true),
                              cl::desc("Instrument function entry and exit"),
                              cl::Hidden);
static cl::opt<bool>
    ClInstrumentLoops("csi-instrument-loops", cl::init(true),
                      cl::desc("Instrument loops"), cl::Hidden);
static cl::opt<bool>
    ClInstrumentBasicBlocks("csi-instrument-basic-blocks", cl::init(true),
                            cl::desc("Instrument basic blocks"), cl::Hidden);
static cl::opt<bool>
    ClInstrumentMemoryAccesses("csi-instrument-memory-accesses", cl::init(true),
                               cl::desc("Instrument memory accesses"),
                               cl::Hidden);
static cl::opt<bool> ClInstrumentCalls("csi-instrument-function-calls",
                                       cl::init(true),
                                       cl::desc("Instrument function calls"),
                                       cl::Hidden);
static cl::opt<bool> ClInstrumentAtomics("csi-instrument-atomics",
                                         cl::init(true),
                                         cl::desc("Instrument atomics"),
                                         cl::Hidden);
static cl::opt<bool> ClInstrumentMemIntrinsics(
    "csi-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);
static cl::opt<bool> ClInstrumentTapir("csi-instrument-tapir", cl::init(true),
                                       cl::desc("Instrument tapir constructs"),
                                       cl::Hidden);
static cl::opt<bool> ClInstrumentAllocas("csi-instrument-alloca",
                                         cl::init(true),
                                         cl::desc("Instrument allocas"),
                                         cl::Hidden);
static cl::opt<bool>
    ClInstrumentAllocFns("csi-instrument-allocfn", cl::init(true),
                         cl::desc("Instrument allocation functions"),
                         cl::Hidden);
static cl::opt<CSIOptions::ArithmeticType>
ClInstrumentArithmetic("csi-instrument-arithmetic",
                       cl::init(CSIOptions::ArithmeticType::All), cl::Hidden,
                       cl::desc("Instrument arithmetic operations"),
                       cl::values(
                           clEnumValN(CSIOptions::ArithmeticType::None, "none",
                                      "Disable instrumentation of arithmetic"),
                           clEnumValN(CSIOptions::ArithmeticType::FP, "fp",
                                      "Instrument floating-point arithmetic"),
                           clEnumValN(CSIOptions::ArithmeticType::Int, "int",
                                      "Instrument integer arithmetic"),
                           clEnumValN(CSIOptions::ArithmeticType::All, "all",
                                      "Instrument all arithmetic")));
static cl::opt<bool>
    ClInstrumentInputs("csi-instrument-inputs", cl::init(true),
                         cl::desc("Instrument data-flow inputs"),
                         cl::Hidden);

static cl::opt<bool> ClInterpose("csi-interpose", cl::init(true),
                                 cl::desc("Enable function interpositioning"),
                                 cl::Hidden);

static cl::opt<std::string> ClToolBitcode(
    "csi-tool-bitcode", cl::init(""),
    cl::desc("Path to the tool bitcode file for compile-time instrumentation"),
    cl::Hidden);

static cl::opt<std::string>
    ClRuntimeBitcode("csi-runtime-bitcode", cl::init(""),
                     cl::desc("Path to the CSI runtime bitcode file for "
                              "optimized compile-time instrumentation"),
                     cl::Hidden);

static cl::opt<std::string> ClConfigurationFilename(
    "csi-config-filename", cl::init(""),
    cl::desc("Path to the configuration file for surgical instrumentation"),
    cl::Hidden);

static cl::opt<InstrumentationConfigMode> ClConfigurationMode(
    "csi-config-mode", cl::init(InstrumentationConfigMode::WHITELIST),
    cl::values(clEnumValN(InstrumentationConfigMode::WHITELIST, "whitelist",
                          "Use configuration file as a whitelist"),
               clEnumValN(InstrumentationConfigMode::BLACKLIST, "blacklist",
                          "Use configuration file as a blacklist")),
    cl::desc("Specifies how to interpret the configuration file"), cl::Hidden);

namespace {

static CSIOptions OverrideFromCL(CSIOptions Options) {
  Options.InstrumentFuncEntryExit = ClInstrumentFuncEntryExit;
  Options.InstrumentLoops = ClInstrumentLoops;
  Options.InstrumentBasicBlocks = ClInstrumentBasicBlocks;
  Options.InstrumentMemoryAccesses = ClInstrumentMemoryAccesses;
  Options.InstrumentCalls = ClInstrumentCalls;
  Options.InstrumentAtomics = ClInstrumentAtomics;
  Options.InstrumentMemIntrinsics = ClInstrumentMemIntrinsics;
  Options.InstrumentTapir = ClInstrumentTapir;
  Options.InstrumentAllocas = ClInstrumentAllocas;
  Options.InstrumentAllocFns = ClInstrumentAllocFns;
  Options.InstrumentArithmetic = ClInstrumentArithmetic;
  Options.InstrumentInputs = ClInstrumentInputs;
  return Options;
}

/// The Comprehensive Static Instrumentation pass.
/// Inserts calls to user-defined hooks at predefined points in the IR.
struct ComprehensiveStaticInstrumentationLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid.

  ComprehensiveStaticInstrumentationLegacyPass(
      const CSIOptions &Options = OverrideFromCL(CSIOptions()))
      : ModulePass(ID), Options(Options) {
    initializeComprehensiveStaticInstrumentationLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }
  StringRef getPassName() const override {
    return "ComprehensiveStaticInstrumentation";
  }
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  CSIOptions Options;
}; // struct ComprehensiveStaticInstrumentation
} // anonymous namespace

char ComprehensiveStaticInstrumentationLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(ComprehensiveStaticInstrumentationLegacyPass, "csi",
                      "ComprehensiveStaticInstrumentation pass", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(ComprehensiveStaticInstrumentationLegacyPass, "csi",
                    "ComprehensiveStaticInstrumentation pass", false, false)

ModulePass *llvm::createComprehensiveStaticInstrumentationLegacyPass() {
  return new ComprehensiveStaticInstrumentationLegacyPass();
}
ModulePass *llvm::createComprehensiveStaticInstrumentationLegacyPass(
    const CSIOptions &Options) {
  return new ComprehensiveStaticInstrumentationLegacyPass(Options);
}

/// Return the first DILocation in the given basic block, or nullptr
/// if none exists.
static const DILocation *getFirstDebugLoc(const BasicBlock &BB) {
  for (const Instruction &Inst : BB)
    if (const DILocation *Loc = Inst.getDebugLoc())
      return Loc;

  return nullptr;
}

/// Set DebugLoc on the call instruction to a CSI hook, based on the
/// debug information of the instrumented instruction.
static void setInstrumentationDebugLoc(Instruction *Instrumented,
                                       Instruction *Call) {
  DISubprogram *Subprog = Instrumented->getFunction()->getSubprogram();
  if (Subprog) {
    if (Instrumented->getDebugLoc()) {
      Call->setDebugLoc(Instrumented->getDebugLoc());
    } else {
      LLVMContext &C = Instrumented->getFunction()->getParent()->getContext();
      Call->setDebugLoc(DILocation::get(C, 0, 0, Subprog));
    }
  }
}

/// Set DebugLoc on the call instruction to a CSI hook, based on the
/// debug information of the instrumented instruction.
static void setInstrumentationDebugLoc(BasicBlock &Instrumented,
                                       Instruction *Call) {
  DISubprogram *Subprog = Instrumented.getParent()->getSubprogram();
  if (Subprog) {
    if (const DILocation *FirstDebugLoc = getFirstDebugLoc(Instrumented))
      Call->setDebugLoc(FirstDebugLoc);
    else {
      LLVMContext &C = Instrumented.getParent()->getParent()->getContext();
      Call->setDebugLoc(DILocation::get(C, 0, 0, Subprog));
    }
  }
}

/// Set DebugLoc on the call instruction to a CSI hook, based on the
/// debug information of the instrumented instruction.
static void setInstrumentationDebugLoc(Function &Instrumented,
                                       Instruction *Call) {
  DISubprogram *Subprog = Instrumented.getSubprogram();
  if (Subprog) {
    LLVMContext &C = Instrumented.getParent()->getContext();
    Call->setDebugLoc(DILocation::get(C, 0, 0, Subprog));
  }
}

bool CSIImpl::callsPlaceholderFunction(const Instruction &I) {
  if (isa<DbgInfoIntrinsic>(I))
    return true;

  if (isDetachedRethrow(&I))
    return true;

  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I))
    if (Intrinsic::syncregion_start == II->getIntrinsicID() ||
        Intrinsic::lifetime_start == II->getIntrinsicID() ||
        Intrinsic::lifetime_end == II->getIntrinsicID())
      return true;

  return false;
}

bool CSIImpl::run() {
  initializeCsi();

  for (GlobalValue &G : M.globals()) {
    if (isa<Function>(G)) continue;
    if (G.isDeclaration()) continue;
    if (G.getName().startswith("__csi")) continue;
    // Assign an ID for this global.
    csi_id_t LocalId = GlobalFED.add(G);
    GlobalOffsetMap[G.getName()] = LocalId;
  }

  for (Function &F : M)
    instrumentFunction(F);

  collectUnitFEDTables();
  collectUnitSizeTables();
  finalizeCsi();

  linkInToolFromBitcode(ClToolBitcode);
  linkInToolFromBitcode(ClRuntimeBitcode);

  return true; // We always insert the unit constructor.
}

Constant *ForensicTable::getObjectStrGV(Module &M, StringRef Str,
                                        const Twine GVName) {
  LLVMContext &C = M.getContext();
  IntegerType *Int32Ty = IntegerType::get(C, 32);
  Constant *Zero = ConstantInt::get(Int32Ty, 0);
  Value *GepArgs[] = {Zero, Zero};
  if (Str.empty())
    return ConstantPointerNull::get(
        PointerType::get(IntegerType::get(C, 8), 0));

  Constant *NameStrConstant = ConstantDataArray::getString(C, Str);
  GlobalVariable *GV = M.getGlobalVariable((GVName + Str).str(), true);
  if (GV == NULL) {
    GV = new GlobalVariable(M, NameStrConstant->getType(), true,
                            GlobalValue::PrivateLinkage, NameStrConstant,
                            GVName + Str, nullptr,
                            GlobalVariable::NotThreadLocal, 0);
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  }
  assert(GV);
  return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
}

ForensicTable::ForensicTable(Module &M, StringRef BaseIdName,
                             StringRef TableName)
    : TableName(TableName) {
  LLVMContext &C = M.getContext();
  IntegerType *Int64Ty = IntegerType::get(C, 64);
  IdCounter = 0;
  BaseId = M.getGlobalVariable(BaseIdName, true);
  if (NULL == BaseId)
    BaseId = new GlobalVariable(M, Int64Ty, false, GlobalValue::InternalLinkage,
                                ConstantInt::get(Int64Ty, 0), BaseIdName);
  assert(BaseId);
}

csi_id_t ForensicTable::getId(const Value *V) {
  if (!ValueToLocalIdMap.count(V))
    ValueToLocalIdMap[V] = IdCounter++;
  assert(ValueToLocalIdMap.count(V) && "Value not in ID map.");
  return ValueToLocalIdMap[V];
}

Value *ForensicTable::localToGlobalId(csi_id_t LocalId,
                                      IRBuilder<> &IRB) const {
  if (CsiUnknownId == LocalId)
    return IRB.getInt64(CsiUnknownId);

  assert(BaseId);
  LLVMContext &C = IRB.getContext();
  LoadInst *Base = IRB.CreateLoad(BaseId);
  MDNode *MD = llvm::MDNode::get(C, None);
  Base->setMetadata(LLVMContext::MD_invariant_load, MD);
  Value *Offset = IRB.getInt64(LocalId);
  return IRB.CreateAdd(Base, Offset);
}

Value *ForensicTable::localToGlobalId(Value *LocalId,
                                      IRBuilder<> &IRB) const {
  assert(BaseId);
  LLVMContext &C = IRB.getContext();
  LoadInst *Base = IRB.CreateLoad(BaseId);
  MDNode *MD = llvm::MDNode::get(C, None);
  Base->setMetadata(LLVMContext::MD_invariant_load, MD);

  return IRB.CreateSelect(
      IRB.CreateICmp(ICmpInst::ICMP_NE, LocalId, IRB.getInt64(CsiUnknownId)),
      IRB.CreateAdd(Base, LocalId), LocalId);
}

csi_id_t SizeTable::add(const BasicBlock &BB) {
  csi_id_t ID = getId(&BB);
  // Count the LLVM IR instructions
  int32_t NonEmptyIRSize = 0;
  for (const Instruction &I : BB) {
    if (isa<PHINode>(I))
      continue;
    if (CSIImpl::callsPlaceholderFunction(I))
      continue;
    NonEmptyIRSize++;
  }
  add(ID, BB.size(), NonEmptyIRSize);
  return ID;
}

PointerType *SizeTable::getPointerType(LLVMContext &C) {
  return PointerType::get(getSizeStructType(C), 0);
}

StructType *SizeTable::getSizeStructType(LLVMContext &C) {
  return StructType::get(
      /* FullIRSize */ IntegerType::get(C, 32),
      /* NonEmptyIRSize */ IntegerType::get(C, 32));
}

void SizeTable::add(csi_id_t ID, int32_t FullIRSize, int32_t NonEmptyIRSize) {
  assert(NonEmptyIRSize <= FullIRSize && "Broken basic block IR sizes");
  assert(LocalIdToSizeMap.find(ID) == LocalIdToSizeMap.end() &&
         "ID already exists in FED table.");
  LocalIdToSizeMap[ID] = {FullIRSize, NonEmptyIRSize};
}

Constant *SizeTable::insertIntoModule(Module &M) const {
  LLVMContext &C = M.getContext();
  StructType *TableType = getSizeStructType(C);
  IntegerType *Int32Ty = IntegerType::get(C, 32);
  Constant *Zero = ConstantInt::get(Int32Ty, 0);
  Value *GepArgs[] = {Zero, Zero};
  SmallVector<Constant *, 1> TableEntries;

  for (csi_id_t LocalID = 0; LocalID < IdCounter; ++LocalID) {
    const SizeInformation &E = LocalIdToSizeMap.find(LocalID)->second;
    Constant *FullIRSize = ConstantInt::get(Int32Ty, E.FullIRSize);
    Constant *NonEmptyIRSize = ConstantInt::get(Int32Ty, E.NonEmptyIRSize);
    // The order of arguments to ConstantStruct::get() must match the
    // sizeinfo_t type in csi.h.
    TableEntries.push_back(
        ConstantStruct::get(TableType, FullIRSize, NonEmptyIRSize));
  }

  ArrayType *TableArrayType = ArrayType::get(TableType, TableEntries.size());
  Constant *Table = ConstantArray::get(TableArrayType, TableEntries);
  GlobalVariable *GV =
      new GlobalVariable(M, TableArrayType, false, GlobalValue::InternalLinkage,
                         Table, TableName);
  return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
}

csi_id_t FrontEndDataTable::add(const Function &F) {
  csi_id_t ID = getId(&F);
  add(ID, F.getSubprogram());
  return ID;
}

csi_id_t FrontEndDataTable::add(const BasicBlock &BB) {
  csi_id_t ID = getId(&BB);
  add(ID, getFirstDebugLoc(BB));
  return ID;
}

csi_id_t FrontEndDataTable::add(const Instruction &I,
                                const StringRef &RealName) {
  csi_id_t ID = getId(&I);
  StringRef Name = RealName;
  if (Name == "") {
    // Look for a llvm.dbg.declare intrinsic.
    TinyPtrVector<DbgInfoIntrinsic *> DbgDeclares =
      FindDbgAddrUses(const_cast<Instruction *>(&I));
    if (!DbgDeclares.empty()) {
      auto *LV = DbgDeclares.front()->getVariable();
      add(ID, (int32_t)LV->getLine(), -1,
          LV->getFilename(), LV->getDirectory(), LV->getName());
      return ID;
    }

    // Examine the llvm.dbg.value intrinsics for this object.
    SmallVector<DbgValueInst *, 1> DbgValues;
    findDbgValues(DbgValues, const_cast<Instruction *>(&I));
    for (auto *DVI : DbgValues) {
      auto *LV = DVI->getVariable();
      if (LV->getName() != "") {
        Name = LV->getName();
        break;
      }
    }
  }
  add(ID, I.getDebugLoc(), Name);
  return ID;
}

csi_id_t FrontEndDataTable::add(Value &V) {
  csi_id_t ID = getId(&V);
  // Look for a llvm.dbg.declare intrinsic.
  TinyPtrVector<DbgInfoIntrinsic *> DbgDeclares = FindDbgAddrUses(&V);
  if (!DbgDeclares.empty()) {
    auto *LV = DbgDeclares.front()->getVariable();
    add(ID, (int32_t)LV->getLine(), -1,
        LV->getFilename(), LV->getDirectory(), LV->getName());
    return ID;
  }

  // Examine the llvm.dbg.value intrinsics for this object.
  SmallVector<DbgValueInst *, 1> DbgValues;
  findDbgValues(DbgValues, &V);
  for (auto *DVI : DbgValues) {
    auto *LV = DVI->getVariable();
    if (LV->getName() != "") {
      add(ID, (int32_t)LV->getLine(), -1,
          LV->getFilename(), LV->getDirectory(), LV->getName());
      return ID;
    }
  }

  add(ID);
  return ID;
}

csi_id_t FrontEndDataTable::add(const GlobalValue &Val) {
  csi_id_t ID = getId(&Val);
  // If the underlying object is a global variable, get that variable's
  // debug information.
  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(&Val)) {
    SmallVector<DIGlobalVariableExpression *, 1> DbgGVExprs;
    GV->getDebugInfo(DbgGVExprs);
    for (auto *GVE : DbgGVExprs) {
      auto *DGV = GVE->getVariable();
      if (DGV->getName() != "") {
        add(ID, (int32_t)DGV->getLine(), -1,
            DGV->getFilename(), DGV->getDirectory(), DGV->getName());
        return ID;
      }
    }
  }
  add(ID);
  return ID;
}

PointerType *FrontEndDataTable::getPointerType(LLVMContext &C) {
  return PointerType::get(getSourceLocStructType(C), 0);
}

StructType *FrontEndDataTable::getSourceLocStructType(LLVMContext &C) {
  return StructType::get(
      /* Name */ PointerType::get(IntegerType::get(C, 8), 0),
      /* Line */ IntegerType::get(C, 32),
      /* Column */ IntegerType::get(C, 32),
      /* File */ PointerType::get(IntegerType::get(C, 8), 0));
}

void FrontEndDataTable::add(csi_id_t ID, const DILocation *Loc,
                            const StringRef &RealName) {
  if (Loc) {
    // TODO: Add location information for inlining
    const DISubprogram *Subprog = Loc->getScope()->getSubprogram();
    add(ID, (int32_t)Loc->getLine(), (int32_t)Loc->getColumn(),
        Loc->getFilename(), Loc->getDirectory(),
        RealName == "" ? Subprog->getName() : RealName);
  } else
    add(ID);
}

void FrontEndDataTable::add(csi_id_t ID, const DISubprogram *Subprog) {
  if (Subprog)
    add(ID, (int32_t)Subprog->getLine(), -1, Subprog->getFilename(),
        Subprog->getDirectory(), Subprog->getName());
  else
    add(ID);
}

void FrontEndDataTable::add(csi_id_t ID, int32_t Line, int32_t Column,
                            StringRef Filename, StringRef Directory,
                            StringRef Name) {
  // TODO: This assert is too strong for unwind basic blocks' FED.
  /*assert(LocalIdToSourceLocationMap.find(ID) ==
             LocalIdToSourceLocationMap.end() &&
         "Id already exists in FED table."); */
  LocalIdToSourceLocationMap[ID] = {Name, Line, Column, Filename, Directory};
}

// The order of arguments to ConstantStruct::get() must match the source_loc_t
// type in csi.h.
static void addFEDTableEntries(SmallVectorImpl<Constant *> &FEDEntries,
                               StructType *FedType, Constant *Name,
                               Constant *Line, Constant *Column,
                               Constant *File) {
  FEDEntries.push_back(ConstantStruct::get(FedType, Name, Line, Column, File));
}

Constant *FrontEndDataTable::insertIntoModule(Module &M) const {
  LLVMContext &C = M.getContext();
  StructType *FedType = getSourceLocStructType(C);
  IntegerType *Int32Ty = IntegerType::get(C, 32);
  Constant *Zero = ConstantInt::get(Int32Ty, 0);
  Value *GepArgs[] = {Zero, Zero};
  SmallVector<Constant *, 11> FEDEntries;

  for (csi_id_t LocalID = 0; LocalID < IdCounter; ++LocalID) {
    const SourceLocation &E = LocalIdToSourceLocationMap.find(LocalID)->second;
    Constant *Line = ConstantInt::get(Int32Ty, E.Line);
    Constant *Column = ConstantInt::get(Int32Ty, E.Column);
    Constant *File;
    {
      std::string Filename = E.Filename.str();
      if (!E.Directory.empty())
        Filename = E.Directory.str() + "/" + Filename;
      File = getObjectStrGV(M, Filename, "__csi_unit_filename_");
    }
    Constant *Name = getObjectStrGV(M, E.Name, DebugNamePrefix);
    addFEDTableEntries(FEDEntries, FedType, Name, Line, Column, File);
  }

  ArrayType *FedArrayType = ArrayType::get(FedType, FEDEntries.size());
  Constant *Table = ConstantArray::get(FedArrayType, FEDEntries);
  GlobalVariable *GV =
      new GlobalVariable(M, FedArrayType, false, GlobalValue::InternalLinkage,
                         Table, TableName);
  return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
}

static Function *CreateNullHook(Function *Hook, Module &M) {
  // Set weak linkage on this hook implementation.
  Hook->setLinkage(GlobalValue::WeakAnyLinkage);
  // Generate the body of the null hook, which simply returns.
  BasicBlock *HookEntry = BasicBlock::Create(M.getContext(), "", Hook);
  IRBuilder<> B(HookEntry);
  B.CreateRetVoid();
  return Hook;
}

template<typename... ArgsTy>
static Function *CreateCSIHookWithNull(Module &M, StringRef Name,
                                       Type *RetTy, ArgsTy... Args) {
  if (Function *Hook = M.getFunction(Name))
    return Hook;

  Function *Hook = checkCsiInterfaceFunction(
      M.getOrInsertFunction(Name, RetTy, Args...));

  return CreateNullHook(Hook, M);
}

Function *CSIImpl::getCSIInputHook(Module &M, CSIDataFlowObject Obj,
                                   Type *InputTy) {
  assert(SupportedType(InputTy) &&
         "No basic-block input hook for unsupported input type.");
  Type *OperandCastTy = getOperandCastTy(M, InputTy);

  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *RetType = IRB.getVoidTy();
  Type *IDType = IRB.getInt64Ty();
  Type *ValCatType = IRB.getInt8Ty();
  Type *FlagsType = CsiArithmeticFlags::getType(C);

  switch (Obj) {
  case CSIDataFlowObject::BasicBlock:
    return CreateCSIHookWithNull(
        M, ("__csi_bb_input_" + TypeToStr(InputTy)),
        RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType);
    // return checkCsiInterfaceFunction(
    //     M.getOrInsertFunction(
    //         ("__csi_bb_input_" + TypeToStr(InputTy)),
    //         RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType));
  case CSIDataFlowObject::Call:
    return CreateCSIHookWithNull(
        M, ("__csi_call_input_" + TypeToStr(InputTy)),
        RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType);
    // return checkCsiInterfaceFunction(
    //     M.getOrInsertFunction(
    //         ("__csi_call_input_" + TypeToStr(InputTy)),
    //         RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType));
  case CSIDataFlowObject::Loop:
    return CreateCSIHookWithNull(
        M, ("__csi_loop_input_" + TypeToStr(InputTy)),
        RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType);
    // return checkCsiInterfaceFunction(
    //     M.getOrInsertFunction(
    //         ("__csi_loop_input_" + TypeToStr(InputTy)),
    //         RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType));
  case CSIDataFlowObject::Task:
    return CreateCSIHookWithNull(
        M, ("__csi_task_input_" + TypeToStr(InputTy)),
        RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType);
    // return checkCsiInterfaceFunction(
    //     M.getOrInsertFunction(
    //         ("__csi_task_input_" + TypeToStr(InputTy)),
    //         RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType));
  case CSIDataFlowObject::FunctionEntry:
    return CreateCSIHookWithNull(
        M, ("__csi_func_parameter_" + TypeToStr(InputTy)),
        RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType);
    // return checkCsiInterfaceFunction(
    //     M.getOrInsertFunction(
    //         ("__csi_func_parameter_" + TypeToStr(InputTy)),
    //         RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType));
  case CSIDataFlowObject::FunctionExit:
    return CreateCSIHookWithNull(
        M, ("__csi_return_val_" + TypeToStr(InputTy)),
        RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType);
    // return checkCsiInterfaceFunction(
    //     M.getOrInsertFunction(
    //         ("__csi_return_val_" + TypeToStr(InputTy)),
    //         RetType, IDType, ValCatType, IDType, OperandCastTy, FlagsType));
  case CSIDataFlowObject::LAST_CSIDataFlowObject:
    llvm_unreachable("Unknown CSIDataFlowObject");
  }
}

static bool checkArithmeticType(CSIOptions::ArithmeticType Target,
                                const Type *Ty) {
  if (CSIOptions::ArithmeticType::All == Target)
    return true;

  if (const VectorType *VecTy = dyn_cast<VectorType>(Ty))
    return checkArithmeticType(Target, VecTy->getElementType());
  if (const PointerType *PtrTy = dyn_cast<PointerType>(Ty))
    return checkArithmeticType(Target, PtrTy->getElementType());
  if (const ArrayType *ArrTy = dyn_cast<ArrayType>(Ty))
    return checkArithmeticType(Target, ArrTy->getElementType());

  if (Ty->isFloatingPointTy())
    return (CSIOptions::ArithmeticType::FP == Target);
  if (Ty->isIntegerTy())
    return (CSIOptions::ArithmeticType::Int == Target);

  return false;
}

bool CSIImpl::IsInstrumentedArithmetic(const Instruction *I) {
  if (isa<BinaryOperator>(I) || isa<TruncInst>(I) || isa<ZExtInst>(I) ||
      isa<SExtInst>(I) || isa<FPToUIInst>(I) || isa<FPToSIInst>(I) ||
      isa<UIToFPInst>(I) || isa<SIToFPInst>(I) || isa<FPTruncInst>(I) ||
      isa<FPExtInst>(I) || isa<BitCastInst>(I) || isa<GetElementPtrInst>(I) ||
      isa<IntToPtrInst>(I) || isa<PtrToIntInst>(I) ||
      isa<PHINode>(I) || isa<CmpInst>(I) ||
      isa<InsertElementInst>(I) || isa<ExtractElementInst>(I) ||
      isa<ShuffleVectorInst>(I))
    // TODO: Handle AddrSpaceCast, ExtractValue, InsertValue
    return checkArithmeticType(Options.InstrumentArithmetic, I->getType());
  return false;
}

Function *CSIImpl::getCSIArithmeticHook(Module &M, Instruction *I, bool Before) {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *RetType = IRB.getVoidTy();
  Type *IDType = IRB.getInt64Ty();
  Type *OpcodeType = IRB.getInt8Ty();
  Type *ValCatType = IRB.getInt8Ty();
  Type *FlagsType = CsiArithmeticFlags::getType(C);

  if (isa<BinaryOperator>(I)) {
    Type *ITy = I->getType();
    Type *Ty = getOperandCastTy(M, ITy);
    if (!Ty)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_arithmetic_" + TypeToStr(ITy)).str(),
            RetType, IDType, OpcodeType, ValCatType, IDType, Ty, ValCatType,
            IDType, Ty, FlagsType));
  }

  Type *ITy = I->getType();
  switch (I->getOpcode()) {
  case Instruction::PHI: {
    Type *Ty = getOperandCastTy(M, ITy);
    if (!Ty)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_phi_" + TypeToStr(ITy)),
            RetType, IDType, IDType, IDType, ValCatType, IDType, Ty,
            FlagsType));
  }
  case Instruction::FCmp:
  case Instruction::ICmp: {
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_cmp_" + TypeToStr(IOpTy)).str(),
            RetType, IDType, OpcodeType, ValCatType, IDType, InTy, ValCatType,
            IDType, InTy, FlagsType));
  }
  case Instruction::FPTrunc: {
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_truncate_" + TypeToStr(IOpTy) + "_" + TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::FPExt: {
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_extend_" + TypeToStr(IOpTy) + "_" + TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::Trunc: {
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_truncate_" + TypeToStr(IOpTy) + "_" + TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::ZExt: {
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_zero_extend_" + TypeToStr(IOpTy) + "_" + TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::SExt: {
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_sign_extend_" + TypeToStr(IOpTy) + "_" + TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::FPToUI: {
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_convert_" + TypeToStr(IOpTy) + "_unsigned_" +
             TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::FPToSI: {
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_convert_" + TypeToStr(IOpTy) + "_signed_" +
             TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::UIToFP: {
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_convert_unsigned_" + TypeToStr(IOpTy) + "_" +
             TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::SIToFP: {
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_convert_signed_" + TypeToStr(IOpTy) + "_" +
             TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::BitCast: {
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_bitcast_" + TypeToStr(IOpTy) + "_" + TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::GetElementPtr: {
    // TODO: Generalize the name of this hook to be less LLVM-specific.
    Type *IOpTy = I->getOperand(0)->getType();
    // TODO: Add support for GEP instructions on pointers of vectors.
    if (isa<VectorType>(IOpTy))
      return nullptr;
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    Type *IdxArgType = StructType::get(IRB.getInt8Ty(), // Index ValCat
                                       IRB.getInt64Ty(), // Index operand ID
                                       IRB.getInt64Ty()); // Index operand
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_getelementptr_" + TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy,
            PointerType::get(IdxArgType, 0), IRB.getInt32Ty(), FlagsType));
  }
  case Instruction::IntToPtr: {
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_inttoptr_" + TypeToStr(IOpTy) + "_" +
             TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::PtrToInt: {
    Type *OutTy = getOperandCastTy(M, ITy);
    Type *IOpTy = I->getOperand(0)->getType();
    Type *InTy = getOperandCastTy(M, IOpTy);
    if (!OutTy || !InTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_ptrtoint_" + TypeToStr(IOpTy) + "_" +
             TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, InTy, FlagsType));
  }
  case Instruction::InsertElement: {
    VectorType *VecTy = cast<VectorType>(getOperandCastTy(M, ITy));
    if (!VecTy)
      return nullptr;
    Type *ElTy = VecTy->getElementType();
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_insert_element_" + TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, VecTy, ValCatType, IDType,
            ElTy, ValCatType, IDType, IRB.getInt32Ty(), FlagsType));
  }
  case Instruction::ExtractElement: {
    Type *IOpTy = I->getOperand(0)->getType();
    VectorType *VecTy = cast<VectorType>(getOperandCastTy(M, IOpTy));
    if (!VecTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_extract_element_" + TypeToStr(IOpTy)).str(),
            RetType, IDType, ValCatType, IDType, VecTy, ValCatType, IDType,
            IRB.getInt32Ty(), FlagsType));
  }
  case Instruction::ShuffleVector: {
    Type *IOp0Ty = I->getOperand(0)->getType();
    Type *IOp1Ty = I->getOperand(1)->getType();
    Type *IOp2Ty = I->getOperand(2)->getType();
    VectorType *Vec1Ty = cast<VectorType>(getOperandCastTy(M, IOp0Ty));
    VectorType *Vec2Ty = cast<VectorType>(getOperandCastTy(M, IOp1Ty));
    VectorType *MaskTy = cast<VectorType>(getOperandCastTy(M, IOp2Ty));
    VectorType *ResTy = cast<VectorType>(getOperandCastTy(M, ITy));
    if (!Vec1Ty || !Vec2Ty || !MaskTy || !ResTy)
      return nullptr;
    return checkCsiInterfaceFunction(
        M.getOrInsertFunction(
            ("__csi_" + Twine(Before ? "before" : "after") +
             "_shuffle_" + TypeToStr(IOp0Ty) + "_" + TypeToStr(IOp1Ty) + "_" +
             TypeToStr(ITy)).str(),
            RetType, IDType, ValCatType, IDType, Vec1Ty, ValCatType, IDType,
            Vec2Ty, ValCatType, IDType, MaskTy, FlagsType));
  }
  default:
    dbgs() << "Arithmetic instruction not instrumented: " << *I << "\n";
    return nullptr;
  }
}

Function *CSIImpl::getCSIBuiltinHook(Module &M, CallInst *I, bool Before) {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *RetType = IRB.getVoidTy();
  Type *IDType = IRB.getInt64Ty();
  Type *FuncOpType = IRB.getInt8Ty();
  // Type *OpcodeType = IRB.getInt8Ty();
  Type *ValCatType = IRB.getInt8Ty();
  Type *PropertyTy = CsiCallProperty::getType(C);

  CallSite CS(I);
  // Create the base name of the hook.
  std::string HookName = ("__csi_" + Twine(Before ? "before" : "after") +
                          "_builtin").str();
  std::vector<Type *> paramTy;

  // Add the CSI ID parameter
  paramTy.push_back(IDType);
  // Add the parameter for the code for the builtin operation.
  paramTy.push_back(FuncOpType);

  // Add the return type of the builtin to the hook name.
  Type *OpTy = I->getType();
  if (!SupportedType(OpTy))
    return nullptr;
  HookName += "_" + TypeToStr(OpTy);

  // Augment the hook name and parameters for each argument.
  for (const Value *Arg : CS.args()) {
    Type *InTy = Arg->getType();
    if (!SupportedType(InTy))
      return nullptr;
    // Append the name of this input type to the hook name.
    HookName += "_" + TypeToStr(InTy);
    // Append a value category, CSI ID, and casted input type to the hook
    // parameters.
    Type *CastedTy = getOperandCastTy(M, InTy);
    paramTy.push_back(ValCatType);
    paramTy.push_back(IDType);
    paramTy.push_back(CastedTy);
  }

  // Add the properties parameter.
  paramTy.push_back(PropertyTy);

  // Return the hook.
  return checkCsiInterfaceFunction(
      M.getOrInsertFunction(
          HookName, FunctionType::get(RetType, paramTy, false)));
}

Function *CSIImpl::getMaskedReadWriteHook(Module &M, Instruction *I,
                                          bool Before) {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *RetType = IRB.getVoidTy();
  Type *IDType = IRB.getInt64Ty();
  Type *ValCatType = IRB.getInt8Ty();
  Type *MaskElTy = IRB.getInt8Ty();
  Type *LoadPropertyTy = CsiLoadStoreProperty::getType(C);
  Type *StorePropertyTy = CsiLoadStoreProperty::getType(C);

  IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
  // Create the base name of the hook.
  std::string HookName = ("__csi_" + Twine(Before ? "before" : "after")).str();
  std::vector<Type *> paramTy;

  // Add the CSI ID parameter
  paramTy.push_back(IDType);

  switch (II->getIntrinsicID()) {
  default:
    return nullptr;
    break;
  case Intrinsic::masked_load: {
    VectorType *OpTy = cast<VectorType>(II->getType());
    if (!SupportedType(OpTy))
      return nullptr;
    HookName += "_masked_load_" + TypeToStr(OpTy);
    Type *PtrArgTy = II->getArgOperand(0)->getType();
    Type *MaskArgTy = VectorType::get(MaskElTy, OpTy->getNumElements());
    Type *PassThruTy = OpTy;
    // Add the pointer argument
    paramTy.push_back(PtrArgTy);
    // Add the mask argument
    paramTy.push_back(ValCatType);
    paramTy.push_back(IDType);
    paramTy.push_back(MaskArgTy);
    // Add the pass-through argument
    paramTy.push_back(ValCatType);
    paramTy.push_back(IDType);
    paramTy.push_back(PassThruTy);
    // Add the properties parameter.
    paramTy.push_back(LoadPropertyTy);
    break;
  }
  case Intrinsic::masked_gather: {
    VectorType *OpTy = cast<VectorType>(II->getType());
    if (!SupportedType(OpTy))
      return nullptr;
    HookName += "_masked_gather_" + TypeToStr(OpTy);
    Type *PtrArgTy = II->getArgOperand(0)->getType();
    Type *MaskArgTy = VectorType::get(MaskElTy, OpTy->getNumElements());
    Type *PassThruTy = OpTy;
    // Add the pointer argument
    paramTy.push_back(PtrArgTy);
    // Add the mask argument
    paramTy.push_back(ValCatType);
    paramTy.push_back(IDType);
    paramTy.push_back(MaskArgTy);
    // Add the pass-through argument
    paramTy.push_back(ValCatType);
    paramTy.push_back(IDType);
    paramTy.push_back(PassThruTy);
    // Add the properties parameter.
    paramTy.push_back(LoadPropertyTy);
    break;
  }
  case Intrinsic::masked_store: {
    VectorType *StoreValTy = cast<VectorType>(II->getArgOperand(0)->getType());
    if (!SupportedType(StoreValTy))
      return nullptr;
    HookName += "_masked_store_" + TypeToStr(StoreValTy);
    Type *PtrArgTy = II->getArgOperand(1)->getType();
    Type *MaskArgTy = VectorType::get(MaskElTy, StoreValTy->getNumElements());
    // Add the pointer argument
    paramTy.push_back(PtrArgTy);
    // Add the mask argument
    paramTy.push_back(ValCatType);
    paramTy.push_back(IDType);
    paramTy.push_back(MaskArgTy);
    // Add the store-value argument
    paramTy.push_back(ValCatType);
    paramTy.push_back(IDType);
    paramTy.push_back(StoreValTy);
    // Add the properties parameter.
    paramTy.push_back(StorePropertyTy);
    break;
  }
  case Intrinsic::masked_scatter: {
    VectorType *StoreValTy = cast<VectorType>(II->getArgOperand(0)->getType());
    if (!SupportedType(StoreValTy))
      return nullptr;
    HookName += "_masked_scatter_" + TypeToStr(StoreValTy);
    Type *PtrArgTy = II->getArgOperand(1)->getType();
    Type *MaskArgTy = VectorType::get(MaskElTy, StoreValTy->getNumElements());
    // Add the pointer argument
    paramTy.push_back(PtrArgTy);
    // Add the mask argument
    paramTy.push_back(ValCatType);
    paramTy.push_back(IDType);
    paramTy.push_back(MaskArgTy);
    // Add the store-value argument
    paramTy.push_back(ValCatType);
    paramTy.push_back(IDType);
    paramTy.push_back(StoreValTy);
    // Add the properties parameter.
    paramTy.push_back(StorePropertyTy);
    break;
  }
  }

  // Create the hook.
  Function *Hook = checkCsiInterfaceFunction(
      M.getOrInsertFunction(
          HookName, FunctionType::get(RetType, paramTy, false)));
  switch (II->getIntrinsicID()) {
  case Intrinsic::masked_load:
  case Intrinsic::masked_store:
    Hook->addParamAttr(1, Attribute::ReadOnly);
    break;
  default:
    break;
  }

  return Hook;
}

/// Function entry and exit hook initialization
void CSIImpl::initializeFuncHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *IDType = IRB.getInt64Ty();
  // Initialize function entry hooks
  Type *FuncPropertyTy = CsiFuncProperty::getType(C);
  CsiFuncEntry = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_func_entry", IRB.getVoidTy(), IDType, IDType, IRB.getInt32Ty(),
      FuncPropertyTy));
  // Initialize function exit hooks
  Type *FuncExitPropertyTy = CsiFuncExitProperty::getType(C);
  CsiFuncExit = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_func_exit", IRB.getVoidTy(), IDType, IDType, FuncExitPropertyTy));
}

/// Basic-block hook initialization
void CSIImpl::initializeBasicBlockHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *PropertyTy = CsiBBProperty::getType(C);
  CsiBBEntry = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_bb_entry", IRB.getVoidTy(), IRB.getInt64Ty(), IRB.getInt64Ty(),
      PropertyTy));
  CsiBBExit = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_bb_exit", IRB.getVoidTy(), IRB.getInt64Ty(), PropertyTy));
}

/// Loop hook initialization
void CSIImpl::initializeLoopHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *IDType = IRB.getInt64Ty();
  Type *LoopPropertyTy = CsiLoopProperty::getType(C);
  Type *LoopExitPropertyTy = CsiLoopExitProperty::getType(C);

  CsiBeforeLoop = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_before_loop", IRB.getVoidTy(), IDType, IRB.getInt64Ty(),
      LoopPropertyTy));
  CsiAfterLoop = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_after_loop", IRB.getVoidTy(), IDType, LoopPropertyTy));

  CsiLoopBodyEntry = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_loopbody_entry", IRB.getVoidTy(), IDType, LoopPropertyTy));
  CsiLoopBodyExit = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_loopbody_exit", IRB.getVoidTy(), IDType, IDType,
      LoopExitPropertyTy));
}

// Call-site hook initialization
void CSIImpl::initializeCallsiteHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *IDType = IRB.getInt64Ty();
  Type *PropertyTy = CsiCallProperty::getType(C);
  CsiBeforeCallsite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_call", IRB.getVoidTy(), IDType,
                            /*parent_bb_id*/ IDType, /*callee func_id*/ IDType,
                            PropertyTy));
  CsiAfterCallsite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_call", IRB.getVoidTy(), IDType,
                            IDType, PropertyTy));
}

// Alloca (local variable) hook initialization
void CSIImpl::initializeAllocaHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *IDType = IRB.getInt64Ty();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *PropType = CsiAllocaProperty::getType(C);

  CsiBeforeAlloca = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_before_alloca", IRB.getVoidTy(), IDType, IntptrTy, PropType));
  CsiAfterAlloca = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_alloca", IRB.getVoidTy(), IDType,
                            AddrType, IntptrTy, PropType));
  CsiAfterAlloca->addParamAttr(1, Attribute::ReadOnly);
}

// Non-local-variable allocation/free hook initialization
void CSIImpl::initializeAllocFnHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *RetType = IRB.getVoidTy();
  Type *IDType = IRB.getInt64Ty();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *LargeNumBytesType = IntptrTy;
  Type *AllocFnPropType = CsiAllocFnProperty::getType(C);
  Type *FreePropType = CsiFreeProperty::getType(C);

  CsiBeforeAllocFn = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_before_allocfn", RetType, IDType, LargeNumBytesType,
      LargeNumBytesType, LargeNumBytesType, AddrType, AllocFnPropType));
  CsiBeforeAllocFn->addParamAttr(4, Attribute::ReadOnly);
  CsiAfterAllocFn = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_allocfn", RetType, IDType,
                            /* new ptr */ AddrType,
                            /* size */ LargeNumBytesType,
                            /* num elements */ LargeNumBytesType,
                            /* alignment */ LargeNumBytesType,
                            /* old ptr */ AddrType,
                            /* property */ AllocFnPropType));
  CsiAfterAllocFn->addParamAttr(1, Attribute::ReadOnly);
  CsiAfterAllocFn->addParamAttr(5, Attribute::ReadOnly);

  CsiBeforeFree = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_before_free", RetType, IDType, AddrType, FreePropType));
  CsiBeforeFree->addParamAttr(1, Attribute::ReadOnly);
  CsiAfterFree = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_after_free", RetType, IDType, AddrType, FreePropType));
  CsiAfterFree->addParamAttr(1, Attribute::ReadOnly);
}

// Load and store hook initialization
void CSIImpl::initializeLoadStoreHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *IDType = IRB.getInt64Ty();
  Type *ValCatType = IRB.getInt8Ty();
  Type *LoadPropertyTy = CsiLoadStoreProperty::getType(C);
  Type *StorePropertyTy = CsiLoadStoreProperty::getType(C);
  Type *RetType = IRB.getVoidTy();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *NumBytesType = IRB.getInt32Ty();

  CsiBeforeRead = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_load", RetType, IDType,
                            AddrType, NumBytesType, ValCatType, IDType,
                            LoadPropertyTy));
  CsiBeforeRead->addParamAttr(1, Attribute::ReadOnly);
  CsiAfterRead = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_load", RetType, IDType,
                            AddrType, NumBytesType, ValCatType, IDType,
                            LoadPropertyTy));
  CsiAfterRead->addParamAttr(1, Attribute::ReadOnly);

  CsiBeforeWrite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_store", RetType, IDType,
                            AddrType, NumBytesType, ValCatType, IDType,
                            ValCatType, IDType, StorePropertyTy));
  CsiBeforeWrite->addParamAttr(1, Attribute::ReadOnly);
  CsiAfterWrite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_store", RetType, IDType,
                            AddrType, NumBytesType, ValCatType, IDType,
                            ValCatType, IDType, StorePropertyTy));
  CsiAfterWrite->addParamAttr(1, Attribute::ReadOnly);
}

// Initialization of hooks for LLVM memory intrinsics
void CSIImpl::initializeMemIntrinsicsHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *RetType = IRB.getVoidTy();
  Type *IDType = IRB.getInt64Ty();
  Type *ValCatType = IRB.getInt8Ty();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *NumBytesType = IRB.getInt64Ty();

  // TODO: Propagate alignment, volatile information to hooks.
  CsiBeforeMemset = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_memset", RetType, IDType,
                            AddrType, NumBytesType, ValCatType, IDType,
                            IRB.getInt8Ty()));
  CsiBeforeMemset->addParamAttr(1, Attribute::ReadOnly);
  CsiAfterMemset = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_memset", RetType, IDType,
                            AddrType, NumBytesType, ValCatType, IDType,
                            IRB.getInt8Ty()));
  CsiAfterMemset->addParamAttr(1, Attribute::ReadOnly);

  CsiBeforeMemcpy = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_memcpy", RetType, IDType,
                            AddrType, AddrType, NumBytesType));
  CsiBeforeMemcpy->addParamAttr(1, Attribute::ReadOnly);
  CsiBeforeMemcpy->addParamAttr(2, Attribute::ReadOnly);
  CsiAfterMemcpy = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_memcpy", RetType, IDType,
                            AddrType, AddrType, NumBytesType));
  CsiAfterMemcpy->addParamAttr(1, Attribute::ReadOnly);
  CsiAfterMemcpy->addParamAttr(2, Attribute::ReadOnly);

  CsiBeforeMemmove = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_memmove", RetType, IDType,
                            AddrType, AddrType, NumBytesType));
  CsiBeforeMemmove->addParamAttr(1, Attribute::ReadOnly);
  CsiBeforeMemmove->addParamAttr(2, Attribute::ReadOnly);
  CsiAfterMemmove = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_memmove", RetType, IDType,
                            AddrType, AddrType, NumBytesType));
  CsiAfterMemmove->addParamAttr(1, Attribute::ReadOnly);
  CsiAfterMemmove->addParamAttr(2, Attribute::ReadOnly);
}

// Initialization of Tapir hooks
void CSIImpl::initializeTapirHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *IDType = IRB.getInt64Ty();
  Type *RetType = IRB.getVoidTy();

  // TODO: Can we change the type of the TrackVars variable to be a simple i32?
  // That would allow the optimizer to more generally optimize away the
  // TrackVars stack allocation.

  CsiDetach = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_detach", RetType,
      /* detach_id */ IDType, IntegerType::getInt32Ty(C)->getPointerTo()));
  CsiDetach->addParamAttr(1, Attribute::ReadOnly);
  CsiTaskEntry = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_task", RetType, /* task_id */ IDType,
                            /* detach_id */ IDType));
  CsiTaskExit = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_task_exit", RetType,
                            /* task_exit_id */ IDType,
                            /* task_id */ IDType,
                            /* detach_id */ IDType));
  CsiDetachContinue = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_detach_continue", RetType,
                            /* detach_continue_id */ IDType,
                            /* detach_id */ IDType));
  CsiBeforeSync = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_sync", RetType, IDType,
                            IntegerType::getInt32Ty(C)->getPointerTo()));
  CsiBeforeSync->addParamAttr(1, Attribute::ReadOnly);
  CsiAfterSync = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_sync", RetType, IDType,
                            IntegerType::getInt32Ty(C)->getPointerTo()));
  CsiAfterSync->addParamAttr(1, Attribute::ReadOnly);
}

// Prepare any calls in the CFG for instrumentation, e.g., by making sure any
// call that can throw is modeled with an invoke.
void CSIImpl::setupCalls(Function &F) {
  // We use the EscapeEnumerator's built-in functionality to promote calls to
  // invokes.
  EscapeEnumerator EE(F, "csi.cleanup", true);
  while (EE.Next())
    ;

  // TODO: Split each basic block immediately after each call, to ensure that
  // calls act like terminators?
}

static BasicBlock *SplitOffPreds(BasicBlock *BB,
                                 SmallVectorImpl<BasicBlock *> &Preds,
                                 DominatorTree *DT) {
  if (BB->isLandingPad()) {
    SmallVector<BasicBlock *, 2> NewBBs;
    SplitLandingPadPredecessors(BB, Preds, ".csi-split-lp", ".csi-split",
                                NewBBs, DT);
    return NewBBs[1];
  }

  SplitBlockPredecessors(BB, Preds, ".csi-split", DT);
  return BB;
}

// Setup each block such that all of its predecessors belong to the same CSI ID
// space.
static void setupBlock(BasicBlock *BB, const TargetLibraryInfo *TLI,
                       DominatorTree *DT) {
  if (BB->getUniquePredecessor())
    return;

  SmallVector<BasicBlock *, 4> DetachPreds;
  SmallVector<BasicBlock *, 4> DetRethrowPreds;
  SmallVector<BasicBlock *, 4> SyncPreds;
  SmallVector<BasicBlock *, 4> AllocFnPreds;
  SmallVector<BasicBlock *, 4> InvokePreds;
  bool HasOtherPredTypes = false;
  unsigned NumPredTypes = 0;

  // Partition the predecessors of the landing pad.
  for (BasicBlock *Pred : predecessors(BB)) {
    if (isa<DetachInst>(Pred->getTerminator()))
      DetachPreds.push_back(Pred);
    else if (isDetachedRethrow(Pred->getTerminator()))
      DetRethrowPreds.push_back(Pred);
    else if (isa<SyncInst>(Pred->getTerminator()))
      SyncPreds.push_back(Pred);
    else if (isAllocationFn(Pred->getTerminator(), TLI))
      AllocFnPreds.push_back(Pred);
    else if (isa<InvokeInst>(Pred->getTerminator()))
      InvokePreds.push_back(Pred);
    else
      HasOtherPredTypes = true;
  }

  NumPredTypes = static_cast<unsigned>(!DetachPreds.empty()) +
                 static_cast<unsigned>(!DetRethrowPreds.empty()) +
                 static_cast<unsigned>(!SyncPreds.empty()) +
                 static_cast<unsigned>(!AllocFnPreds.empty()) +
                 static_cast<unsigned>(!InvokePreds.empty()) +
                 static_cast<unsigned>(HasOtherPredTypes);

  BasicBlock *BBToSplit = BB;
  // Split off the predecessors of each type.
  if (!DetachPreds.empty() && NumPredTypes > 1) {
    BBToSplit = SplitOffPreds(BBToSplit, DetachPreds, DT);
    NumPredTypes--;
  }
  if (!DetRethrowPreds.empty() && NumPredTypes > 1) {
    BBToSplit = SplitOffPreds(BBToSplit, DetRethrowPreds, DT);
    NumPredTypes--;
  }
  if (!SyncPreds.empty() && NumPredTypes > 1) {
    BBToSplit = SplitOffPreds(BBToSplit, SyncPreds, DT);
    NumPredTypes--;
  }
  if (!AllocFnPreds.empty() && NumPredTypes > 1) {
    BBToSplit = SplitOffPreds(BBToSplit, AllocFnPreds, DT);
    NumPredTypes--;
  }
  if (!InvokePreds.empty() && NumPredTypes > 1) {
    BBToSplit = SplitOffPreds(BBToSplit, InvokePreds, DT);
    NumPredTypes--;
  }
}

// Setup all basic blocks such that each block's predecessors belong entirely to
// one CSI ID space.
void CSIImpl::setupBlocks(Function &F, const TargetLibraryInfo *TLI,
                          DominatorTree *DT) {
  SmallPtrSet<BasicBlock *, 8> BlocksToSetup;
  for (BasicBlock &BB : F) {
    if (BB.isLandingPad())
      BlocksToSetup.insert(&BB);

    if (InvokeInst *II = dyn_cast<InvokeInst>(BB.getTerminator()))
      BlocksToSetup.insert(II->getNormalDest());
    else if (SyncInst *SI = dyn_cast<SyncInst>(BB.getTerminator()))
      BlocksToSetup.insert(SI->getSuccessor(0));
  }

  for (BasicBlock *BB : BlocksToSetup)
    setupBlock(BB, TLI, DT);
}

int CSIImpl::getNumBytesAccessed(Value *Addr, const DataLayout &DL) {
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize % 8 != 0)
    return -1;
  return TypeSize / 8;
}

void CSIImpl::addLoadStoreInstrumentation(Instruction *I, Function *BeforeFn,
                                          Function *AfterFn, Value *CsiId,
                                          Type *AddrType, Value *Addr,
                                          int NumBytes, Value *StoreValCat,
                                          Value *StoreValID, Value *ObjValCat,
                                          Value *ObjValID,
                                          CsiLoadStoreProperty &Prop) {
  IRBuilder<> IRB(I);
  Value *PropVal = Prop.getValue(IRB);
  if (StoreValCat && StoreValID)
    insertHookCall(I, BeforeFn,
                   {CsiId, IRB.CreatePointerCast(Addr, AddrType),
                    IRB.getInt32(NumBytes), StoreValCat, StoreValID,
                    ObjValCat, ObjValID, PropVal});
  else
    insertHookCall(I, BeforeFn,
                   {CsiId, IRB.CreatePointerCast(Addr, AddrType),
                    IRB.getInt32(NumBytes), ObjValCat, ObjValID, PropVal});

  BasicBlock::iterator Iter = ++I->getIterator();
  IRB.SetInsertPoint(&*Iter);
  if (StoreValCat && StoreValID)
    insertHookCall(&*Iter, AfterFn,
                   {CsiId, IRB.CreatePointerCast(Addr, AddrType),
                    IRB.getInt32(NumBytes), StoreValCat, StoreValID, ObjValCat,
                    ObjValID, PropVal});
  else
    insertHookCall(&*Iter, AfterFn,
                   {CsiId, IRB.CreatePointerCast(Addr, AddrType),
                    IRB.getInt32(NumBytes), ObjValCat, ObjValID, PropVal});
}

void CSIImpl::assignLoadOrStoreID(Instruction *I) {
  if (isa<StoreInst>(I))
    StoreFED.add(*I);
  else if (isa<LoadInst>(I))
    LoadFED.add(*I);
  else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::masked_load:
    case Intrinsic::masked_gather:
      LoadFED.add(*I);
      break;
    case Intrinsic::masked_store:
    case Intrinsic::masked_scatter:
      StoreFED.add(*I);
      break;
    default:
      dbgs() << "Unknown load/store instruction " << *I << "\n";
      break;
    }
  }
}

static bool checkHasOneUse(Instruction *I, LoopInfo &LI) {
  const Loop *DefLoop = LI.getLoopFor(I->getParent());
  unsigned NumUses = 0;
  for (const Use &U : I->uses()) {
    const User *Usr = U.getUser();
    // Ignore users that are not instructions or that don't perform real
    // computation.
    if (!isa<Instruction>(Usr))
      continue;
    const Instruction *UsrI = cast<Instruction>(Usr);
    if (CSIImpl::callsPlaceholderFunction(*UsrI))
      continue;

    // If the user is in a different loop from the instruction, conservatively
    // declare that this instruction has more than one use.
    if (DefLoop != LI.getLoopFor(UsrI->getParent()))
      return false;
    // If we count too many uses, return false.
    if (++NumUses > 1)
      return false;
  }
  return true;
}

static bool checkBBLocal(const Value *V, const BasicBlock &BB) {
  for (const Use &U : V->uses()) {
    const User *Usr = U.getUser();
    // Ignore users that are not instructions or that don't perform real
    // computation.
    if (!isa<Instruction>(Usr))
      continue;
    const Instruction *UsrI = cast<Instruction>(Usr);
    if (CSIImpl::callsPlaceholderFunction(*UsrI))
      continue;

    // If the parent of this instruction does not match BBParent, then we have a
    // non-local use.
    if (&BB != UsrI->getParent())
      return false;
  }
  return true;
}

void CSIImpl::instrumentParams(IRBuilder<> &IRB, Function &F,
                               Value *FuncId) {
  if (!Options.InstrumentInputs)
    return;

  for (Argument &Arg : F.args()) {
    Value *Input = &Arg;
    // Ignore inputs that do not match the type or arithmetic we're
    // instrumenting.
    if (Options.InstrumentArithmetic != CSIOptions::ArithmeticType::All) {
      if ((Options.InstrumentArithmetic != CSIOptions::ArithmeticType::FP) &&
          Input->getType()->isFPOrFPVectorTy())
        continue;
      if ((Options.InstrumentArithmetic != CSIOptions::ArithmeticType::Int) &&
          Input->getType()->isIntOrIntVectorTy())
        continue;
    }
    // Get the input hook we need, based on the input type.
    Type *InputTy = Input->getType();
    if (!SupportedType(InputTy)) {
      // Skip recording inputs for unsupported types
      DEBUG(dbgs() << "[CSI] Skipping unsupported type " << *InputTy << "\n");
      continue;
    }
    Function *InputHook = getCSIInputHook(M, CSIDataFlowObject::FunctionEntry,
                                          InputTy);
    // Get information on this operand.
    std::pair<Value *, Value *> OperandID = getOperandID(Input, IRB);
    // Cast the operand as needed.
    Type *OperandCastTy = getOperandCastTy(M, InputTy);
    Value *CastInput = Input;
    if (OperandCastTy != InputTy)
      CastInput = IRB.CreateZExtOrBitCast(Input, OperandCastTy);
    // TODO: Compute flags.  Not sure what flags to compute.
    CsiArithmeticFlags Flags;
    Value *FlagsVal = Flags.getValue(IRB);
    // Insert the hook call.
    CallInst *Call = IRB.CreateCall(InputHook, {FuncId, OperandID.first,
                                                OperandID.second, CastInput,
                                                FlagsVal});
    setInstrumentationDebugLoc(&*IRB.GetInsertPoint(), (Instruction *)Call);
  }
}

void CSIImpl::instrumentInputs(IRBuilder<> &IRB, CSIDataFlowObject DFObj,
                               Value *DFObjCsiId,
                               const SmallPtrSetImpl<Value *> &Inputs) {
  if (!Options.InstrumentInputs)
    return;

  for (Value *Input : Inputs) {
    // Ignore inputs that do not match the type or arithmetic we're
    // instrumenting.
    if (Options.InstrumentArithmetic != CSIOptions::ArithmeticType::All) {
      if ((Options.InstrumentArithmetic != CSIOptions::ArithmeticType::FP) &&
          Input->getType()->isFPOrFPVectorTy())
        continue;
      if ((Options.InstrumentArithmetic != CSIOptions::ArithmeticType::Int) &&
          Input->getType()->isIntOrIntVectorTy())
        continue;
    }
    // Get the input hook we need, based on the input type.
    Type *InputTy = Input->getType();
    if (!SupportedType(InputTy)) {
      // Skip recording inputs for unsupported types
      DEBUG(dbgs() << "[CSI] Skipping unsupported type " << *InputTy << "\n");
      continue;
    }
    Function *InputHook = getCSIInputHook(M, DFObj, InputTy);
    // Get the CSI ID information for the operand.
    std::pair<Value *, Value *> OperandID = getOperandID(Input, IRB);
    // Cast the operand as needed.
    Type *OperandCastTy = getOperandCastTy(M, InputTy);
    Value *CastInput = Input;
    if (OperandCastTy != InputTy)
      CastInput = IRB.CreateZExtOrBitCast(Input, OperandCastTy);
    // Compute flags.
    CsiArithmeticFlags Flags;
    if (const Instruction *I = dyn_cast<Instruction>(Input))
      Flags.setBBLocal(checkBBLocal(I, (DFObj == CSIDataFlowObject::BasicBlock) ?
                                    *IRB.GetInsertPoint()->getParent() :
                                    *I->getParent()));
    Value *FlagsVal = Flags.getValue(IRB);
    // Insert the hook call.
    CallInst *Call = IRB.CreateCall(InputHook, {DFObjCsiId, OperandID.first,
                                                OperandID.second, CastInput,
                                                FlagsVal});
    setInstrumentationDebugLoc(&*IRB.GetInsertPoint(), (Instruction *)Call);
  }
}

void CSIImpl::instrumentLoadOrStore(Instruction *I, CsiLoadStoreProperty &Prop,
                                    const DataLayout &DL) {
  IRBuilder<> IRB(I);
  bool IsWrite = isa<StoreInst>(I);
  Value *Addr = IsWrite ? cast<StoreInst>(I)->getPointerOperand()
                        : cast<LoadInst>(I)->getPointerOperand();
  int NumBytes = getNumBytesAccessed(Addr, DL);
  Type *AddrType = IRB.getInt8PtrTy();

  if (NumBytes == -1)
    return; // size that we don't recognize

  Value *Obj = GetUnderlyingObject(Addr, DL);
  std::pair<Value *, Value *> ObjID = getOperandID(Obj, IRB);

  if (IsWrite) {
    csi_id_t LocalId = StoreFED.lookupId(I);
    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    StoreInst *SI = cast<StoreInst>(I);
    Value *Operand = SI->getValueOperand();
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    addLoadStoreInstrumentation(I, CsiBeforeWrite, CsiAfterWrite, CsiId,
                                AddrType, Addr, NumBytes, OperandID.first,
                                OperandID.second, ObjID.first, ObjID.second,
                                Prop);
  } else { // is read
    csi_id_t LocalId = LoadFED.lookupId(I);
    Value *CsiId = LoadFED.localToGlobalId(LocalId, IRB);

    addLoadStoreInstrumentation(I, CsiBeforeRead, CsiAfterRead, CsiId, AddrType,
                                Addr, NumBytes, nullptr, nullptr, ObjID.first,
                                ObjID.second, Prop);
  }
}

void CSIImpl::instrumentVectorMemBuiltin(Instruction *I) {
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
  if (!II)
    return;

  IRBuilder<> IRB(I);
  CsiLoadStoreProperty Prop;
  csi_id_t LocalId;
  Value *Operand0, *Operand1, *Operand2;
  unsigned Alignment = 0;
  // Get the before and after hooks.
  Function *BeforeHook = getMaskedReadWriteHook(M, I, true);
  Function *AfterHook = getMaskedReadWriteHook(M, I, false);
  if (!BeforeHook || !AfterHook) {
    dbgs() << "Unknown VectorMemBuiltin " << *I << "\n";
    return;
  }
  // Process the arguments of this vector memory builtin.
  switch (II->getIntrinsicID()) {
  default:
    dbgs() << "Unknown VectorMemBuiltin " << *I << "\n";
    return;
  case Intrinsic::masked_load: {
    LocalId = LoadFED.lookupId(I);
    Operand0 = II->getArgOperand(0);
    if (ConstantInt *C = dyn_cast<ConstantInt>(II->getArgOperand(1)))
      Alignment = C->getZExtValue();
    Operand1 = II->getArgOperand(2);
    Operand2 = II->getArgOperand(3);
  }
  case Intrinsic::masked_gather: {
    LocalId = LoadFED.lookupId(I);
    Operand0 = II->getArgOperand(0);
    if (ConstantInt *C = dyn_cast<ConstantInt>(II->getArgOperand(1)))
      Alignment = C->getZExtValue();
    Operand1 = II->getArgOperand(2);
    Operand2 = II->getArgOperand(3);
  }
  case Intrinsic::masked_store: {
    LocalId = StoreFED.lookupId(I);
    Operand0 = II->getArgOperand(1);
    if (ConstantInt *C = dyn_cast<ConstantInt>(II->getArgOperand(2)))
      Alignment = C->getZExtValue();
    Operand1 = II->getArgOperand(3);
    Operand2 = II->getArgOperand(0);
  }
  case Intrinsic::masked_scatter: {
    LocalId = StoreFED.lookupId(I);
    Operand0 = II->getArgOperand(1);
    if (ConstantInt *C = dyn_cast<ConstantInt>(II->getArgOperand(2)))
      Alignment = C->getZExtValue();
    Operand1 = II->getArgOperand(3);
    Operand2 = II->getArgOperand(0);
  }
  }
  // Compute the property
  Prop.setAlignment(Alignment);
  Value *PropVal = Prop.getValue(IRB);
  // Get the IDs
  Value *CsiId = LoadFED.localToGlobalId(LocalId, IRB);
  std::pair<Value *, Value *> Operand1ID = getOperandID(Operand1, IRB);
  std::pair<Value *, Value *> Operand2ID = getOperandID(Operand2, IRB);
  // Create a mask argument.
  unsigned NumEls = cast<VectorType>(Operand1->getType())->getNumElements();
  Value *CastMask = IRB.CreateZExtOrBitCast(Operand1,
                                            VectorType::get(IRB.getInt8Ty(),
                                                            NumEls));
  // Insert the hooks.
  insertHookCall(I, BeforeHook,
                 {CsiId, Operand0, Operand1ID.first, Operand1ID.second,
                  CastMask, Operand2ID.first, Operand2ID.second, Operand2,
                  PropVal});
  BasicBlock::iterator Iter(I);
  Iter++;
  IRB.SetInsertPoint(&*Iter);
  insertHookCall(&*Iter, AfterHook,
                 {CsiId, Operand0, Operand1ID.first, Operand1ID.second,
                  CastMask, Operand2ID.first, Operand2ID.second, Operand2,
                  PropVal});
}

void CSIImpl::assignAtomicID(Instruction *I) {
  // TODO: Instrument atomics.
}

void CSIImpl::instrumentAtomic(Instruction *I, const DataLayout &DL) {
  // TODO: Instrument atomics.

  // For now, print a message that this code contains atomics.
  dbgs()
      << "WARNING: Uninstrumented atomic operations in program-under-test!\n";
}

bool CSIImpl::instrumentMemIntrinsic(Instruction *I) {
  IRBuilder<> IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    // Retrieve operands of instruction
    Value *Addr = M->getArgOperand(0);
    Value *Operand = M->getArgOperand(1);
    Value *NumBytes = M->getArgOperand(2);

    // Get arguments for hooks
    csi_id_t LocalId = CallsiteFED.lookupId(I);
    Value *CsiId = CallsiteFED.localToGlobalId(LocalId, IRB);
    Value *AddrArg = IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy());
    Value *NumBytesArg = IRB.CreateSExtOrBitCast(NumBytes, IRB.getInt64Ty());
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);

    // Insert hooks
    insertHookCall(I, CsiBeforeMemset, {CsiId, AddrArg, NumBytesArg,
                                        OperandID.first, OperandID.second,
                                        Operand});
    BasicBlock::iterator Iter(I);
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    insertHookCall(&*Iter, CsiAfterMemset, {CsiId, AddrArg, NumBytesArg,
                                            OperandID.first, OperandID.second,
                                            Operand});
    return true;
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    // Retrieve operands of instruction
    Value *Dest = M->getArgOperand(0);
    Value *Src = M->getArgOperand(1);
    Value *NumBytes = M->getArgOperand(2);

    // Get hooks.
    Function *BeforeHook = nullptr, *AfterHook = nullptr;
    if (isa<MemCpyInst>(M)) {
      BeforeHook = CsiBeforeMemcpy;
      AfterHook = CsiAfterMemcpy;
    } else if (isa<MemMoveInst>(M)) {
      BeforeHook = CsiBeforeMemmove;
      AfterHook = CsiAfterMemmove;
    } else {
      dbgs() << "Uninstrumented memory intrinsic " << *M << "\n";
      return false;
    }

    // Get arguments for hooks
    csi_id_t LocalId = CallsiteFED.lookupId(I);
    Value *CsiId = CallsiteFED.localToGlobalId(LocalId, IRB);
    Value *DestArg = IRB.CreatePointerCast(Dest, IRB.getInt8PtrTy());
    Value *SrcArg = IRB.CreatePointerCast(Src, IRB.getInt8PtrTy());
    Value *NumBytesArg = IRB.CreateSExtOrBitCast(NumBytes, IRB.getInt64Ty());

    // Insert hooks
    insertHookCall(I, BeforeHook, {CsiId, DestArg, SrcArg, NumBytesArg});
    BasicBlock::iterator Iter(I);
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    insertHookCall(&*Iter, AfterHook, {CsiId, DestArg, SrcArg, NumBytesArg});
    return true;
  } else {
    dbgs() << "Uninstrumented memory intrinsic " << *M << "\n";
  }
  return false;
}

// Helper function to check if the given basic block is devoid of instructions
// other than PHI's, calls to placeholders, and its terminator.
std::pair<bool, bool> CSIImpl::isBBEmpty(BasicBlock &BB) {
  bool IsEmpty = false;
  BasicBlock::iterator Iter = BB.getFirstInsertionPt();
  while ((&*Iter != BB.getTerminator()) && callsPlaceholderFunction(*Iter))
    Iter++;
  IsEmpty = (&*Iter == BB.getTerminator());

  // TODO: Update these checks once we instrument atomics
  while (&*Iter != BB.getTerminator()) {
    Instruction *I = &*Iter++;

    // TODO: Instrument atomics.
    // if (isAtomic(I))
    //   if (AtomicFED.hasId(I))
    //     return std::make_pair(IsEmpty, false);

    if (isa<LoadInst>(I))
      if (LoadFED.hasId(I))
        return std::make_pair(IsEmpty, false);

    if (isa<StoreInst>(I))
      if (StoreFED.hasId(I))
        return std::make_pair(IsEmpty, false);

    if (CallInst *CI = dyn_cast<CallInst>(I)) {
      // Skip this call if it calls a placeholder function.
      if (callsPlaceholderFunction(*I))
        continue;

      if (isAllocationFn(CI, TLI))
        if (AllocFnFED.hasId(CI))
          return std::make_pair(IsEmpty, false);

      if (isFreeCall(CI, TLI))
        if (FreeFED.hasId(CI))
          return std::make_pair(IsEmpty, false);

      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI)) {
        // Skip this intrinsic if there's no FED entry.
        switch (II->getIntrinsicID()) {
          // Handle instrinsics for masked vector loads, stores, gathers, and
          // scatters specially.
        case Intrinsic::masked_load:
        case Intrinsic::masked_gather:
          if (LoadFED.hasId(II))
            return std::make_pair(IsEmpty, false);
        case Intrinsic::masked_store:
        case Intrinsic::masked_scatter:
          if (StoreFED.hasId(II))
            return std::make_pair(IsEmpty, false);
        default:
          if (CallsiteFED.hasId(II))
            return std::make_pair(IsEmpty, false);
        }
      } else {
        if (CallsiteFED.hasId(CI))
          return std::make_pair(IsEmpty, false);
      }
    }

    if (isa<AllocaInst>(I))
      if (AllocaFED.hasId(I))
        return std::make_pair(IsEmpty, false);

    if (IsInstrumentedArithmetic(I))
      if (ArithmeticFED.hasId(I))
        return std::make_pair(IsEmpty, false);
  }
  return std::make_pair(IsEmpty, true);
}

static void getBBInputs(BasicBlock &BB, SmallPtrSetImpl<Value *> &Inputs) {
  // If a used value is defined outside the region, it's an input.  If an
  // instruction is used outside the region, it's an output.
  for (Instruction &II : BB) {
    // Because PHI-node instrumentation is inserted before bb_entry, we consider
    // the PHI nodes inputs of the basic block.
    if (isa<PHINode>(II)) {
      Inputs.insert(&II);
      continue;
    }

    if (isa<InvokeInst>(II))
      // Skip invoke instructions, because they are handled separately along
      // with calls.
      continue;

    if (isa<DetachInst>(II) || isa<ReattachInst>(II) || isa<SyncInst>(II))
      // Skip the Tapir instructions, because they don't directly use any
      // values.  This check ensures that we ignore sync regions as BB inputs.
      continue;

    // Examine all operands of this instruction.
    for (User::const_op_iterator OI = II.op_begin(), OE = II.op_end(); OI != OE;
         ++OI) {
      // If this operand is not defined in this basic block, it's an input.
      if (isa<Argument>(*OI) || isa<GlobalVariable>(*OI))
        Inputs.insert(*OI);
      if (Instruction *UI = dyn_cast<Instruction>(&*OI))
        if (UI->getParent() != &BB)
          Inputs.insert(*OI);
    }
  }
}

void CSIImpl::assignBasicBlockID(BasicBlock &BB) {
  csi_id_t LocalId = BasicBlockFED.add(BB);
  csi_id_t BBSizeId = BBSize.add(BB);
  assert(LocalId == BBSizeId &&
         "BB recieved different ID's in FED and sizeinfo tables.");
}

void CSIImpl::instrumentBasicBlock(BasicBlock &BB,
                                   const SmallPtrSetImpl<Value *> &Inputs) {
  // Compute the properties of this basic block.
  CsiBBProperty Prop;
  Prop.setIsLandingPad(BB.isLandingPad());
  Prop.setIsEHPad(BB.isEHPad());
  TerminatorInst *TI = BB.getTerminator();
  assert(TI && "Found a BB with no terminator.");
  Prop.setTerminatorTy(static_cast<unsigned>(getTerminatorTy(*TI)));
  std::pair<bool, bool> EmptyBB = isBBEmpty(BB);
  Prop.setIsEmpty(EmptyBB.first);
  Prop.setNoInstrumentedContent(EmptyBB.second);

  IRBuilder<> IRB(&*BB.getFirstInsertionPt());
  csi_id_t LocalId = BasicBlockFED.lookupId(&BB);
  Value *CsiId = BasicBlockFED.localToGlobalId(LocalId, IRB);
  Type *IDType = IRB.getInt64Ty();

  // Insert a PHI to track the BB predecessor.
  Value *LocalPredID = IRB.getInt64(CsiUnknownId);
  if (!pred_empty(&BB)) {
    IRBuilder<> ArgB(&BB.front());
    PHINode *LocalPredIDPN = ArgB.CreatePHI(IDType, 0);
    for (BasicBlock *Pred : predecessors(&BB)) {
      csi_id_t LocalId = BasicBlockFED.lookupId(Pred);
      LocalPredIDPN->addIncoming(IRB.getInt64(LocalId), Pred);
    }
    LocalPredID = LocalPredIDPN;
  }
  Value *PredID = BasicBlockFED.localToGlobalId(LocalPredID, IRB);

  // Insert input hooks for the inputs to the basic block.
  instrumentInputs(IRB, CSIDataFlowObject::BasicBlock, CsiId, Inputs);

  // Insert entry and exit hooks.
  // csi_id_t BBSizeId = BBSize.add(BB);
  // assert(LocalId == BBSizeId &&
  //        "BB recieved different ID's in FED and sizeinfo tables.");
  Value *PropVal = Prop.getValue(IRB);
  insertHookCall(&*IRB.GetInsertPoint(), CsiBBEntry, {CsiId, PredID, PropVal});

  IRB.SetInsertPoint(TI);
  insertHookCall(TI, CsiBBExit, {CsiId, PropVal});
}

// Helper function to get a value for the runtime trip count of the given loop.
static const SCEV *getRuntimeTripCount(Loop &L, ScalarEvolution *SE) {
  BasicBlock *Latch = L.getLoopLatch();

  const SCEV *BECountSC = SE->getExitCount(&L, Latch);
  if (isa<SCEVCouldNotCompute>(BECountSC) ||
      !BECountSC->getType()->isIntegerTy()) {
    DEBUG(dbgs() << "Could not compute exit block SCEV\n");
    return SE->getCouldNotCompute();
  }

  // Add 1 since the backedge count doesn't include the first loop iteration.
  const SCEV *TripCountSC =
      SE->getAddExpr(BECountSC, SE->getConstant(BECountSC->getType(), 1));
  if (isa<SCEVCouldNotCompute>(TripCountSC)) {
    DEBUG(dbgs() << "Could not compute trip count SCEV.\n");
    return SE->getCouldNotCompute();
  }

  return TripCountSC;
}

void CSIImpl::getAllLoopInputs(Loop &L, LoopInfo &LI, InputMap<Loop> &Inputs) {
  // Recursively get the loop inputs for all subloops.
  for (Loop *SubL : L)
    getAllLoopInputs(*SubL, LI, Inputs);

  SmallPtrSetImpl<Value *> &LInputs = Inputs[&L];
  // Check the loop inputs for the subloops to see which are defined in this
  // loop.
  for (Loop *SubL : L) {
    for (Value *SubLInput : Inputs[SubL]) {
      // Add all inputs that are arguments or global values
      if (isa<Argument>(SubLInput) || isa<GlobalValue>(SubLInput))
        LInputs.insert(SubLInput);

      // Check if the input is defined in this loop.
      if (Instruction *I = dyn_cast<Instruction>(SubLInput))
        if (LI.getLoopFor(I->getParent()) != &L)
          LInputs.insert(SubLInput);
    }
  }

  // Now check the basic blocks in L and not any subloop of L.
  for (BasicBlock *BB : L.blocks()) {
    // Skip basic blocks in subloops of L.
    if (LI.getLoopFor(BB) != &L)
      continue;

    // If a used value is defined outside the region, it's an input.  If an
    // instruction is used outside the region, it's an output.
    for (Instruction &II : *BB) {
      if (isa<DetachInst>(II) || isa<ReattachInst>(II) || isa<SyncInst>(II))
        // Skip the Tapir instructions, because they don't directly use any
        // values.  This check ensures that we ignore sync regions as inputs.
        continue;

      // Examine all operands of this instruction.
      for (User::const_op_iterator OI = II.op_begin(), OE = II.op_end(); OI != OE;
           ++OI) {
        // If this operand is not defined in this basic block, it's an input.
        if (isa<Argument>(*OI) || isa<GlobalVariable>(*OI))
          LInputs.insert(*OI);
        if (Instruction *UI = dyn_cast<Instruction>(&*OI))
          if (!L.contains(UI->getParent()))
            LInputs.insert(*OI);
      }
    }
  }

  DEBUG({
      dbgs() << "Inputs for " << L << "\n";
      for (Value *Input : LInputs) {
        dbgs() << "\t" << *Input;
        if (Instruction *I = dyn_cast<Instruction>(Input))
          dbgs() << " in " << I->getParent()->getName();
        else if (Argument *A = dyn_cast<Argument>(Input))
          dbgs() << " in " << A->getParent()->getName();
        dbgs() << "\n";
      }
    });
}

void CSIImpl::instrumentLoop(Loop &L, const InputMap<Loop> &LoopInputs,
                             TaskInfo &TI, ScalarEvolution *SE) {
  assert(L.isLoopSimplifyForm() && "CSI assumes loops are in simplified form.");
  BasicBlock *Preheader = L.getLoopPreheader();
  BasicBlock *Header = L.getHeader();
  SmallVector<BasicBlock *, 4> ExitingBlocks, ExitBlocks;
  L.getExitingBlocks(ExitingBlocks);
  L.getUniqueExitBlocks(ExitBlocks);

  // We assign a local ID for this loop here, so that IDs for loops follow a
  // depth-first ordering.
  csi_id_t LocalId = LoopFED.add(*Header);

  // Recursively instrument each subloop.
  for (Loop *SubL : L)
    instrumentLoop(*SubL, LoopInputs, TI, SE);

  // Record properties of this loop.
  CsiLoopProperty LoopProp;
  LoopProp.setIsTapirLoop(static_cast<bool>(getTaskIfTapirLoop(&L, &TI)));
  LoopProp.setHasUniqueExitingBlock((ExitingBlocks.size() == 1));

  IRBuilder<> IRB(Preheader->getTerminator());
  Value *LoopCsiId = LoopFED.localToGlobalId(LocalId, IRB);
  Value *LoopPropVal = LoopProp.getValue(IRB);

  // Try to evaluate the runtime trip count for this loop.  Default to a count
  // of -1 for unknown trip counts.
  Value *TripCount = IRB.getInt64(-1);
  if (SE) {
    const SCEV *TripCountSC = getRuntimeTripCount(L, SE);
    if (!isa<SCEVCouldNotCompute>(TripCountSC)) {
      // Extend the TripCount type if necessary.
      if (TripCountSC->getType() != IRB.getInt64Ty())
        TripCountSC = SE->getZeroExtendExpr(TripCountSC, IRB.getInt64Ty());
      // Compute the trip count to pass to the CSI hook.
      SCEVExpander Expander(*SE, DL, "csi");
      TripCount = Expander.expandCodeFor(TripCountSC, IRB.getInt64Ty(),
                                         &*IRB.GetInsertPoint());
    }
  }

  // Insert input hooks for the inputs to the basic block.
  instrumentInputs(IRB, CSIDataFlowObject::Loop, LoopCsiId,
                   LoopInputs.lookup(&L));

  // Insert before-loop hook.
  insertHookCall(&*IRB.GetInsertPoint(), CsiBeforeLoop, {LoopCsiId, TripCount,
                                                         LoopPropVal});

  // Insert loop-body-entry hook.
  IRB.SetInsertPoint(&*Header->getFirstInsertionPt());
  // TODO: Pass IVs to hook?
  insertHookCall(&*IRB.GetInsertPoint(), CsiLoopBodyEntry, {LoopCsiId,
                                                            LoopPropVal});

  // Insert hooks at the ends of the exiting blocks.
  for (BasicBlock *BB : ExitingBlocks) {
    // Record properties of this loop exit
    CsiLoopExitProperty LoopExitProp;
    LoopExitProp.setIsLatch(L.isLoopLatch(BB));

    // Insert the loop-exit hook
    IRB.SetInsertPoint(BB->getTerminator());
    csi_id_t LocalExitId = LoopExitFED.add(*BB);
    Value *ExitCsiId = LoopFED.localToGlobalId(LocalExitId, IRB);
    Value *LoopExitPropVal = LoopExitProp.getValue(IRB);
    // TODO: For latches, record whether the loop will repeat.
    insertHookCall(&*IRB.GetInsertPoint(), CsiLoopBodyExit,
                   {ExitCsiId, LoopCsiId, LoopExitPropVal});
  }
  // Insert after-loop hooks.
  for (BasicBlock *BB : ExitBlocks) {
    IRB.SetInsertPoint(&*BB->getFirstInsertionPt());
    insertHookCall(&*IRB.GetInsertPoint(), CsiAfterLoop, {LoopCsiId,
                                                          LoopPropVal});
  }
}

void CSIImpl::assignCallsiteID(Instruction *I) {
  Function *Called = nullptr;
  if (CallInst *CI = dyn_cast<CallInst>(I))
    Called = CI->getCalledFunction();
  else if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    Called = II->getCalledFunction();
  CallsiteFED.add(*I, Called ? Called->getName() : "");
}

CSIImpl::CSIBuiltinFuncOp CSIImpl::getBuiltinFuncOp(CallSite &CS) {
  const Function *F = CS.getCalledFunction();
  switch (F->getIntrinsicID()) {
  default: break;
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
    return CSIBuiltinFuncOp::F_Fma;
  case Intrinsic::sqrt:
    return CSIBuiltinFuncOp::F_Sqrt;
  case Intrinsic::sin:
    return CSIBuiltinFuncOp::F_Sin;
  case Intrinsic::cos:
    return CSIBuiltinFuncOp::F_Cos;
  case Intrinsic::log:
    return CSIBuiltinFuncOp::F_Log;
  case Intrinsic::log10:
    return CSIBuiltinFuncOp::F_Log10;
  case Intrinsic::log2:
    return CSIBuiltinFuncOp::F_Log2;
  case Intrinsic::exp:
    return CSIBuiltinFuncOp::F_Exp;
  case Intrinsic::exp2:
    return CSIBuiltinFuncOp::F_Exp2;
  case Intrinsic::fabs:
    return CSIBuiltinFuncOp::F_Fabs;
  case Intrinsic::floor:
    return CSIBuiltinFuncOp::F_Floor;
  case Intrinsic::ceil:
    return CSIBuiltinFuncOp::F_Ceil;
  case Intrinsic::trunc:
    return CSIBuiltinFuncOp::F_Trunc;
  case Intrinsic::rint:
    return CSIBuiltinFuncOp::F_Rint;
  case Intrinsic::nearbyint:
    return CSIBuiltinFuncOp::F_NearbyInt;
  case Intrinsic::round:
    return CSIBuiltinFuncOp::F_Round;
  case Intrinsic::canonicalize:
    return CSIBuiltinFuncOp::F_Canonicalize;
  case Intrinsic::pow:
  case Intrinsic::powi:
    return CSIBuiltinFuncOp::F_Pow;
  case Intrinsic::copysign:
    return CSIBuiltinFuncOp::F_CopySign;
  case Intrinsic::minnum:
    return CSIBuiltinFuncOp::F_MinNum;
  case Intrinsic::maxnum:
    return CSIBuiltinFuncOp::F_MaxNum;
  }

  // Called function is not an intrinsic.  Try to handle it as a library call.
  LibFunc Func;
  if (!TLI || !TLI->getLibFunc(*F, Func))
    return CSIBuiltinFuncOp::LAST_CSIBuiltinFuncOp;

  // TODO: Handle frexp, frexpf, modf, modff
  switch (Func) {
  case LibFunc_log:
  case LibFunc_log_finite:
  case LibFunc_logf:
  case LibFunc_logf_finite:
    return CSIBuiltinFuncOp::F_Log;
  case LibFunc_log2:
  case LibFunc_log2_finite:
  case LibFunc_log2f:
  case LibFunc_log2f_finite:
    return CSIBuiltinFuncOp::F_Log2;
  case LibFunc_logb:
  case LibFunc_logbf:
    return CSIBuiltinFuncOp::F_Logb;
  case LibFunc_log10:
  case LibFunc_log10_finite:
  case LibFunc_log10f:
  case LibFunc_log10f_finite:
    return CSIBuiltinFuncOp::F_Log10;
  case LibFunc_log1p:
  case LibFunc_log1pf:
    return CSIBuiltinFuncOp::F_Log1p;
  case LibFunc_exp:
  case LibFunc_exp_finite:
  case LibFunc_expf:
  case LibFunc_expf_finite:
    return CSIBuiltinFuncOp::F_Exp;
  case LibFunc_exp2:
  case LibFunc_exp2_finite:
  case LibFunc_exp2f:
  case LibFunc_exp2f_finite:
    return CSIBuiltinFuncOp::F_Exp2;
  case LibFunc_exp10:
  case LibFunc_exp10_finite:
  case LibFunc_exp10f:
  case LibFunc_exp10f_finite:
    return CSIBuiltinFuncOp::F_Exp10;
  case LibFunc_expm1:
  case LibFunc_expm1f:
    return CSIBuiltinFuncOp::F_Expm1;
  case LibFunc_sin:
  case LibFunc_sinf:
    return CSIBuiltinFuncOp::F_Sin;
  case LibFunc_cos:
  case LibFunc_cosf:
    return CSIBuiltinFuncOp::F_Cos;
  case LibFunc_sinpi:
  case LibFunc_sinpif:
    return CSIBuiltinFuncOp::F_SinPi;
  case LibFunc_cospi:
  case LibFunc_cospif:
    return CSIBuiltinFuncOp::F_CosPi;
  case LibFunc_sincospi_stret:
  case LibFunc_sincospif_stret:
    return CSIBuiltinFuncOp::F_SinCosPi;
  case LibFunc_tan:
  case LibFunc_tanf:
    return CSIBuiltinFuncOp::F_Tan;
  case LibFunc_asin:
  case LibFunc_asin_finite:
  case LibFunc_asinf:
  case LibFunc_asinf_finite:
    return CSIBuiltinFuncOp::F_ASin;
  case LibFunc_acos:
  case LibFunc_acos_finite:
  case LibFunc_acosf:
  case LibFunc_acosf_finite:
    return CSIBuiltinFuncOp::F_ACos;
  case LibFunc_atan:
  case LibFunc_atanf:
    return CSIBuiltinFuncOp::F_ATan;
  case LibFunc_sinh:
  case LibFunc_sinh_finite:
  case LibFunc_sinhf:
  case LibFunc_sinhf_finite:
    return CSIBuiltinFuncOp::F_Sinh;
  case LibFunc_cosh:
  case LibFunc_cosh_finite:
  case LibFunc_coshf:
  case LibFunc_coshf_finite:
    return CSIBuiltinFuncOp::F_Cosh;
  case LibFunc_tanh:
  case LibFunc_tanhf:
    return CSIBuiltinFuncOp::F_Tanh;
  case LibFunc_asinh:
  case LibFunc_asinhf:
    return CSIBuiltinFuncOp::F_ASinh;
  case LibFunc_acosh:
  case LibFunc_acosh_finite:
  case LibFunc_acoshf:
  case LibFunc_acoshf_finite:
    return CSIBuiltinFuncOp::F_ACosh;
  case LibFunc_atanh:
  case LibFunc_atanh_finite:
  case LibFunc_atanhf:
  case LibFunc_atanhf_finite:
    return CSIBuiltinFuncOp::F_ATanh;
  case LibFunc_sqrt:
  case LibFunc_sqrt_finite:
  case LibFunc_sqrtf:
  case LibFunc_sqrtf_finite:
    return CSIBuiltinFuncOp::F_Sqrt;
  case LibFunc_cbrt:
  case LibFunc_cbrtf:
    return CSIBuiltinFuncOp::F_Cbrt;
  case LibFunc_ceil:
  case LibFunc_ceilf:
    return CSIBuiltinFuncOp::F_Ceil;
  case LibFunc_fabs:
  case LibFunc_fabsf:
    return CSIBuiltinFuncOp::F_Fabs;
  case LibFunc_floor:
  case LibFunc_floorf:
    return CSIBuiltinFuncOp::F_Floor;
  case LibFunc_nearbyint:
  case LibFunc_nearbyintf:
    return CSIBuiltinFuncOp::F_NearbyInt;
  case LibFunc_rint:
  case LibFunc_rintf:
    return CSIBuiltinFuncOp::F_Rint;
  case LibFunc_round:
  case LibFunc_roundf:
    return CSIBuiltinFuncOp::F_Round;
  case LibFunc_trunc:
  case LibFunc_truncf:
    return CSIBuiltinFuncOp::F_Trunc;
  case LibFunc_atan2:
  case LibFunc_atan2_finite:
  case LibFunc_atan2f:
  case LibFunc_atan2f_finite:
    return CSIBuiltinFuncOp::F_ATan2;
  case LibFunc_copysign:
  case LibFunc_copysignf:
    return CSIBuiltinFuncOp::F_CopySign;
  case LibFunc_pow:
  case LibFunc_pow_finite:
  case LibFunc_powf:
  case LibFunc_powf_finite:
    return CSIBuiltinFuncOp::F_Pow;
  case LibFunc_fmod:
  case LibFunc_fmodf:
    return CSIBuiltinFuncOp::F_Fmod;
  case LibFunc_fmin:
  case LibFunc_fminf:
    return CSIBuiltinFuncOp::F_MinNum;
  case LibFunc_fmax:
  case LibFunc_fmaxf:
    return CSIBuiltinFuncOp::F_MaxNum;
  case LibFunc_ldexp:
  case LibFunc_ldexpf:
    return CSIBuiltinFuncOp::F_Ldexp;
  default: break;
  }

  return CSIBuiltinFuncOp::LAST_CSIBuiltinFuncOp;
}

bool CSIImpl::handleFPBuiltinCall(CallInst *I, Function *F, LoopInfo &LI) {
  CallSite CS(I);

  CSIBuiltinFuncOp Op = getBuiltinFuncOp(CS);
  if (CSIBuiltinFuncOp::LAST_CSIBuiltinFuncOp == Op)
    // Unrecognized builtin; just instrument it like any other call.
    return false;

  IRBuilder<> IRB(I);
  CsiCallProperty Prop;
  csi_id_t LocalId = CallsiteFED.lookupId(I);

  // Get the hooks
  Function *BeforeHook = getCSIBuiltinHook(M, I, true);
  Function *AfterHook = getCSIBuiltinHook(M, I, false);
  // Get the ID and builtin-func-op code.
  Value *CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
  Value *OpArg = IRB.getInt8(static_cast<unsigned>(Op));

  // Collect all hook arguments
  SmallVector<Value *, 3> HookArgs;
  HookArgs.push_back(CallsiteId);
  HookArgs.push_back(OpArg);
  // Add the Operand ID and argument information for each argument.
  for (Value *Arg : CS.args()) {
    std::pair<Value *, Value *> OperandID = getOperandID(Arg, IRB);
    HookArgs.push_back(OperandID.first);
    HookArgs.push_back(OperandID.second);
    // No casting is needed for FP builtins.
    HookArgs.push_back(Arg);
  }
  // Add the property value.
  Prop.setHasOneUse(checkHasOneUse(I, LI));
  Value *PropVal = Prop.getValue(IRB);
  HookArgs.push_back(PropVal);

  // Insert the before hook.
  insertHookCall(I, BeforeHook, HookArgs);

  // Insert the after hook.
  BasicBlock::iterator Iter(I);
  Iter++;
  IRB.SetInsertPoint(&*Iter);
  insertHookCall(&*Iter, AfterHook, HookArgs);
  return true;
}

void CSIImpl::instrumentCallsite(Instruction *I, DominatorTree *DT,
                                 LoopInfo &LI) {
  if (callsPlaceholderFunction(*I))
    return;

  bool IsInvoke = isa<InvokeInst>(I);
  Function *Called = nullptr;
  unsigned NumArgs = 0;
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    Called = CI->getCalledFunction();
    NumArgs = CI->getNumArgOperands();
  } else if (InvokeInst *II = dyn_cast<InvokeInst>(I)) {
    Called = II->getCalledFunction();
    NumArgs = II->getNumArgOperands();
  }

  // Handle calls to builtins
  if (!IsInvoke && Called &&
      handleFPBuiltinCall(cast<CallInst>(I), Called, LI))
    return;

  bool shouldInstrumentBefore = true;
  bool shouldInstrumentAfter = true;

  // Does this call require instrumentation before or after?
  if (Called) {
    shouldInstrumentBefore = Config->DoesFunctionRequireInstrumentationForPoint(
        Called->getName(), InstrumentationPoint::INSTR_BEFORE_CALL);
    shouldInstrumentAfter = Config->DoesFunctionRequireInstrumentationForPoint(
        Called->getName(), InstrumentationPoint::INSTR_AFTER_CALL);
  }

  if (!shouldInstrumentAfter && !shouldInstrumentBefore)
    return;

  IRBuilder<> IRB(I);
  // Get the CSI ID of this callsite, along with a default value for handling
  // invokes.
  Value *DefaultID = getDefaultID(IRB);
  csi_id_t LocalId = CallsiteFED.lookupId(I);
  Value *CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);

  // Get the CSI ID of the called function.
  Value *FuncId = nullptr;
  GlobalVariable *FuncIdGV = nullptr;
  if (Called) {
    // Because we're calling this function, we must ensure that the CSI global
    // storing the CSI ID of the callee is available.  Create the CSI global in
    // this module if necessary.
    std::string GVName = CsiFuncIdVariablePrefix + Called->getName().str();
    FuncIdGV = dyn_cast<GlobalVariable>(M.getOrInsertGlobal(GVName,
                                                            IRB.getInt64Ty()));
    assert(FuncIdGV);
    FuncIdGV->setConstant(false);
    FuncIdGV->setLinkage(GlobalValue::WeakAnyLinkage);
    FuncIdGV->setInitializer(IRB.getInt64(CsiCallsiteUnknownTargetId));
    FuncId = IRB.CreateLoad(FuncIdGV);
  } else {
    // Unknown targets (i.e. indirect calls) are always unknown.
    FuncId = IRB.getInt64(CsiCallsiteUnknownTargetId);
  }
  assert(FuncId != NULL);

  // Insert input hooks for the inputs to the call.
  if (Options.InstrumentInputs) {
    CallSite CS(I);
    SmallPtrSet<Value *, 4> Inputs;
    for (Value *Input : CS.args())
      Inputs.insert(Input);
    instrumentInputs(IRB, CSIDataFlowObject::Call, CallsiteId, Inputs);
  }

  // Get the ID of the basic block containing this call.  This is helpful in
  // particular for invokes, which terminate a basic block and occur after the
  // bb_exit hook.
  csi_id_t LocalBBId = BasicBlockFED.lookupId(I->getParent());
  Value *BBID = BasicBlockFED.localToGlobalId(LocalBBId, IRB);

  // Get properties of this call.
  CsiCallProperty Prop;
  Value *DefaultPropVal = Prop.getValue(IRB);
  Prop.setIsIndirect(!Called);
  Prop.setHasOneUse(checkHasOneUse(I, LI));
  Prop.setBBLocal(checkBBLocal(I, *I->getParent()));
  Value *PropVal = Prop.getValue(IRB);

  // Instrument the call
  if (shouldInstrumentBefore)
    insertHookCall(I, CsiBeforeCallsite, {CallsiteId, BBID, FuncId, PropVal});
  BasicBlock::iterator Iter(I);
  if (shouldInstrumentAfter) {
    if (IsInvoke) {
      // There are two "after" positions for invokes: the normal block and the
      // exception block.
      InvokeInst *II = cast<InvokeInst>(I);
      insertHookCallInSuccessorBB(II, II->getNormalDest(), II->getParent(),
                                  CsiAfterCallsite,
                                  {CallsiteId, FuncId, PropVal},
                                  {DefaultID, DefaultID, DefaultPropVal});
      insertHookCallInSuccessorBB(II, II->getUnwindDest(), II->getParent(),
                                  CsiAfterCallsite,
                                  {CallsiteId, FuncId, PropVal},
                                  {DefaultID, DefaultID, DefaultPropVal});
    } else {
      // Simple call instruction; there is only one "after" position.
      Iter++;
      IRB.SetInsertPoint(&*Iter);
      PropVal = Prop.getValue(IRB);
      insertHookCall(&*Iter, CsiAfterCallsite, {CallsiteId, FuncId, PropVal});
    }
  }
}

void CSIImpl::interposeCall(Instruction *I) {

  Function *Called = nullptr;
  if (CallInst *CI = dyn_cast<CallInst>(I))
    Called = CI->getCalledFunction();
  else if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    Called = II->getCalledFunction();

  // Should we interpose this call?
  if (Called && Called->getName().size() > 0) {
    bool shouldInterpose =
        Config->DoesFunctionRequireInterposition(Called->getName());

    if (shouldInterpose) {
      Function *interpositionFunction = getInterpositionFunction(Called);
      assert(interpositionFunction != nullptr);
      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        CI->setCalledFunction(interpositionFunction);
      } else if (InvokeInst *II = dyn_cast<InvokeInst>(I)) {
        II->setCalledFunction(interpositionFunction);
      }
    }
  }
}

static void getTaskExits(DetachInst *DI,
                         SmallVectorImpl<BasicBlock *> &TaskReturns,
                         SmallVectorImpl<BasicBlock *> &TaskResumes,
                         SmallVectorImpl<Spindle *> &SharedEHExits,
                         TaskInfo &TI) {
  BasicBlock *DetachedBlock = DI->getDetached();
  Task *T = TI.getTaskFor(DetachedBlock);
  BasicBlock *ContinueBlock = DI->getContinue();

  // Examine the predecessors of the continue block and save any predecessors in
  // the task as a task return.
  for (BasicBlock *Pred : predecessors(ContinueBlock)) {
    if (T->simplyEncloses(Pred)) {
      assert(isa<ReattachInst>(Pred->getTerminator()));
      TaskReturns.push_back(Pred);
    }
  }

  // If the detach cannot throw, we're done.
  if (!DI->hasUnwindDest())
    return;

  // Detached-rethrow exits can appear in strange places within a task-exiting
  // spindle.  Hence we loop over all blocks in the spindle to find
  // detached rethrows.
  for (Spindle *S : depth_first<InTask<Spindle *>>(T->getEntrySpindle())) {
    if (S->isSharedEH()) {
      if (llvm::any_of(predecessors(S),
                       [](const Spindle *Pred) { return !Pred->isSharedEH(); }))
        SharedEHExits.push_back(S);
      continue;
    }

    for (BasicBlock *B : S->blocks())
      if (isDetachedRethrow(B->getTerminator()))
        TaskResumes.push_back(B);
  }
}

// TODO: See if there's a simple way to combine this logic with the
// findTaskInputsOutputs logic currently in Tapir/LoweringUtils.
void CSIImpl::getAllTaskInputs(TaskInfo &TI, InputMap<Task> &Inputs) {
  for (Task *T : post_order(TI.getRootTask())) {
    // Skip the root task
    if (T->isRootTask())
      break;

    SmallPtrSetImpl<Value *> &TInputs = Inputs[T];
    // Check the inputs for all subtasks to see which are defined in this task.
    for (Task *SubT : T->subtasks())
      if (Inputs.count(SubT))
        for (Value *V : Inputs[SubT]) {
          if (isa<Argument>(V) || isa<GlobalValue>(V))
            TInputs.insert(V);

          if (Instruction *I = dyn_cast<Instruction>(V))
            if (TI.getTaskFor(I->getParent()) != T)
              TInputs.insert(V);
        }

    for (Spindle *S : depth_first<InTask<Spindle *>>(T->getEntrySpindle())) {
      for (BasicBlock *BB : S->blocks()) {
        // Skip basic blocks that are successors of detached rethrows.  They're
        // dead anyway.
        if (isSuccessorOfDetachedRethrow(BB))
          continue;

        // If a used value is defined outside the region, it's an input.  If an
        // instruction is used outside the region, it's an output.
        for (Instruction &II : *BB) {
          if (isa<DetachInst>(II) || isa<ReattachInst>(II) || isa<SyncInst>(II))
            // Skip the Tapir instructions, because they don't directly use any
            // values.  This check ensures that we ignore sync regions as inputs.
            continue;

          // Examine all operands of this instruction.
          for (User::op_iterator OI = II.op_begin(), OE = II.op_end(); OI != OE;
               ++OI) {
            // PHI nodes in the entry block of a shared-EH exit will be
            // rewritten in any cloned helper, so we skip operands of these PHI
            // nodes for blocks not in this task.
            if (S->isSharedEH() && S->isEntry(BB))
              if (PHINode *PN = dyn_cast<PHINode>(&II)) {
                DEBUG(dbgs() << "\tPHI node in shared-EH spindle: " << *PN << "\n");
                if (!T->simplyEncloses(PN->getIncomingBlock(*OI))) {
                  DEBUG(dbgs() << "skipping\n");
                  continue;
                }
              }
            // If this operand is not defined in this basic block, it's an input.
            if (isa<Argument>(*OI) || isa<GlobalVariable>(*OI))
              TInputs.insert(*OI);
            // If this operand is defined in the parent, it's an input.
            if (T->definedInParent(*OI))
              TInputs.insert(*OI);
          }
        }
      }
    }

    DEBUG({
        dbgs() << "Inputs for " << *T << "\n";
        for (Value *Input : TInputs) {
          dbgs() << "\t" << *Input;
          if (Instruction *I = dyn_cast<Instruction>(Input))
            dbgs() << " in " << I->getParent()->getName();
          else if (Argument *A = dyn_cast<Argument>(Input))
            dbgs() << " in " << A->getParent()->getName();
          dbgs() << "\n";
        }
      });
  }
}

void CSIImpl::instrumentDetach(DetachInst *DI, DominatorTree *DT, TaskInfo &TI,
                               const DenseMap<Value *, Value *> &TrackVars,
                               const InputMap<Task> &TaskInputs) {
  // Instrument the detach instruction itself
  Value *DetachID;
  {
    IRBuilder<> IRB(DI);
    csi_id_t LocalID = DetachFED.add(*DI);
    DetachID = DetachFED.localToGlobalId(LocalID, IRB);
    Value *TrackVar = TrackVars.lookup(DI->getSyncRegion());
    // TODO? Rather than use TrackVars to record the boolean of whether or not a
    // detach in this sync region has happened, we can count the number of such
    // detaches.
    IRB.CreateStore(
        Constant::getIntegerValue(IntegerType::getInt32Ty(DI->getContext()),
                                  APInt(32, 1)),
        TrackVar);
    insertHookCall(DI, CsiDetach, {DetachID, TrackVar});
  }

  // Find the detached block, continuation, and associated reattaches.
  BasicBlock *DetachedBlock = DI->getDetached();
  BasicBlock *ContinueBlock = DI->getContinue();
  SmallVector<BasicBlock *, 8> TaskExits, TaskResumes;
  SmallVector<Spindle *, 2> SharedEHExits;
  getTaskExits(DI, TaskExits, TaskResumes, SharedEHExits, TI);

  // Instrument the entry and exit points of the detached task.
  {
    // Instrument the entry point of the detached task.
    IRBuilder<> IRB(&*DetachedBlock->getFirstInsertionPt());
    csi_id_t LocalID = TaskFED.add(*DetachedBlock);
    Value *TaskID = TaskFED.localToGlobalId(LocalID, IRB);
    // Instrument inputs to the task.
    instrumentInputs(IRB, CSIDataFlowObject::Task, TaskID,
                     TaskInputs.lookup(TI.getTaskFor(DetachedBlock)));
    // Insert hook.
    Instruction *Call = IRB.CreateCall(CsiTaskEntry, {TaskID, DetachID});
    setInstrumentationDebugLoc(*DetachedBlock, Call);

    // Instrument the exit points of the detached tasks.
    for (BasicBlock *Exit : TaskExits) {
      IRBuilder<> IRB(Exit->getTerminator());
      csi_id_t LocalID = TaskExitFED.add(*Exit->getTerminator());
      Value *ExitID = TaskExitFED.localToGlobalId(LocalID, IRB);
      insertHookCall(Exit->getTerminator(), CsiTaskExit,
                     {ExitID, TaskID, DetachID});
    }
    // Instrument the EH exits of the detached task.
    for (BasicBlock *Exit : TaskResumes) {
      IRBuilder<> IRB(Exit->getTerminator());
      csi_id_t LocalID = TaskExitFED.add(*Exit->getTerminator());
      Value *ExitID = TaskExitFED.localToGlobalId(LocalID, IRB);
      insertHookCall(Exit->getTerminator(), CsiTaskExit,
                     {ExitID, TaskID, DetachID});
    }

    Task *T = TI.getTaskFor(DetachedBlock);
    Value *DefaultID = getDefaultID(IRB);
    for (Spindle *SharedEH : SharedEHExits)
      insertHookCallAtSharedEHSpindleExits(DI, SharedEH, T, CsiTaskExit,
                                           TaskExitFED, {TaskID, DetachID},
                                           {DefaultID, DefaultID});
  }

  // Instrument the continuation of the detach.
  {
    if (isCriticalContinueEdge(DI, 1))
      ContinueBlock = SplitCriticalEdge(
          DI, 1, CriticalEdgeSplittingOptions(DT).setSplitDetachContinue());

    IRBuilder<> IRB(&*ContinueBlock->getFirstInsertionPt());
    csi_id_t LocalID = DetachContinueFED.add(*ContinueBlock);
    Value *ContinueID = DetachContinueFED.localToGlobalId(LocalID, IRB);
    Instruction *Call =
        IRB.CreateCall(CsiDetachContinue, {ContinueID, DetachID});
    setInstrumentationDebugLoc(*ContinueBlock, Call);
  }
  // Instrument the unwind of the detach, if it exists.
  if (DI->hasUnwindDest()) {
    BasicBlock *UnwindBlock = DI->getUnwindDest();
    IRBuilder<> IRB(DI);
    Value *DefaultID = getDefaultID(IRB);
    csi_id_t LocalID = DetachContinueFED.add(*UnwindBlock);
    Value *ContinueID = DetachContinueFED.localToGlobalId(LocalID, IRB);
    insertHookCallInSuccessorBB(DI, UnwindBlock, DI->getParent(),
                                CsiDetachContinue,
                                {ContinueID, DetachID}, {DefaultID, DefaultID});
  }
}

void CSIImpl::instrumentSync(SyncInst *SI,
                             const DenseMap<Value *, Value *> &TrackVars) {
  IRBuilder<> IRB(SI);
  Value *DefaultID = getDefaultID(IRB);
  // Get the ID of this sync.
  csi_id_t LocalID = SyncFED.add(*SI);
  Value *SyncID = SyncFED.localToGlobalId(LocalID, IRB);

  Value *TrackVar = TrackVars.lookup(SI->getSyncRegion());

  // Insert instrumentation before the sync.
  insertHookCall(SI, CsiBeforeSync, {SyncID, TrackVar});
  CallInst *call = insertHookCallInSuccessorBB(
      SI, SI->getSuccessor(0), SI->getParent(), CsiAfterSync,
      {SyncID, TrackVar},
      {DefaultID,
       ConstantPointerNull::get(
           IntegerType::getInt32Ty(SI->getContext())->getPointerTo())});

  // Reset the tracking variable to 0.
  if (call != nullptr) {
    callsAfterSync.insert({SI->getSuccessor(0), call});
    IRB.SetInsertPoint(call->getNextNode());
    IRB.CreateStore(
        Constant::getIntegerValue(IntegerType::getInt32Ty(SI->getContext()),
                                  APInt(32, 0)),
        TrackVar);
  } else {
    assert(callsAfterSync.find(SI->getSuccessor(0)) != callsAfterSync.end());
  }
}

void CSIImpl::assignArithmeticID(Instruction *I) {
  ArithmeticFED.add(*I);
}

void CSIImpl::instrumentArithmetic(Instruction *I, LoopInfo &LI) {
  IRBuilder<> IRB(I);
  // We have to make sure not to disrupt the block of PHIs in the block.
  if (isa<PHINode>(I))
    IRB.SetInsertPoint(&*I->getParent()->getFirstInsertionPt());

  if (!ArithmeticFED.hasId(I))
    llvm_unreachable("Missing local ID for arithmetic instruction");

  csi_id_t LocalId = ArithmeticFED.lookupId(I);
  Value *CsiId = ArithmeticFED.localToGlobalId(LocalId, IRB);
  CsiArithmeticFlags Flags;
  Flags.setHasOneUse(checkHasOneUse(I, LI));
  Flags.setBBLocal(checkBBLocal(I, *I->getParent()));

  Function *ArithmeticHook = getCSIArithmeticHook(M, I, true);
  // Exit early if we don't have a hook for this op.
  if (!ArithmeticHook)
    return;
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
    // Standard binary operators, including all basic arithmetic.
    Value *Opcode = getOpcodeID(BO->getOpcode(), IRB);
    Value *Operand0 = BO->getOperand(0);
    Value *Operand1 = BO->getOperand(1);
    std::pair<Value *, Value *> Operand0ID = getOperandID(Operand0, IRB);
    std::pair<Value *, Value *> Operand1ID = getOperandID(Operand1, IRB);
    Type *OpTy = BO->getType();
    Type *OperandCastTy = getOperandCastTy(M, OpTy);
    Value *CastOperand0 = Operand0;
    Value *CastOperand1 = Operand1;
    if (OpTy->isIntegerTy()) {
      CastOperand0 = IRB.CreateZExtOrBitCast(Operand0, OperandCastTy);
      CastOperand1 = IRB.CreateZExtOrBitCast(Operand1, OperandCastTy);
    }
    Flags.copyIRFlags(BO);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook,
                   {CsiId, Opcode, Operand0ID.first, Operand0ID.second,
                    CastOperand0, Operand1ID.first, Operand1ID.second,
                    CastOperand1, FlagsVal});
    // TODO: Insert CsiAfterArithmetic hooks
    // BasicBlock::iterator Iter(I);
    // Iter++;
    // IRB.SetInsertPoint(&*Iter);
  } else if (TruncInst *TI = dyn_cast<TruncInst>(I)) {
    Value *Operand = TI->getOperand(0);
    Type *OpTy = TI->getType();
    if (OpTy->isVectorTy()) {
      dbgs() << "Uninstrumented operation " << *TI << "\n";
      return;
    }
    Type *OperandTy = Operand->getType();
    Type *OperandCastTy = getOperandCastTy(M, OperandTy);
    assert(OperandCastTy && "No type found for operand.");
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *CastOperand = IRB.CreateZExtOrBitCast(Operand, OperandCastTy);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       CastOperand, FlagsVal});
  } else if (FPTruncInst *TI = dyn_cast<FPTruncInst>(I)) {
    // Floating-point truncation
    Value *Operand = TI->getOperand(0);
    Type *OpTy = TI->getType();
    if (OpTy->isVectorTy()) {
      dbgs() << "Uninstrumented operation " << *TI << "\n";
      return;
    }
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       Operand, FlagsVal});
  } else if (ZExtInst *EI = dyn_cast<ZExtInst>(I)) {
    Value *Operand = EI->getOperand(0);
    Type *OpTy = EI->getType();
    if (OpTy->isVectorTy()) {
      dbgs() << "Uninstrumented operation " << *EI << "\n";
      return;
    }
    Type *OperandTy = Operand->getType();
    Type *OperandCastTy = getOperandCastTy(M, OperandTy);
    assert(OperandCastTy && "No type found for operand.");
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *CastOperand = IRB.CreateZExtOrBitCast(Operand, OperandCastTy);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       CastOperand, FlagsVal});
  } else if (SExtInst *EI = dyn_cast<SExtInst>(I)) {
    Value *Operand = EI->getOperand(0);
    Type *OpTy = EI->getType();
    if (OpTy->isVectorTy()) {
      dbgs() << "Uninstrumented operation " << *EI << "\n";
      return;
    }
    Type *OperandTy = Operand->getType();
    Type *OperandCastTy = getOperandCastTy(M, OperandTy);
    assert(OperandCastTy && "No type found for operand.");
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *CastOperand = IRB.CreateSExtOrBitCast(Operand, OperandCastTy);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       CastOperand, FlagsVal});
  } else if (FPExtInst *EI = dyn_cast<FPExtInst>(I)) {
    // Floating-point extension
    Value *Operand = EI->getOperand(0);
    Type *OpTy = EI->getType();
    if (OpTy->isVectorTy()) {
      dbgs() << "Uninstrumented operation " << *EI << "\n";
      return;
    }
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       Operand, FlagsVal});
  } else if (FPToUIInst *CI = dyn_cast<FPToUIInst>(I)) {
    // Floating-point conversion to unsigned integer
    Value *Operand = CI->getOperand(0);
    Type *OperandTy = Operand->getType();

    Type *OperandCastTy = getOperandCastTy(M, OperandTy);
    assert(OperandCastTy && "No type found for operand.");
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *CastOperand = IRB.CreateFPCast(Operand, OperandCastTy);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       CastOperand, FlagsVal});
  } else if (FPToSIInst *CI = dyn_cast<FPToSIInst>(I)) {
    // Floating-point conversion to signed integer
    Value *Operand = CI->getOperand(0);
    Type *OperandTy = Operand->getType();

    Type *OperandCastTy = getOperandCastTy(M, OperandTy);
    assert(OperandCastTy && "No type found for operand.");
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *CastOperand = IRB.CreateFPCast(Operand, OperandCastTy);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       CastOperand, FlagsVal});
  } else if (UIToFPInst *CI = dyn_cast<UIToFPInst>(I)) {
    // Unsigned integer conversion to floating point
    Value *Operand = CI->getOperand(0);
    Type *OperandTy = Operand->getType();

    Type *OperandCastTy = getOperandCastTy(M, OperandTy);
    assert(OperandCastTy && "No type found for operand.");
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *CastOperand = IRB.CreateZExtOrBitCast(Operand, OperandCastTy);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       CastOperand, FlagsVal});

  } else if (SIToFPInst *CI = dyn_cast<SIToFPInst>(I)) {
    // Signed integer conversion to floating point
    Value *Operand = CI->getOperand(0);
    Type *OperandTy = Operand->getType();

    Type *OperandCastTy = getOperandCastTy(M, OperandTy);
    assert(OperandCastTy && "No type found for operand.");
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *CastOperand = IRB.CreateSExtOrBitCast(Operand, OperandCastTy);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       CastOperand, FlagsVal});

  } else if (BitCastInst *BC = dyn_cast<BitCastInst>(I)) {
    // Arbitrary bit cast
    Value *Operand = BC->getOperand(0);
    Type *OperandTy = Operand->getType();
    Type *OperandCastTy = getOperandCastTy(M, OperandTy);
    assert(OperandCastTy && "No type found for operand.");
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *CastOperand = IRB.CreateSExtOrBitCast(Operand, OperandCastTy);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       CastOperand, FlagsVal});

  } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
    // GEP instruction
    // TODO? Special-case GEP's with constant offsets.
    Value *Operand = GEP->getOperand(0);
    Type *OperandTy = Operand->getType();
    Type *OperandCastTy = getOperandCastTy(M, OperandTy);
    Type *IdxOpTy = IRB.getInt64Ty();
    Type *IdxArrayElTy = StructType::get(IRB.getInt8Ty(), IRB.getInt64Ty(),
                                         IdxOpTy);
    // Handle the indices of the GEP, if it has them.
    Value *IdxArray =
      ConstantPointerNull::get(PointerType::get(IdxArrayElTy, 0));
    ConstantInt *IdxArraySize = nullptr;
    Value *NumIdxVal = IRB.getInt32(0);
    Value *StackAddr = nullptr;
    if (GEP->hasIndices()) {
      unsigned NumIdx = GEP->getNumIndices();
      NumIdxVal = IRB.getInt32(NumIdx);
      // Save information about the stack before allocating the index array.
      StackAddr =
        IRB.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::stacksave));
      // Allocate the index array.
      IdxArray = IRB.CreateAlloca(IdxArrayElTy, NumIdxVal);
      IdxArraySize =
        IRB.getInt64(DL.getTypeAllocSize(
                         cast<AllocaInst>(IdxArray)->getAllocatedType())
                     * NumIdx);
      IRB.CreateLifetimeStart(IdxArray, IdxArraySize);
      // Populate the index array.
      unsigned IdxNum = 0;
      for (Value *Idx : GEP->indices()) {
        std::pair<Value *, Value *> OperandID = getOperandID(Idx, IRB);
        IRB.CreateStore(OperandID.first,
                        IRB.CreateInBoundsGEP(IdxArray, {IRB.getInt32(IdxNum),
                                                         IRB.getInt32(0)}));
        IRB.CreateStore(OperandID.second,
                        IRB.CreateInBoundsGEP(IdxArray, {IRB.getInt32(IdxNum),
                                                         IRB.getInt32(1)}));
        IRB.CreateStore(IRB.CreateSExt(Idx, IdxOpTy),
                        IRB.CreateInBoundsGEP(IdxArray, {IRB.getInt32(IdxNum),
                                                         IRB.getInt32(2)}));
        IdxNum++;
      }
    }
    // Get information on the pointer operand.
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *CastOperand = IRB.CreateSExtOrBitCast(Operand, OperandCastTy);
    Flags.copyIRFlags(GEP);
    Value *FlagsVal = Flags.getValue(IRB);
    // Insert the hook.
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       CastOperand, IdxArray, NumIdxVal,
                                       FlagsVal});
    if (GEP->hasIndices()) {
      // Clean up the index array.
      IRB.CreateLifetimeEnd(IdxArray, IdxArraySize);
      IRB.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::stackrestore),
                     {StackAddr});
    }

  } else if (IntToPtrInst *I2P = dyn_cast<IntToPtrInst>(I)) {
    // Integer conversion to pointer
    Value *Operand = I2P->getOperand(0);
    Type *OperandTy = Operand->getType();

    Type *OperandCastTy = getOperandCastTy(M, OperandTy);
    assert(OperandCastTy && "No type found for operand.");
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *CastOperand = IRB.CreateSExtOrBitCast(Operand, OperandCastTy);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       CastOperand, FlagsVal});

  } else if (PtrToIntInst *P2I = dyn_cast<PtrToIntInst>(I)) {
    // Pointer conversion to integer
    Value *Operand = P2I->getOperand(0);
    Type *OperandTy = Operand->getType();

    Type *OperandCastTy = getOperandCastTy(M, OperandTy);
    assert(OperandCastTy && "No type found for operand.");
    std::pair<Value *, Value *> OperandID = getOperandID(Operand, IRB);
    Value *CastOperand = IRB.CreateBitCast(Operand, OperandCastTy);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook, {CsiId, OperandID.first, OperandID.second,
                                       CastOperand, FlagsVal});

  } else if (PHINode *PN = dyn_cast<PHINode>(I)) {
    Type *OpTy = PN->getType();
    Function *CsiPhiHook = ArithmeticHook;
    Type *OperandCastTy = getOperandCastTy(M, OpTy);
    if (CsiPhiHook) {
      PHINode *PHIArgs[3];
      {
        // Make sure these PHI nodes are inserted at the beginning of the block.
        IRBuilder<> ArgB(&PN->getParent()->front());
        // OperandID.first type
        PHIArgs[0] = ArgB.CreatePHI(ArgB.getInt8Ty(), PN->getNumIncomingValues());
        // OperandID.second type
        PHIArgs[1] = ArgB.CreatePHI(ArgB.getInt64Ty(),
                                    PN->getNumIncomingValues());
        // BBID type
        PHIArgs[2] = ArgB.CreatePHI(ArgB.getInt64Ty(),
                                    PN->getNumIncomingValues());
      }

      for (BasicBlock *Pred : predecessors(PN->getParent())) {
        IRBuilder<> PredB(Pred->getTerminator());
        Value *Operand = PN->getIncomingValueForBlock(Pred);
        std::pair<Value *, Value *> OperandID = getOperandID(Operand, PredB);
        PHIArgs[0]->addIncoming(OperandID.first, Pred);
        PHIArgs[1]->addIncoming(OperandID.second, Pred);
        // Basic-block ID of the predecessor.
        if (const Instruction *OpI = dyn_cast<Instruction>(Operand))
          PHIArgs[2]->addIncoming(
              IRB.getInt64(BasicBlockFED.lookupId(OpI->getParent())), Pred);
        else
          PHIArgs[2]->addIncoming(IRB.getInt64(CsiUnknownId), Pred);
      }

      csi_id_t LocalBBID = BasicBlockFED.lookupId(PN->getParent());
      Value *BBID = BasicBlockFED.localToGlobalId(LocalBBID, IRB);
      Value *SrcID = BasicBlockFED.localToGlobalId(PHIArgs[2], IRB);
      Value *CastPN = PN;
      if (OperandCastTy != OpTy)
        CastPN = IRB.CreateZExtOrBitCast(PN, OperandCastTy);
      Value *FlagsVal = Flags.getValue(IRB);

      // Don't use insertHookCall for PHI instrumentation, because we must make
      // sure not to disrupt the PHIs in the block.
      CallInst *Call = IRB.CreateCall(CsiPhiHook, {CsiId, BBID, SrcID,
                                                   PHIArgs[0], PHIArgs[1],
                                                   CastPN, FlagsVal});
      setInstrumentationDebugLoc(I, (Instruction *)Call);
    }
  } else if (CmpInst *Cmp = dyn_cast<CmpInst>(I)) {
    // Integer or floating-point comparison
    Value *Pred = getPredicateID(Cmp->getPredicate(), IRB);
    Value *Operand0 = Cmp->getOperand(0);
    Value *Operand1 = Cmp->getOperand(1);
    std::pair<Value *, Value *> Operand0ID = getOperandID(Operand0, IRB);
    std::pair<Value *, Value *> Operand1ID = getOperandID(Operand1, IRB);
    Type *Operand0CastTy = getOperandCastTy(M, Operand0->getType());
    Type *Operand1CastTy = getOperandCastTy(M, Operand1->getType());
    Value *CastOperand0 = Operand0;
    Value *CastOperand1 = Operand1;
    if (Cmp->isIntPredicate()) {
      CastOperand0 = IRB.CreateZExtOrBitCast(Operand0, Operand0CastTy);
      CastOperand1 = IRB.CreateZExtOrBitCast(Operand1, Operand1CastTy);
    }
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook,
                   {CsiId, Pred, Operand0ID.first, Operand0ID.second,
                    CastOperand0, Operand1ID.first, Operand1ID.second,
                    CastOperand1, FlagsVal});
    // TODO: Insert CsiAfterArithmetic hooks
    // BasicBlock::iterator Iter(I);
    // Iter++;
    // IRB.SetInsertPoint(&*Iter);

  } else if (InsertElementInst *IE = dyn_cast<InsertElementInst>(I)) {
    Value *Operand0 = IE->getOperand(0);
    Value *Operand1 = IE->getOperand(1);
    Value *Operand2 = IE->getOperand(2);
    std::pair<Value *, Value *> Operand0ID = getOperandID(Operand0, IRB);
    std::pair<Value *, Value *> Operand1ID = getOperandID(Operand1, IRB);
    std::pair<Value *, Value *> Operand2ID = getOperandID(Operand2, IRB);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook,
                   {CsiId, Operand0ID.first, Operand0ID.second, Operand0,
                    Operand1ID.first, Operand1ID.second, Operand1,
                    Operand2ID.first, Operand2ID.second, Operand2,
                    FlagsVal});

  } else if (ExtractElementInst *EE = dyn_cast<ExtractElementInst>(I)) {
    Value *Operand0 = EE->getOperand(0);
    Value *Operand1 = EE->getOperand(1);
    std::pair<Value *, Value *> Operand0ID = getOperandID(Operand0, IRB);
    std::pair<Value *, Value *> Operand1ID = getOperandID(Operand1, IRB);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook,
                   {CsiId, Operand0ID.first, Operand0ID.second, Operand0,
                    Operand1ID.first, Operand1ID.second, Operand1,
                    FlagsVal});

  } else if (ShuffleVectorInst *SV = dyn_cast<ShuffleVectorInst>(I)) {
    Value *Operand0 = SV->getOperand(0);
    Value *Operand1 = SV->getOperand(1);
    Value *Operand2 = SV->getOperand(2);
    std::pair<Value *, Value *> Operand0ID = getOperandID(Operand0, IRB);
    std::pair<Value *, Value *> Operand1ID = getOperandID(Operand1, IRB);
    std::pair<Value *, Value *> Operand2ID = getOperandID(Operand2, IRB);
    Value *FlagsVal = Flags.getValue(IRB);
    insertHookCall(I, ArithmeticHook,
                   {CsiId, Operand0ID.first, Operand0ID.second, Operand0,
                    Operand1ID.first, Operand1ID.second, Operand1,
                    Operand2ID.first, Operand2ID.second, Operand2,
                    FlagsVal});
  }
}

void CSIImpl::assignAllocaID(Instruction *I) {
  AllocaFED.add(*I);
}

void CSIImpl::instrumentAlloca(Instruction *I) {
  IRBuilder<> IRB(I);
  AllocaInst *AI = cast<AllocaInst>(I);

  csi_id_t LocalId = AllocaFED.lookupId(I);
  Value *CsiId = AllocaFED.localToGlobalId(LocalId, IRB);

  CsiAllocaProperty Prop;
  Prop.setIsStatic(AI->isStaticAlloca());
  // Set may-be-captured property
  Prop.setMayBeCaptured(PointerMayBeCaptured(AI, false, true));
  Value *PropVal = Prop.getValue(IRB);

  // Get size of allocation.
  uint64_t Size = DL.getTypeAllocSize(AI->getAllocatedType());
  Value *SizeVal = IRB.getInt64(Size);
  if (AI->isArrayAllocation())
    SizeVal = IRB.CreateMul(SizeVal, AI->getArraySize());

  insertHookCall(I, CsiBeforeAlloca, {CsiId, SizeVal, PropVal});
  BasicBlock::iterator Iter(I);
  Iter++;
  IRB.SetInsertPoint(&*Iter);

  Type *AddrType = IRB.getInt8PtrTy();
  Value *Addr = IRB.CreatePointerCast(I, AddrType);
  insertHookCall(&*Iter, CsiAfterAlloca, {CsiId, Addr, SizeVal, PropVal});
}

void CSIImpl::getAllocFnArgs(const Instruction *I,
                             SmallVectorImpl<Value *> &AllocFnArgs,
                             Type *SizeTy, Type *AddrTy,
                             const TargetLibraryInfo &TLI) {
  const Function *Called = nullptr;
  if (const CallInst *CI = dyn_cast<CallInst>(I))
    Called = CI->getCalledFunction();
  else if (const InvokeInst *II = dyn_cast<InvokeInst>(I))
    Called = II->getCalledFunction();

  LibFunc F;
  bool FoundLibFunc = TLI.getLibFunc(*Called, F);
  if (!FoundLibFunc)
    return;

  switch (F) {
  default:
    return;
    // TODO: Add aligned new's to this list after they're added to TLI.
  case LibFunc_malloc:
  case LibFunc_valloc:
  case LibFunc_Znwj:
  case LibFunc_ZnwjRKSt9nothrow_t:
  case LibFunc_Znwm:
  case LibFunc_ZnwmRKSt9nothrow_t:
  case LibFunc_Znaj:
  case LibFunc_ZnajRKSt9nothrow_t:
  case LibFunc_Znam:
  case LibFunc_ZnamRKSt9nothrow_t:
  case LibFunc_msvc_new_int:
  case LibFunc_msvc_new_int_nothrow:
  case LibFunc_msvc_new_longlong:
  case LibFunc_msvc_new_longlong_nothrow:
  case LibFunc_msvc_new_array_int:
  case LibFunc_msvc_new_array_int_nothrow:
  case LibFunc_msvc_new_array_longlong:
  case LibFunc_msvc_new_array_longlong_nothrow: {
    // Allocated size
    if (isa<CallInst>(I))
      AllocFnArgs.push_back(cast<CallInst>(I)->getArgOperand(0));
    else
      AllocFnArgs.push_back(cast<InvokeInst>(I)->getArgOperand(0));
    // Number of elements = 1
    AllocFnArgs.push_back(ConstantInt::get(SizeTy, 1));
    // Alignment = 0
    // TODO: Fix this for aligned new's, once they're added to TLI.
    AllocFnArgs.push_back(ConstantInt::get(SizeTy, 0));
    // Old pointer = NULL
    AllocFnArgs.push_back(Constant::getNullValue(AddrTy));
    return;
  }
  case LibFunc_calloc: {
    const CallInst *CI = cast<CallInst>(I);
    // Allocated size
    AllocFnArgs.push_back(CI->getArgOperand(1));
    // Number of elements
    AllocFnArgs.push_back(CI->getArgOperand(0));
    // Alignment = 0
    AllocFnArgs.push_back(ConstantInt::get(SizeTy, 0));
    // Old pointer = NULL
    AllocFnArgs.push_back(Constant::getNullValue(AddrTy));
    return;
  }
  case LibFunc_realloc:
  case LibFunc_reallocf: {
    const CallInst *CI = cast<CallInst>(I);
    // Allocated size
    AllocFnArgs.push_back(CI->getArgOperand(1));
    // Number of elements = 1
    AllocFnArgs.push_back(ConstantInt::get(SizeTy, 1));
    // Alignment = 0
    AllocFnArgs.push_back(ConstantInt::get(SizeTy, 0));
    // Old pointer
    AllocFnArgs.push_back(CI->getArgOperand(0));
    return;
  }
  }
}

void CSIImpl::assignAllocFnID(Instruction *I) {
  AllocFnFED.add(*I);
}

void CSIImpl::instrumentAllocFn(Instruction *I, DominatorTree *DT) {
  bool IsInvoke = isa<InvokeInst>(I);
  Function *Called = nullptr;
  if (CallInst *CI = dyn_cast<CallInst>(I))
    Called = CI->getCalledFunction();
  else if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    Called = II->getCalledFunction();

  assert(Called && "Could not get called function for allocation fn.");

  IRBuilder<> IRB(I);
  Value *DefaultID = getDefaultID(IRB);
  csi_id_t LocalId = AllocFnFED.lookupId(I);
  Value *AllocFnId = AllocFnFED.localToGlobalId(LocalId, IRB);

  SmallVector<Value *, 4> AllocFnArgs;
  getAllocFnArgs(I, AllocFnArgs, IntptrTy, IRB.getInt8PtrTy(), *TLI);
  SmallVector<Value *, 4> DefaultAllocFnArgs({
      /* Allocated size */ Constant::getNullValue(IntptrTy),
      /* Number of elements */ Constant::getNullValue(IntptrTy),
      /* Alignment */ Constant::getNullValue(IntptrTy),
      /* Old pointer */ Constant::getNullValue(IRB.getInt8PtrTy()),
  });

  CsiAllocFnProperty Prop;
  // Set may-be-captured property
  Prop.setMayBeCaptured(PointerMayBeCaptured(I, true, true));
  Value *DefaultPropVal = Prop.getValue(IRB);
  LibFunc AllocLibF;
  TLI->getLibFunc(*Called, AllocLibF);
  Prop.setAllocFnTy(static_cast<unsigned>(getAllocFnTy(AllocLibF)));
  AllocFnArgs.push_back(Prop.getValue(IRB));
  DefaultAllocFnArgs.push_back(DefaultPropVal);

  // Instrument the call
  {
    SmallVector<Value *, 4> BeforeAllocFnArgs;
    BeforeAllocFnArgs.push_back(AllocFnId);
    BeforeAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
    insertHookCall(I, CsiBeforeAllocFn, BeforeAllocFnArgs);
  }

  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block and the
    // exception block.
    InvokeInst *II = cast<InvokeInst>(I);

    BasicBlock *NormalBB = II->getNormalDest();
    unsigned SuccNum = GetSuccessorNumber(II->getParent(), NormalBB);
    if (isCriticalEdge(II, SuccNum))
      NormalBB =
          SplitCriticalEdge(II, SuccNum, CriticalEdgeSplittingOptions(DT));
    // Insert hook into normal destination.
    {
      IRB.SetInsertPoint(&*NormalBB->getFirstInsertionPt());
      SmallVector<Value *, 4> AfterAllocFnArgs;
      AfterAllocFnArgs.push_back(AllocFnId);
      AfterAllocFnArgs.push_back(IRB.CreatePointerCast(I, IRB.getInt8PtrTy()));
      AfterAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
      insertHookCall(&*IRB.GetInsertPoint(), CsiAfterAllocFn, AfterAllocFnArgs);
    }
    // Insert hook into unwind destination.
    {
      // The return value of the allocation function is not valid in the unwind
      // destination.
      SmallVector<Value *, 4> AfterAllocFnArgs, DefaultAfterAllocFnArgs;
      AfterAllocFnArgs.push_back(AllocFnId);
      AfterAllocFnArgs.push_back(Constant::getNullValue(IRB.getInt8PtrTy()));
      AfterAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
      DefaultAfterAllocFnArgs.push_back(DefaultID);
      DefaultAfterAllocFnArgs.push_back(
          Constant::getNullValue(IRB.getInt8PtrTy()));
      DefaultAfterAllocFnArgs.append(DefaultAllocFnArgs.begin(),
                                     DefaultAllocFnArgs.end());
      insertHookCallInSuccessorBB(II, II->getUnwindDest(), II->getParent(),
                                  CsiAfterAllocFn, AfterAllocFnArgs,
                                  DefaultAfterAllocFnArgs);
    }
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    SmallVector<Value *, 4> AfterAllocFnArgs;
    AfterAllocFnArgs.push_back(AllocFnId);
    AfterAllocFnArgs.push_back(IRB.CreatePointerCast(I, IRB.getInt8PtrTy()));
    AfterAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
    insertHookCall(&*Iter, CsiAfterAllocFn, AfterAllocFnArgs);
  }
}

void CSIImpl::instrumentFree(Instruction *I) {
  // It appears that frees (and deletes) never throw.
  assert(isa<CallInst>(I) && "Free call is not a call instruction");

  CallInst *FC = cast<CallInst>(I);
  Function *Called = FC->getCalledFunction();
  assert(Called && "Could not get called function for free.");

  IRBuilder<> IRB(I);
  csi_id_t LocalId = FreeFED.add(*I);
  Value *FreeId = FreeFED.localToGlobalId(LocalId, IRB);

  Value *Addr = FC->getArgOperand(0);
  CsiFreeProperty Prop;
  LibFunc FreeLibF;
  TLI->getLibFunc(*Called, FreeLibF);
  Prop.setFreeTy(static_cast<unsigned>(getFreeTy(FreeLibF)));

  insertHookCall(I, CsiBeforeFree, {FreeId, Addr, Prop.getValue(IRB)});
  BasicBlock::iterator Iter(I);
  Iter++;
  insertHookCall(&*Iter, CsiAfterFree, {FreeId, Addr, Prop.getValue(IRB)});
}

CallInst *CSIImpl::insertHookCall(Instruction *I, Function *HookFunction,
                                  ArrayRef<Value *> HookArgs) {
  IRBuilder<> IRB(I);
  CallInst *Call = IRB.CreateCall(HookFunction, HookArgs);
  setInstrumentationDebugLoc(I, (Instruction *)Call);
  return Call;
}

bool CSIImpl::updateArgPHIs(Instruction *I, BasicBlock *Succ, BasicBlock *BB,
                            ArrayRef<Value *> HookArgs,
                            ArrayRef<Value *> DefaultArgs) {
  auto Key = std::make_pair(I, Succ);
  // If we've already created a PHI node in this block for the hook arguments,
  // just add the incoming arguments to the PHIs.
  if (ArgPHIs.count(Key)) {
    unsigned HookArgNum = 0;
    for (PHINode *ArgPHI : ArgPHIs[Key]) {
      ArgPHI->setIncomingValue(ArgPHI->getBasicBlockIndex(BB),
                               HookArgs[HookArgNum]);
      ++HookArgNum;
    }
    return true;
  }

  // Create PHI nodes in this block for each hook argument.
  IRBuilder<> IRB(&Succ->front());
  unsigned HookArgNum = 0;
  for (Value *Arg : HookArgs) {
    PHINode *ArgPHI = IRB.CreatePHI(Arg->getType(), 2);
    for (BasicBlock *Pred : predecessors(Succ)) {
      if (Pred == BB)
        ArgPHI->addIncoming(Arg, BB);
      else
        ArgPHI->addIncoming(DefaultArgs[HookArgNum], Pred);
    }
    ArgPHIs[Key].push_back(ArgPHI);
    ++HookArgNum;
  }
  return false;
}

CallInst *CSIImpl::insertHookCallInSuccessorBB(Instruction *I, BasicBlock *Succ,
                                               BasicBlock *BB,
                                               Function *HookFunction,
                                               ArrayRef<Value *> HookArgs,
                                               ArrayRef<Value *> DefaultArgs) {
  assert(HookFunction && "No hook function given.");
  // If this successor block has a unique predecessor, just insert the hook call
  // as normal.
  if (Succ->getUniquePredecessor()) {
    assert(Succ->getUniquePredecessor() == BB &&
           "BB is not unique predecessor of successor block");
    return insertHookCall(&*Succ->getFirstInsertionPt(), HookFunction,
                          HookArgs);
  }

  if (updateArgPHIs(I, Succ, BB, HookArgs, DefaultArgs))
    return nullptr;

  SmallVector<Value *, 2> SuccessorHookArgs;
  for (PHINode *ArgPHI : ArgPHIs[std::make_pair(I, Succ)])
    SuccessorHookArgs.push_back(ArgPHI);

  IRBuilder<> IRB(&*Succ->getFirstInsertionPt());
  // Insert the hook call, using the PHI as the CSI ID.
  CallInst *Call = IRB.CreateCall(HookFunction, SuccessorHookArgs);
  setInstrumentationDebugLoc(*Succ, (Instruction *)Call);

  return Call;
}

void CSIImpl::insertHookCallAtSharedEHSpindleExits(
    Instruction *I, Spindle *SharedEHSpindle, Task *T, Function *HookFunction,
    FrontEndDataTable &FED, ArrayRef<Value *> HookArgs,
    ArrayRef<Value *> DefaultArgs) {
  // Get the set of shared EH spindles to examine.  Store them in post order, so
  // they can be evaluated in reverse post order.
  SmallVector<Spindle *, 2> WorkList;
  for (Spindle *S : post_order<InTask<Spindle *>>(SharedEHSpindle))
    WorkList.push_back(S);

  // Traverse the shared-EH spindles in reverse post order, updating the
  // hook-argument PHI's along the way.
  SmallPtrSet<Spindle *, 2> Visited;
  for (Spindle *S : llvm::reverse(WorkList)) {
    bool NewPHINode = false;
    // If this spindle is the first shared-EH spindle in the traversal, use the
    // given hook arguments to update the PHI node.
    if (S == SharedEHSpindle) {
      for (Spindle::SpindleEdge &InEdge : S->in_edges()) {
        Spindle *SPred = InEdge.first;
        BasicBlock *Pred = InEdge.second;
        if (T->contains(SPred))
          NewPHINode |=
              updateArgPHIs(I, S->getEntry(), Pred, HookArgs, DefaultArgs);
      }
    } else {
      // Otherwise update the PHI node based on the predecessor shared-eh
      // spindles in this RPO traversal.
      for (Spindle::SpindleEdge &InEdge : S->in_edges()) {
        Spindle *SPred = InEdge.first;
        BasicBlock *Pred = InEdge.second;
        if (Visited.count(SPred)) {
          auto Key = std::make_pair(I, SPred->getEntry());
          SmallVector<Value *, 4> NewHookArgs(
              ArgPHIs[Key].begin(), ArgPHIs[Key].end());
          NewPHINode |=
              updateArgPHIs(I, S->getEntry(), Pred, NewHookArgs, DefaultArgs);
        }
      }
    }
    Visited.insert(S);

    if (!NewPHINode)
      continue;

    // Detached-rethrow exits can appear in strange places within a task-exiting
    // spindle.  Hence we loop over all blocks in the spindle to find detached
    // rethrows.
    for (BasicBlock *B : S->blocks()) {
      if (isDetachedRethrow(B->getTerminator())) {
        IRBuilder<> IRB(B->getTerminator());
        csi_id_t LocalID = FED.add(*B->getTerminator());
        Value *HookID = FED.localToGlobalId(LocalID, IRB);
        SmallVector<Value *, 4> Args({HookID});
        auto Key = std::make_pair(I, S->getEntry());
        Args.append(ArgPHIs[Key].begin(), ArgPHIs[Key].end());
        Instruction *Call = IRB.CreateCall(HookFunction, Args);
        setInstrumentationDebugLoc(*B, Call);
      }
    }
  }
}

void CSIImpl::initializeFEDTables() {
  FunctionFED = FrontEndDataTable(M, CsiFunctionBaseIdName,
                                  "__csi_unit_fed_table_function",
                                  "__csi_unit_function_name_");
  FunctionExitFED = FrontEndDataTable(M, CsiFunctionExitBaseIdName,
                                      "__csi_unit_fed_table_function_exit",
                                      "__csi_unit_function_name_");
  LoopFED = FrontEndDataTable(M, CsiLoopBaseIdName,
                              "__csi_unit_fed_table_loop");
  LoopExitFED = FrontEndDataTable(M, CsiLoopExitBaseIdName,
                                  "__csi_unit_fed_table_loop");
  BasicBlockFED = FrontEndDataTable(M, CsiBasicBlockBaseIdName,
                                    "__csi_unit_fed_table_basic_block");
  CallsiteFED = FrontEndDataTable(M, CsiCallsiteBaseIdName,
                                  "__csi_unit_fed_table_callsite",
                                  "__csi_unit_function_name_");
  LoadFED = FrontEndDataTable(M, CsiLoadBaseIdName,
                              "__csi_unit_fed_table_load");
  StoreFED = FrontEndDataTable(M, CsiStoreBaseIdName,
                               "__csi_unit_fed_table_store");
  AllocaFED = FrontEndDataTable(M, CsiAllocaBaseIdName,
                                "__csi_unit_fed_table_alloca",
                                "__csi_unit_variable_name_");
  DetachFED = FrontEndDataTable(M, CsiDetachBaseIdName,
                                "__csi_unit_fed_table_detach");
  TaskFED = FrontEndDataTable(M, CsiTaskBaseIdName,
                              "__csi_unit_fed_table_task");
  TaskExitFED = FrontEndDataTable(M, CsiTaskExitBaseIdName,
                                  "__csi_unit_fed_table_task_exit");
  DetachContinueFED = FrontEndDataTable(M, CsiDetachContinueBaseIdName,
                                        "__csi_unit_fed_table_detach_continue");
  SyncFED = FrontEndDataTable(M, CsiSyncBaseIdName,
                              "__csi_unit_fed_table_sync");
  AllocFnFED = FrontEndDataTable(M, CsiAllocFnBaseIdName,
                                 "__csi_unit_fed_table_allocfn",
                                 "__csi_unit_variable_name_");
  FreeFED = FrontEndDataTable(M, CsiFreeBaseIdName,
                              "__csi_unit_fed_free");
  ArithmeticFED = FrontEndDataTable(M, CsiArithmeticBaseIdName,
                                    "__csi_unit_fed_table_arithmetic");
  ParameterFED = FrontEndDataTable(M, CsiParameterBaseIdName,
                                   "__csi_unit_fed_parameter",
                                   "__csi_unit_argument_name_");
  GlobalFED = FrontEndDataTable(M, CsiGlobalBaseIdName,
                                "__csi_unit_fed_table_global",
                                "__csi_unit_global_name_");
}

void CSIImpl::initializeSizeTables() {
  BBSize = SizeTable(M, CsiBasicBlockBaseIdName);
}

csi_id_t CSIImpl::getLocalFunctionID(Function &F) {
  csi_id_t LocalId = FunctionFED.add(F);
  FuncOffsetMap[F.getName()] = LocalId;
  return LocalId;
}

void CSIImpl::generateInitCallsiteToFunction() {
  LLVMContext &C = M.getContext();
  BasicBlock *EntryBB = BasicBlock::Create(C, "", InitCallsiteToFunction);
  IRBuilder<> IRB(ReturnInst::Create(C, EntryBB));

  GlobalVariable *Base = FunctionFED.baseId();
  LoadInst *LI = IRB.CreateLoad(Base);
  // Traverse the map of function name -> function local id. Generate
  // a store of each function's global ID to the corresponding weak
  // global variable.
  for (const auto &it : FuncOffsetMap) {
    std::string GVName = CsiFuncIdVariablePrefix + it.first.str();
    GlobalVariable *GV = nullptr;
    if ((GV = M.getGlobalVariable(GVName)) == nullptr) {
      GV = new GlobalVariable(M, IRB.getInt64Ty(), false,
                              GlobalValue::WeakAnyLinkage,
                              IRB.getInt64(CsiCallsiteUnknownTargetId), GVName);
    }
    assert(GV);
    IRB.CreateStore(IRB.CreateAdd(LI, IRB.getInt64(it.second)), GV);
  }

  GlobalVariable *GlobalBase = GlobalFED.baseId();
  LI = IRB.CreateLoad(GlobalBase);
  for (const auto &it : GlobalOffsetMap) {
    std::string GVName = CsiGlobalIdVariablePrefix + it.first.str();
    GlobalVariable *GV = nullptr;
    if ((GV = M.getGlobalVariable(GVName)) == nullptr) {
      GV = new GlobalVariable(M, IRB.getInt64Ty(), false,
                              GlobalValue::WeakAnyLinkage,
                              IRB.getInt64(CsiUnknownId), GVName);
    }
    assert(GV);
    IRB.CreateStore(IRB.CreateAdd(LI, IRB.getInt64(it.second)), GV);
  }
}

void CSIImpl::initializeCsi() {
  IntptrTy = DL.getIntPtrType(M.getContext());

  initializeFEDTables();
  initializeSizeTables();
  if (Options.InstrumentFuncEntryExit)
    initializeFuncHooks();
  if (Options.InstrumentMemoryAccesses)
    initializeLoadStoreHooks();
  if (Options.InstrumentBasicBlocks)
    initializeBasicBlockHooks();
  if (Options.InstrumentLoops)
    initializeLoopHooks();
  if (Options.InstrumentCalls)
    initializeCallsiteHooks();
  if (Options.InstrumentMemIntrinsics)
    initializeMemIntrinsicsHooks();
  if (Options.InstrumentTapir)
    initializeTapirHooks();
  if (Options.InstrumentAllocas)
    initializeAllocaHooks();
  if (Options.InstrumentAllocFns)
    initializeAllocFnHooks();

  FunctionType *FnType =
      FunctionType::get(Type::getVoidTy(M.getContext()), {}, false);
  InitCallsiteToFunction = checkCsiInterfaceFunction(
      M.getOrInsertFunction(CsiInitCallsiteToFunctionName, FnType));
  assert(InitCallsiteToFunction);
  InitCallsiteToFunction->setLinkage(GlobalValue::InternalLinkage);

  /*
  The runtime declares this as a __thread var --- need to change this decl
  generation or the tool won't compile DisableInstrGV = new GlobalVariable(M,
  IntegerType::get(M.getContext(), 1), false, GlobalValue::ExternalLinkage,
  nullptr, CsiDisableInstrumentationName, nullptr,
                                      GlobalValue::GeneralDynamicTLSModel, 0,
  true);
  */
}

// Create a struct type to match the unit_fed_entry_t type in csirt.c.
StructType *CSIImpl::getUnitFedTableType(LLVMContext &C,
                                         PointerType *EntryPointerType) {
  return StructType::get(IntegerType::get(C, 64), Type::getInt8PtrTy(C, 0),
                         EntryPointerType);
}

Constant *CSIImpl::fedTableToUnitFedTable(Module &M,
                                          StructType *UnitFedTableType,
                                          FrontEndDataTable &FedTable) {
  Constant *NumEntries =
      ConstantInt::get(IntegerType::get(M.getContext(), 64), FedTable.size());
  Constant *BaseIdPtr = ConstantExpr::getPointerCast(
      FedTable.baseId(), Type::getInt8PtrTy(M.getContext(), 0));
  Constant *InsertedTable = FedTable.insertIntoModule(M);
  return ConstantStruct::get(UnitFedTableType, NumEntries, BaseIdPtr,
                             InsertedTable);
}

void CSIImpl::collectUnitFEDTables() {
  LLVMContext &C = M.getContext();
  StructType *UnitFedTableType =
      getUnitFedTableType(C, FrontEndDataTable::getPointerType(C));

  // The order of the FED tables here must match the enum in csirt.c and the
  // instrumentation_counts_t in csi.h.
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, FunctionFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, FunctionExitFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, LoopFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, LoopExitFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, BasicBlockFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, CallsiteFED));
  UnitFedTables.push_back(fedTableToUnitFedTable(M, UnitFedTableType, LoadFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, StoreFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, DetachFED));
  UnitFedTables.push_back(fedTableToUnitFedTable(M, UnitFedTableType, TaskFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, TaskExitFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, DetachContinueFED));
  UnitFedTables.push_back(fedTableToUnitFedTable(M, UnitFedTableType, SyncFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, AllocaFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, AllocFnFED));
  UnitFedTables.push_back(fedTableToUnitFedTable(M, UnitFedTableType, FreeFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, ArithmeticFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, ParameterFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, GlobalFED));
}

// Create a struct type to match the unit_obj_entry_t type in csirt.c.
StructType *CSIImpl::getUnitSizeTableType(LLVMContext &C,
                                          PointerType *EntryPointerType) {
  return StructType::get(IntegerType::get(C, 64), EntryPointerType);
}

Constant *CSIImpl::sizeTableToUnitSizeTable(Module &M,
                                            StructType *UnitSizeTableType,
                                            SizeTable &SzTable) {
  Constant *NumEntries =
      ConstantInt::get(IntegerType::get(M.getContext(), 64), SzTable.size());
  // Constant *BaseIdPtr =
  //   ConstantExpr::getPointerCast(FedTable.baseId(),
  //                                Type::getInt8PtrTy(M.getContext(), 0));
  Constant *InsertedTable = SzTable.insertIntoModule(M);
  return ConstantStruct::get(UnitSizeTableType, NumEntries, InsertedTable);
}

void CSIImpl::collectUnitSizeTables() {
  LLVMContext &C = M.getContext();
  StructType *UnitSizeTableType =
      getUnitSizeTableType(C, SizeTable::getPointerType(C));

  UnitSizeTables.push_back(
      sizeTableToUnitSizeTable(M, UnitSizeTableType, BBSize));
}

CallInst *CSIImpl::createRTUnitInitCall(IRBuilder<> &IRB) {
  LLVMContext &C = M.getContext();

  StructType *UnitFedTableType =
      getUnitFedTableType(C, FrontEndDataTable::getPointerType(C));
  StructType *UnitSizeTableType =
      getUnitSizeTableType(C, SizeTable::getPointerType(C));

  // Lookup __csirt_unit_init
  SmallVector<Type *, 4> InitArgTypes({IRB.getInt8PtrTy(),
                                       PointerType::get(UnitFedTableType, 0),
                                       PointerType::get(UnitSizeTableType, 0),
                                       InitCallsiteToFunction->getType()});
  FunctionType *InitFunctionTy =
      FunctionType::get(IRB.getVoidTy(), InitArgTypes, false);
  RTUnitInit = checkCsiInterfaceFunction(
      M.getOrInsertFunction(CsiRtUnitInitName, InitFunctionTy));
  assert(RTUnitInit);

  ArrayType *UnitFedTableArrayType =
      ArrayType::get(UnitFedTableType, UnitFedTables.size());
  Constant *FEDTable = ConstantArray::get(UnitFedTableArrayType, UnitFedTables);
  GlobalVariable *FEDGV = new GlobalVariable(
      M, UnitFedTableArrayType, false, GlobalValue::InternalLinkage, FEDTable,
      CsiUnitFedTableArrayName);
  ArrayType *UnitSizeTableArrayType =
      ArrayType::get(UnitSizeTableType, UnitSizeTables.size());
  Constant *SzTable =
      ConstantArray::get(UnitSizeTableArrayType, UnitSizeTables);
  GlobalVariable *SizeGV = new GlobalVariable(
      M, UnitSizeTableArrayType, false, GlobalValue::InternalLinkage, SzTable,
      CsiUnitSizeTableArrayName);

  Constant *Zero = ConstantInt::get(IRB.getInt32Ty(), 0);
  Value *GepArgs[] = {Zero, Zero};

  // Insert call to __csirt_unit_init
  return IRB.CreateCall(
      RTUnitInit,
      {IRB.CreateGlobalStringPtr(M.getName()),
       ConstantExpr::getGetElementPtr(FEDGV->getValueType(), FEDGV, GepArgs),
       ConstantExpr::getGetElementPtr(SizeGV->getValueType(), SizeGV, GepArgs),
       InitCallsiteToFunction});
}

void CSIImpl::finalizeCsi() {
  LLVMContext &C = M.getContext();

  // Add CSI global constructor, which calls unit init.
  Function *Ctor =
      Function::Create(FunctionType::get(Type::getVoidTy(C), false),
                       GlobalValue::InternalLinkage, CsiRtUnitCtorName, &M);
  BasicBlock *CtorBB = BasicBlock::Create(C, "", Ctor);
  IRBuilder<> IRB(ReturnInst::Create(C, CtorBB));

  // Insert __csi_func_id_<f> weak symbols for all defined functions and
  // generate the runtime code that stores to all of them.
  generateInitCallsiteToFunction();

  CallInst *Call = createRTUnitInitCall(IRB);

  // Add the constructor to the global list
  appendToGlobalCtors(M, Ctor, CsiUnitCtorPriority);

  CallGraphNode *CNCtor = CG->getOrInsertFunction(Ctor);
  CallGraphNode *CNFunc = CG->getOrInsertFunction(RTUnitInit);
  CNCtor->addCalledFunction(Call, CNFunc);
}

void llvm::CSIImpl::linkInToolFromBitcode(const std::string &bitcodePath) {
  if (bitcodePath != "") {
    std::unique_ptr<Module> toolModule;

    SMDiagnostic error;
    auto m = parseIRFile(bitcodePath, error, M.getContext());
    if (m) {
      toolModule = std::move(m);
    } else {
      llvm::errs() << "Error loading bitcode (" << bitcodePath
                   << "): " << error.getMessage() << "\n";
      report_fatal_error(error.getMessage());
    }

    std::vector<std::string> functions;

    for (Function &F : *toolModule) {
      if (!F.isDeclaration() && F.hasName()) {
        functions.push_back(F.getName());
      }
    }

    std::vector<std::string> globalVariables;

    std::vector<GlobalValue *> toRemove;
    for (GlobalValue &val : toolModule->getGlobalList()) {
      if (!val.isDeclaration()) {
        if (val.hasName() && (val.getName() == "llvm.global_ctors" ||
                              val.getName() == "llvm.global_dtors")) {
          toRemove.push_back(&val);
          continue;
        }

        // We can't have globals with internal linkage due to how compile-time
        // instrumentation works. Treat "static" variables as non-static.
        if (val.getLinkage() == GlobalValue::InternalLinkage)
          val.setLinkage(llvm::GlobalValue::CommonLinkage);

        if (val.hasName())
          globalVariables.push_back(val.getName());
      }
    }

    // We remove global constructors and destructors because they'll be linked
    // in at link time when the tool is linked. We can't have duplicates for
    // each translation unit.
    for (auto &val : toRemove) {
      val->eraseFromParent();
    }

    llvm::Linker linker(M);

    linker.linkInModule(std::move(toolModule),
                        llvm::Linker::Flags::LinkOnlyNeeded);

    // Set all tool's globals and functions to be "available externally" so
    // the linker won't complain about multiple definitions.
    for (auto &globalVariableName : globalVariables) {
      auto var = M.getGlobalVariable(globalVariableName);

      if (var && !var->isDeclaration() && !var->hasComdat()) {
        var->setLinkage(
            llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage);
      }
    }
    for (auto &functionName : functions) {
      auto function = M.getFunction(functionName);
      if (function && !function->isDeclaration() && !function->hasComdat()) {
        function->setLinkage(
            GlobalValue::LinkageTypes::AvailableExternallyLinkage);
      }
    }
  }
}

void llvm::CSIImpl::loadConfiguration() {
  if (ClConfigurationFilename != "")
    Config = InstrumentationConfig::ReadFromConfigurationFile(
        ClConfigurationFilename);
  else
    Config = InstrumentationConfig::GetDefault();

  Config->SetConfigMode(ClConfigurationMode);
}

bool CSIImpl::shouldNotInstrumentFunction(Function &F) {
  // Don't instrument standard library calls.
#ifdef WIN32
  if (F.hasName() && F.getName().find("_") == 0) {
    return true;
  }
#endif

  if (F.hasName() && F.getName().startswith("__csi"))
    return true;

  // Never instrument the CSI ctor.
  if (F.hasName() && F.getName() == CsiRtUnitCtorName)
    return true;

  // Don't instrument functions that will run before or
  // simultaneously with CSI ctors.
  GlobalVariable *GV = M.getGlobalVariable("llvm.global_ctors");
  if (GV == nullptr)
    return false;
  ConstantArray *CA = cast<ConstantArray>(GV->getInitializer());
  for (Use &OP : CA->operands()) {
    if (isa<ConstantAggregateZero>(OP))
      continue;
    ConstantStruct *CS = cast<ConstantStruct>(OP);

    if (Function *CF = dyn_cast<Function>(CS->getOperand(1))) {
      uint64_t Priority =
          dyn_cast<ConstantInt>(CS->getOperand(0))->getLimitedValue();
      if (Priority <= CsiUnitCtorPriority && CF->getName() == F.getName()) {
        // Do not instrument F.
        return true;
      }
    }
  }
  // false means do instrument it.
  return false;
}

bool CSIImpl::isVtableAccess(Instruction *I) {
  if (MDNode *Tag = I->getMetadata(LLVMContext::MD_tbaa))
    return Tag->isTBAAVtableAccess();
  return false;
}

bool CSIImpl::addrPointsToConstantData(Value *Addr) {
  // If this is a GEP, just analyze its pointer operand.
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr))
    Addr = GEP->getPointerOperand();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->isConstant()) {
      return true;
    }
  } else if (LoadInst *L = dyn_cast<LoadInst>(Addr)) {
    if (isVtableAccess(L)) {
      return true;
    }
  }
  return false;
}

bool CSIImpl::isAtomic(Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->isAtomic() && LI->getSyncScopeID() != SyncScope::SingleThread;
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isAtomic() && SI->getSyncScopeID() != SyncScope::SingleThread;
  if (isa<AtomicRMWInst>(I))
    return true;
  if (isa<AtomicCmpXchgInst>(I))
    return true;
  if (isa<FenceInst>(I))
    return true;
  return false;
}

void CSIImpl::computeLoadAndStoreProperties(
    SmallVectorImpl<std::pair<Instruction *, CsiLoadStoreProperty>>
        &LoadAndStoreProperties,
    SmallVectorImpl<Instruction *> &BBLoadsAndStores, const DataLayout &DL,
    LoopInfo &LI) {
  SmallSet<Value *, 8> WriteTargets;

  for (SmallVectorImpl<Instruction *>::reverse_iterator
           It = BBLoadsAndStores.rbegin(),
           E = BBLoadsAndStores.rend();
       It != E; ++It) {
    Instruction *I = *It;
    unsigned Alignment;
    if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
      Value *Addr = Store->getPointerOperand();
      WriteTargets.insert(Addr);
      CsiLoadStoreProperty Prop;
      // Update alignment property data
      Alignment = Store->getAlignment();
      Prop.setAlignment(Alignment);
      // Set vtable-access property
      Prop.setIsVtableAccess(isVtableAccess(Store));
      // Set constant-data-access property
      Prop.setIsConstant(addrPointsToConstantData(Addr));
      Value *Obj = GetUnderlyingObject(Addr, DL);
      // Set is-on-stack property
      Prop.setIsOnStack(isa<AllocaInst>(Obj));
      // Set may-be-captured property
      Prop.setMayBeCaptured(isa<GlobalValue>(Obj) ||
                            PointerMayBeCaptured(Addr, true, true));
      // Set is-volatile property
      Prop.setIsVolatile(Store->isVolatile());
      LoadAndStoreProperties.push_back(std::make_pair(I, Prop));
    } else {
      LoadInst *Load = cast<LoadInst>(I);
      Value *Addr = Load->getPointerOperand();
      CsiLoadStoreProperty Prop;
      // Update alignment property data
      Alignment = Load->getAlignment();
      Prop.setAlignment(Alignment);
      // Set vtable-access property
      Prop.setIsVtableAccess(isVtableAccess(Load));
      // Set constant-data-access-property
      Prop.setIsConstant(addrPointsToConstantData(Addr));
      Value *Obj = GetUnderlyingObject(Addr, DL);
      // Set is-on-stack property
      Prop.setIsOnStack(isa<AllocaInst>(Obj));
      // Set may-be-captured property
      Prop.setMayBeCaptured(isa<GlobalValue>(Obj) ||
                            PointerMayBeCaptured(Addr, true, true));
      // Set is-volatile property
      Prop.setIsVolatile(Load->isVolatile());
      // Set load-read-before-write-in-bb property
      bool HasBeenSeen = WriteTargets.count(Addr) > 0;
      Prop.setLoadReadBeforeWriteInBB(HasBeenSeen);
      Prop.setHasOneUse(checkHasOneUse(I, LI));
      Prop.setBBLocal(checkBBLocal(I, *I->getParent()));
      LoadAndStoreProperties.push_back(std::make_pair(I, Prop));
    }
  }
  BBLoadsAndStores.clear();
}

// Update the attributes on the instrumented function that might be invalidated
// by the inserted instrumentation.
void CSIImpl::updateInstrumentedFnAttrs(Function &F) {
  AttrBuilder B;
  B.addAttribute(Attribute::ReadOnly)
      .addAttribute(Attribute::ReadNone)
      .addAttribute(Attribute::ArgMemOnly)
      .addAttribute(Attribute::InaccessibleMemOnly)
      .addAttribute(Attribute::InaccessibleMemOrArgMemOnly);
  F.removeAttributes(AttributeList::FunctionIndex, B);
}

void CSIImpl::instrumentFunction(Function &F) {
  // This is required to prevent instrumenting the call to
  // __csi_module_init from within the module constructor.

  if (F.empty() || shouldNotInstrumentFunction(F))
    return;

  setupCalls(F);

  setupBlocks(F, TLI);

  SmallVector<std::pair<Instruction *, CsiLoadStoreProperty>, 8>
      LoadAndStoreProperties;
  SmallVector<Instruction *, 8> AllocationFnCalls;
  SmallVector<Instruction *, 8> FreeCalls;
  SmallVector<Instruction *, 8> MemIntrinsics;
  SmallVector<Instruction *, 8> Callsites;
  SmallVector<BasicBlock *, 8> BasicBlocks;
  SmallVector<Instruction *, 8> AtomicAccesses;
  SmallVector<DetachInst *, 8> Detaches;
  SmallVector<SyncInst *, 8> Syncs;
  SmallVector<Instruction *, 8> Allocas;
  SmallVector<Instruction *, 8> AllCalls;
  SmallVector<Instruction *, 32> Arithmetic;
  SmallVector<Instruction *, 8> VectorMemBuiltins;
  bool MaySpawn = false;

  DominatorTree *DT = &GetDomTree(F);
  LoopInfo &LI = GetLoopInfo(F);
  TaskInfo &TI = GetTaskInfo(F);
  ScalarEvolution *SE = nullptr;
  if (GetScalarEvolution)
    SE = &(*GetScalarEvolution)(F);
  for (Loop *L : LI)
    simplifyLoop(L, DT, &LI, SE, nullptr, false /* PreserveLCSSA */);

  for (Argument &Arg : F.args())
    // Add an ID for this function argument.
    ParameterFED.add(Arg);

  // Compile lists of all instrumentation points before anything is modified.
  for (BasicBlock &BB : F) {
    SmallVector<Instruction *, 8> BBLoadsAndStores;
    for (Instruction &I : BB) {
      if (isAtomic(&I))
        AtomicAccesses.push_back(&I);
      else if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
        BBLoadsAndStores.push_back(&I);
      } else if (DetachInst *DI = dyn_cast<DetachInst>(&I)) {
        MaySpawn = true;
        Detaches.push_back(DI);
      } else if (SyncInst *SI = dyn_cast<SyncInst>(&I)) {
        Syncs.push_back(SI);
      } else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {

        // Record this function call as either an allocation function, a call to
        // free (or delete), a memory intrinsic, or an ordinary real function
        // call.
        if (isAllocationFn(&I, TLI))
          AllocationFnCalls.push_back(&I);
        else if (isFreeCall(&I, TLI))
          FreeCalls.push_back(&I);
        else if (isa<MemIntrinsic>(I))
          MemIntrinsics.push_back(&I);
        else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I)) {
          if (!callsPlaceholderFunction(I)) {
            switch (II->getIntrinsicID()) {
            case Intrinsic::masked_load:
            case Intrinsic::masked_store:
            case Intrinsic::masked_gather:
            case Intrinsic::masked_scatter:
              // Handle instrinsics for masked vector loads, stores, gathers, and
              // scatters specially.
              VectorMemBuiltins.push_back(II);
              break;
            default:
              Callsites.push_back(II);
              break;
            }
          }
        } else
          Callsites.push_back(&I);

        // All calls are candidates for interpositioning except for calls to
        // placeholder functions.
        if (!callsPlaceholderFunction(I))
          AllCalls.push_back(&I);

        computeLoadAndStoreProperties(LoadAndStoreProperties, BBLoadsAndStores,
                                      DL, LI);
      } else if (isa<AllocaInst>(I)) {
        Allocas.push_back(&I);
      } else if (IsInstrumentedArithmetic(&I)) {
        Arithmetic.push_back(&I);
      }
    }
    computeLoadAndStoreProperties(LoadAndStoreProperties, BBLoadsAndStores, DL,
                                  LI);
    BasicBlocks.push_back(&BB);
  }

  csi_id_t LocalId = getLocalFunctionID(F);
  // First assign local ID's to all ops in the function that are involved in
  // data flow, so that these IDs can be supplied as arguments to other hooks.
  for (std::pair<Instruction *, CsiLoadStoreProperty> p :
         LoadAndStoreProperties)
    assignLoadOrStoreID(p.first);
  for (Instruction *I : VectorMemBuiltins)
    assignLoadOrStoreID(I);
  for (Instruction *I : AtomicAccesses)
    assignAtomicID(I);
  for (Instruction *I : Callsites)
    assignCallsiteID(I);
  for (Instruction *I : MemIntrinsics)
    assignCallsiteID(I);
  for (Instruction *I : Allocas)
    assignAllocaID(I);
  for (Instruction *I : AllocationFnCalls)
    assignAllocFnID(I);
  for (Instruction *I : Arithmetic)
    assignArithmeticID(I);
  for (BasicBlock *BB : BasicBlocks)
    assignBasicBlockID(*BB);

  // Determine inputs for CFG structures.
  InputMap<BasicBlock> BBInputs;
  InputMap<Loop> LoopInputs;
  InputMap<Task> TaskInputs;

  if (Options.InstrumentBasicBlocks)
    for (BasicBlock *BB : BasicBlocks)
      getBBInputs(*BB, BBInputs[BB]);
  if (Options.InstrumentLoops)
    for (Loop *L : LI)
      getAllLoopInputs(*L, LI, LoopInputs);
  if (Options.InstrumentTapir)
    getAllTaskInputs(TI, TaskInputs);

  // Instrument basic blocks.  Note that we do this before other instrumentation
  // so that we put this at the beginning of the basic block, and then the
  // function entry call goes before the call to basic block entry.
  if (Options.InstrumentBasicBlocks)
    for (BasicBlock *BB : BasicBlocks)
      instrumentBasicBlock(*BB, BBInputs[BB]);

  // Instrument Tapir constructs.
  if (Options.InstrumentTapir) {
    // Allocate a local variable that will keep track of whether
    // a spawn has occurred before a sync. It will be set to 1 after
    // a spawn and reset to 0 after a sync.
    auto TrackVars = keepTrackOfSpawns(F, Detaches, Syncs);

    if (Config->DoesFunctionRequireInstrumentationForPoint(
            F.getName(), InstrumentationPoint::INSTR_TAPIR_DETACH)) {
      for (DetachInst *DI : Detaches)
        instrumentDetach(DI, DT, TI, TrackVars, TaskInputs);
    }
    if (Config->DoesFunctionRequireInstrumentationForPoint(
            F.getName(), InstrumentationPoint::INSTR_TAPIR_SYNC)) {
      for (SyncInst *SI : Syncs)
        instrumentSync(SI, TrackVars);
    }
  }

  if (Options.InstrumentLoops)
    // Recursively instrument all loops
    for (Loop *L : LI)
      instrumentLoop(*L, LoopInputs, TI, SE);

  // Do this work in a separate loop after copying the iterators so that we
  // aren't modifying the list as we're iterating.
  if (Options.InstrumentMemoryAccesses) {
    for (std::pair<Instruction *, CsiLoadStoreProperty> p :
         LoadAndStoreProperties)
      instrumentLoadOrStore(p.first, p.second, DL);
    for (Instruction *I : VectorMemBuiltins)
      instrumentVectorMemBuiltin(I);
  }

  // Instrument atomic memory accesses in any case (they can be used to
  // implement synchronization).
  if (Options.InstrumentAtomics)
    for (Instruction *I : AtomicAccesses)
      instrumentAtomic(I, DL);

  if (Options.InstrumentMemIntrinsics)
    for (Instruction *I : MemIntrinsics)
      instrumentMemIntrinsic(I);

  if (Options.InstrumentCalls)
    for (Instruction *I : Callsites)
      instrumentCallsite(I, DT, LI);

  if (Options.InstrumentAllocas)
    for (Instruction *I : Allocas)
      instrumentAlloca(I);

  if (Options.InstrumentAllocFns) {
    for (Instruction *I : AllocationFnCalls)
      instrumentAllocFn(I, DT);
    for (Instruction *I : FreeCalls)
      instrumentFree(I);
  }

  if (Options.InstrumentArithmetic != CSIOptions::ArithmeticType::None)
    for (Instruction *I : Arithmetic)
      instrumentArithmetic(I, LI);

  if (Options.Interpose) {
    for (Instruction *I : AllCalls)
      interposeCall(I);
  }

  // Instrument function entry/exit points.
  if (Options.InstrumentFuncEntryExit) {
    IRBuilder<> IRB(&*F.getEntryBlock().getFirstInsertionPt());
    Value *FuncId = FunctionFED.localToGlobalId(LocalId, IRB);
    if (Config->DoesFunctionRequireInstrumentationForPoint(
            F.getName(), InstrumentationPoint::INSTR_FUNCTION_ENTRY)) {
      // Instrument the parameters of this function.
      instrumentParams(IRB, F, FuncId);

      // TODO? Remove this other instrumentation about function parameters.
      // Get the ID of the first function argument
      Value *FirstArgID = ParameterFED.localToGlobalId(
          ParameterFED.lookupId(&*F.arg_begin()), IRB);
      // Get the number of function arguments
      Value *NumArgs = IRB.getInt32(F.arg_size());

      // Compute the properties of this function entry.
      CsiFuncProperty FuncEntryProp;
      FuncEntryProp.setMaySpawn(MaySpawn);
      Value *PropVal = FuncEntryProp.getValue(IRB);

      // Insert the function-entry hook.
      insertHookCall(&*IRB.GetInsertPoint(), CsiFuncEntry,
                     {FuncId, FirstArgID, NumArgs, PropVal});
    }
    if (Config->DoesFunctionRequireInstrumentationForPoint(
            F.getName(), InstrumentationPoint::INSTR_FUNCTION_EXIT)) {
      // Iterate over the exits from this function.  Any implicit exceptional
      // exits should have been made explicit already by setupCalls(), so we
      // don't need to do that here.
      EscapeEnumerator EE(F, "csi.cleanup", false);
      while (IRBuilder<> *AtExit = EE.Next()) {
        Instruction *ExitInst = cast<Instruction>(AtExit->GetInsertPoint());
        csi_id_t ExitLocalId = FunctionExitFED.add(*ExitInst);
        Value *ExitCsiId =
            FunctionExitFED.localToGlobalId(ExitLocalId, *AtExit);

        if (Options.InstrumentInputs && ExitInst->getNumOperands() > 0) {
          Value *ReturnOp = ExitInst->getOperand(0);
          // Get the input hook we need, based on the input type.
          Type *RetOpTy = ReturnOp->getType();
          if (!SupportedType(RetOpTy)) {
            // Skip recording inputs for unsupported types
            DEBUG(dbgs() << "[CSI] Skipping unsupported type " << *RetOpTy <<
                  "\n");
            continue;
          }
          Function *InputHook =
            getCSIInputHook(M, CSIDataFlowObject::FunctionExit, RetOpTy);
          std::pair<Value *, Value *> OperandID = getOperandID(ReturnOp,
                                                               *AtExit);
          // Cast the operand as needed.
          Type *OperandCastTy = getOperandCastTy(M, RetOpTy);
          Value *CastRetOp = ReturnOp;
          if (OperandCastTy != RetOpTy)
            CastRetOp = AtExit->CreateZExtOrBitCast(ReturnOp, OperandCastTy);
          // TODO: Compute flags.  Not sure what flags to compute.
          CsiArithmeticFlags Flags;
          Value *FlagsVal = Flags.getValue(IRB);
          // Insert the hook call.
          CallInst *Call =
            AtExit->CreateCall(InputHook, {ExitCsiId, OperandID.first,
                                           OperandID.second, CastRetOp,
                                           FlagsVal});
          setInstrumentationDebugLoc(ExitInst, (Instruction *)Call);
        }

        // Compute the properties of this function exit.
        CsiFuncExitProperty FuncExitProp;
        FuncExitProp.setMaySpawn(MaySpawn);
        FuncExitProp.setEHReturn(!isa<ReturnInst>(ExitInst));
        Value *PropVal = FuncExitProp.getValue(*AtExit);

        // Insert the hook call.
        insertHookCall(&*AtExit->GetInsertPoint(), CsiFuncExit,
                       {ExitCsiId, FuncId, PropVal});
      }
    }
  }

  updateInstrumentedFnAttrs(F);
}

DenseMap<Value *, Value *>
llvm::CSIImpl::keepTrackOfSpawns(Function &F,
                                 const SmallVectorImpl<DetachInst *> &Detaches,
                                 const SmallVectorImpl<SyncInst *> &Syncs) {
  DenseMap<Value *, Value *> TrackVars;
  // SyncRegions are created using an LLVM instrinsic within the task where the
  // region is used.  Store each sync region as an LLVM instruction.
  SmallPtrSet<Instruction *, 8> Regions;
  for (auto &Detach : Detaches) {
    Regions.insert(cast<Instruction>(Detach->getSyncRegion()));
  }
  for (auto &Sync : Syncs) {
    Regions.insert(cast<Instruction>(Sync->getSyncRegion()));
  }

  LLVMContext &C = F.getContext();

  size_t RegionIndex = 0;
  for (auto Region : Regions) {
    // Build a local variable for the sync region within the task where it is
    // used.
    IRBuilder<> Builder(Region);
    Value *TrackVar = Builder.CreateAlloca(IntegerType::getInt32Ty(C), nullptr,
                                           "has_spawned_region_" +
                                               std::to_string(RegionIndex));
    Builder.CreateStore(
        Constant::getIntegerValue(IntegerType::getInt32Ty(C), APInt(32, 0)),
        TrackVar);

    TrackVars.insert({Region, TrackVar});
    RegionIndex++;
  }

  return TrackVars;
}

Function *llvm::CSIImpl::getInterpositionFunction(Function *F) {
  if (InterpositionFunctions.find(F) != InterpositionFunctions.end()) {
    return InterpositionFunctions.lookup(F);
  }

  std::string InterposedName =
      (std::string) "__csi_interpose_" + F->getName().str();

  Function *InterpositionFunction =
      (Function *)M.getOrInsertFunction(InterposedName, F->getFunctionType());

  InterpositionFunctions.insert({F, InterpositionFunction});

  return InterpositionFunction;
}

void ComprehensiveStaticInstrumentationLegacyPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<CallGraphWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addRequired<TaskInfoWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
}

bool ComprehensiveStaticInstrumentationLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  CallGraph *CG = &getAnalysis<CallGraphWrapperPass>().getCallGraph();
  const TargetLibraryInfo *TLI =
      &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  auto GetDomTree = [this](Function &F) -> DominatorTree & {
    return this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
  };
  auto GetLoopInfo = [this](Function &F) -> LoopInfo & {
    return this->getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
  };
  auto GetSE = [this](Function &F) -> ScalarEvolution & {
    return this->getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
  };
  auto GetTaskInfo = [this](Function &F) -> TaskInfo & {
    return this->getAnalysis<TaskInfoWrapperPass>(F).getTaskInfo();
  };

  bool res = CSIImpl(M, CG, GetDomTree, GetLoopInfo, GetTaskInfo, TLI, GetSE,
                     Options).run();

  verifyModule(M, &llvm::errs());

  return res;
}

ComprehensiveStaticInstrumentationPass::ComprehensiveStaticInstrumentationPass(
    const CSIOptions &Options)
    : Options(OverrideFromCL(Options)) {}

PreservedAnalyses
ComprehensiveStaticInstrumentationPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  auto &CG = AM.getResult<CallGraphAnalysis>(M);
  auto GetDT = [&FAM](Function &F) -> DominatorTree & {
    return FAM.getResult<DominatorTreeAnalysis>(F);
  };
  auto GetLI = [&FAM](Function &F) -> LoopInfo & {
    return FAM.getResult<LoopAnalysis>(F);
  };
  auto GetSE = [&FAM](Function &F) -> ScalarEvolution & {
    return FAM.getResult<ScalarEvolutionAnalysis>(F);
  };
  auto GetTI = [&FAM](Function &F) -> TaskInfo & {
    return FAM.getResult<TaskAnalysis>(F);
  };
  auto *TLI = &AM.getResult<TargetLibraryAnalysis>(M);

  if (!CSIImpl(M, &CG, GetDT, GetLI, GetTI, TLI, GetSE, Options).run())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
