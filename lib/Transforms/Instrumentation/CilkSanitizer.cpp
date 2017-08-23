//===- CilkSanitizer.cpp - determinacy race detector for Cilk/Tapir -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of CilkSan, a determinacy race detector for Cilk
// programs.
//
// This instrumentation pass inserts calls to the runtime library before
// appropriate memory accesses.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/DetachSSA.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Transforms/CSI.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cilksan"

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumOmittedReadsBeforeWrite,
          "Number of reads ignored due to following writes");
STATISTIC(NumOmittedReadsFromConstants,
          "Number of reads from constant data");
STATISTIC(NumInstrumentedDetaches, "Number of instrumented detaches");
STATISTIC(NumInstrumentedDetachExits, "Number of instrumented detach exits");
STATISTIC(NumInstrumentedSyncs, "Number of instrumented syncs");

static const char *const CsanDetachBaseIdName = "__csan_unit_detach_base_id";
static const char *const CsanTaskBaseIdName = "__csan_unit_task_base_id";
static const char *const CsanTaskExitBaseIdName =
  "__csan_unit_task_exit_base_id";
static const char *const CsanDetachContinueBaseIdName =
  "__csan_unit_detach_continue_base_id";
static const char *const CsanSyncBaseIdName = "__csan_unit_sync_base_id";
static const char *const CsiUnitObjTableName = "__csi_unit_obj_table";
static const char *const CsiUnitObjTableArrayName = "__csi_unit_obj_tables";

/// Maintains a mapping from CSI ID of a load or store to the source information
/// of the object accessed by that load or store.
class ObjectTable : public ForensicTable {
public:
  ObjectTable() : ForensicTable() {}
  ObjectTable(Module &M, StringRef BaseIdName)
      : ForensicTable(M, BaseIdName) {}

  /// The number of entries in this table
  uint64_t size() const { return LocalIdToSourceLocationMap.size(); }

  /// Add the given instruction to this table.
  /// \returns The local ID of the Instruction.
  uint64_t add(Instruction &I, const DataLayout &DL);

  /// Get the Type for a pointer to a table entry.
  ///
  /// A table entry is just a source location.
  static PointerType *getPointerType(LLVMContext &C);

  /// Insert this table into the given Module.
  ///
  /// The table is constructed as a ConstantArray indexed by local IDs.  The
  /// runtime is responsible for performing the mapping that allows the table to
  /// be indexed by global ID.
  Constant *insertIntoModule(Module &M) const;

private:
  struct SourceLocation {
    StringRef Name;
    int32_t Line;
    StringRef Filename;
    StringRef Directory;
  };

  /// Map of local ID to SourceLocation.
  DenseMap<uint64_t, SourceLocation> LocalIdToSourceLocationMap;

  /// Create a struct type to match the "struct SourceLocation" type.
  /// (and the source_loc_t type in csi.h).
  static StructType *getSourceLocStructType(LLVMContext &C);

  /// Append the line and file information to the table.
  void add(uint64_t ID, int32_t Line = -1,
           StringRef Filename = "", StringRef Directory = "",
           StringRef Name = "");
};

namespace {

struct CilkSanitizerImpl : public CSIImpl {
  // CilkSanitizerImpl(Module &M, CallGraph *CG,
  //                   function_ref<DetachSSA &(Function &)> GetDSSA,
  //                   function_ref<MemorySSA &(Function &)> GetMSSA)
  //     : CSIImpl(M, CG), GetDSSA(GetDSSA), GetMSSA(GetMSSA) {
  CilkSanitizerImpl(Module &M, CallGraph *CG,
                    function_ref<DominatorTree &(Function &)> GetDomTree,
                    const TargetLibraryInfo *TLI)
      : CSIImpl(M, CG), GetDomTree(GetDomTree), TLI(TLI),
        CsanFuncEntry(nullptr), CsanFuncExit(nullptr), CsanRead(nullptr),
        CsanWrite(nullptr), CsanDetach(nullptr), CsanDetachContinue(nullptr),
        CsanTaskEntry(nullptr), CsanTaskExit(nullptr), CsanSync(nullptr) {
    // Even though we're doing our own instrumentation, we want the CSI setup
    // for the instrumentation of function entry/exit, memory accesses (i.e.,
    // loads and stores), atomics, memory intrinsics.  We also want call sites,
    // for extracting debug information.
    Options.InstrumentBasicBlocks = false;
    // Options.InstrumentCalls = false;
    Options.InstrumentMemoryAccesses = false;
  }
  bool run();

  static StructType *getUnitObjTableType(LLVMContext &C,
                                         PointerType *EntryPointerType);
  static Constant *objTableToUnitObjTable(Module &M,
                                          StructType *UnitObjTableType,
                                          ObjectTable &ObjTable);

  // Methods for handling FED tables
  void initializeCsanFEDTables();
  void collectUnitFEDTables();

  // Methods for handling object tables
  void initializeCsanObjectTables();
  void collectUnitObjectTables();

  CallInst *createRTUnitInitCall(IRBuilder<> &IRB) override;

  // Initialize custom hooks for CilkSanitizer
  void initializeCsanHooks();

  // Insert hooks at relevant program points
  bool instrumentLoadOrStore(Instruction *I, const DataLayout &DL);
  bool instrumentCallsite(Instruction *I, DominatorTree *DT);
  bool instrumentDetach(DetachInst *DI, DominatorTree *DT);
  bool instrumentSync(SyncInst *SI);
  bool instrumentFunction(Function &F);
  void chooseInstructionsToInstrument(
      SmallVectorImpl<Instruction *> &Local,
      SmallVectorImpl<Instruction *> &All,
      const DataLayout &DL);

private:
  // Analysis results
  // function_ref<DetachSSA &(Function &)> GetDSSA;
  // function_ref<MemorySSA &(Function &)> GetMSSA;
  function_ref<DominatorTree &(Function &)> GetDomTree;
  const TargetLibraryInfo *TLI;

  // Instrumentation hooks
  Function *CsanFuncEntry, *CsanFuncExit; 
  Function *CsanRead, *CsanWrite;
  // Function *CsanAtomicXchg, *CsanAtomicAdd, *CsanAtomicSub, *CsanAtomicOr,
  //   *CsanAtomicXor, *CsanAtomicNand, *CsanAtomicCAS;
  Function *CsanDetach, *CsanDetachContinue;
  Function *CsanTaskEntry, *CsanTaskExit;
  Function *CsanSync;

  // CilkSanitizer FED tables
  FrontEndDataTable DetachFED, TaskFED, TaskExitFED, DetachContinueFED,
    SyncFED; 

  // CilkSanitizer custom forensic tables
  ObjectTable LoadObj, StoreObj;

  SmallVector<Constant *, 2> UnitObjTables;

};

/// CilkSanitizer: instrument the code in module to find races.
struct CilkSanitizer : public ModulePass {
  static char ID;  // Pass identification, replacement for typeid.
  CilkSanitizer() : ModulePass(ID) {}
  StringRef getPassName() const override {
    return "CilkSanitizer";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M);
};
} // namespace

char CilkSanitizer::ID = 0;
INITIALIZE_PASS_BEGIN(
    CilkSanitizer, "csan",
    "CilkSanitizer: detects determinacy races in Cilk programs.",
    false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
// INITIALIZE_PASS_DEPENDENCY(DetachSSAWrapperPass)
INITIALIZE_PASS_END(
    CilkSanitizer, "csan",
    "CilkSanitizer: detects determinacy races in Cilk programs.",
    false, false)

void CilkSanitizer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<CallGraphWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  // AU.addRequired<DetachSSAWrapperPass>();
  // AU.addRequired<MemorySSAWrapperPass>();
}

ModulePass *llvm::createCilkSanitizerPass() {
  return new CilkSanitizer();
}

uint64_t ObjectTable::add(Instruction &I,
                          const DataLayout &DL) {
  assert((isa<StoreInst>(I) || isa<LoadInst>(I)) &&
         "Invalid instruction to add to the object table.");
  uint64_t ID = getId(&I);
  Value *Addr = isa<StoreInst>(I)
    ? cast<StoreInst>(I).getPointerOperand()
    : cast<LoadInst>(I).getPointerOperand();
  Value *Obj = GetUnderlyingObject(Addr, DL);

  // First, if the underlying object is a global variable, get that variable's
  // debug information.
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Obj)) {
    SmallVector<DIGlobalVariableExpression *, 1> DbgGVExprs;
    GV->getDebugInfo(DbgGVExprs);
    for (auto *GVE : DbgGVExprs) {
      auto *DGV = GVE->getVariable();
      if (DGV->getName() != "") {
        DEBUG({
            if (isa<StoreInst>(I))
              dbgs() << "StoreObj[" << ID << "] = global variable "
                     << DGV->getName() << "\n";
            else
              dbgs() << "LoadObj[" << ID << "] = global variable "
                     << DGV->getName() << "\n";});
        add(ID, DGV->getLine(), DGV->getFilename(), DGV->getDirectory(),
            DGV->getName());
        return ID;
      }
    }
    add(ID);
    return ID;
  }

  // Next, if this is an alloca instruction, look for a llvm.dbg.declare
  // intrinsic.
  if (isa<AllocaInst>(Obj)) {
    if (auto *DDI = FindAllocaDbgDeclare(Obj)) {
      auto *LV = DDI->getVariable();
      if (LV->getName() != "") {
        DEBUG({
            if (isa<StoreInst>(I))
              dbgs() << "StoreObj[" << ID << "] = local variable "
                     << LV->getName() << "\n";
            else
              dbgs() << "LoadObj[" << ID << "] = local variable "
                     << LV->getName() << "\n";});
        add(ID, LV->getLine(), LV->getFilename(), LV->getDirectory(),
            LV->getName());
        return ID;
      }
    }
  }

  // Otherwise just examine the llvm.dbg.value intrinsics for this object.
  SmallVector<DbgValueInst *, 1> DbgValues;
  findDbgValues(DbgValues, Obj);
  for (auto *DVI : DbgValues) {
    auto *LV = DVI->getVariable();
    if (LV->getName() != "") {
      DEBUG({
          if (isa<StoreInst>(I))
            dbgs() << "StoreObj[" << ID << "] = local variable "
                   << LV->getName() << "\n";
          else
            dbgs() << "LoadObj[" << ID << "] = local variable "
                   << LV->getName() << "\n";});
      add(ID, LV->getLine(), LV->getFilename(), LV->getDirectory(),
          LV->getName());
      return ID;
    }
  }

  add(ID);
  return ID;
}

PointerType *ObjectTable::getPointerType(LLVMContext &C) {
  return PointerType::get(getSourceLocStructType(C), 0);
}

StructType *ObjectTable::getSourceLocStructType(LLVMContext &C) {
  return StructType::get(
      /* Name */ PointerType::get(IntegerType::get(C, 8), 0),
      /* Line */ IntegerType::get(C, 32),
      /* File */ PointerType::get(IntegerType::get(C, 8), 0));
}

void ObjectTable::add(uint64_t ID, int32_t Line,
                      StringRef Filename, StringRef Directory,
                      StringRef Name) {
  assert(LocalIdToSourceLocationMap.find(ID) ==
             LocalIdToSourceLocationMap.end() &&
         "Id already exists in FED table.");
  LocalIdToSourceLocationMap[ID] = {Name, Line, Filename, Directory};
}

Constant *ObjectTable::insertIntoModule(Module &M) const {
  LLVMContext &C = M.getContext();
  StructType *TableType = getSourceLocStructType(C);
  IntegerType *Int32Ty = IntegerType::get(C, 32);
  Constant *Zero = ConstantInt::get(Int32Ty, 0);
  Value *GepArgs[] = {Zero, Zero};
  SmallVector<Constant *, 6> TableEntries;

  for (uint64_t LocalID = 0; LocalID < IdCounter; ++LocalID) {
    const SourceLocation &E = LocalIdToSourceLocationMap.find(LocalID)->second;
    Constant *Line = ConstantInt::get(Int32Ty, E.Line);
    Constant *File;
    {
      std::string Filename = E.Filename.str();
      if (!E.Directory.empty())
        Filename = E.Directory.str() + "/" + Filename;
      Constant *FileStrConstant = ConstantDataArray::getString(C, Filename);
      GlobalVariable *GV = M.getGlobalVariable("__csi_unit_filename", true);
      if (GV == NULL) {
        GV = new GlobalVariable(M, FileStrConstant->getType(),
                                true, GlobalValue::PrivateLinkage,
                                FileStrConstant, "__csi_unit_filename",
                                nullptr,
                                GlobalVariable::NotThreadLocal, 0);
        GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      }
      assert(GV);
      File =
        ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
    }
    Constant *Name;
    if (E.Name.empty())
      Name = ConstantPointerNull::get(PointerType::get(
                                          IntegerType::get(C, 8), 0));
    else {
      Constant *NameStrConstant = ConstantDataArray::getString(C, E.Name);
      GlobalVariable *GV =
        M.getGlobalVariable(("__csi_unit_object_name_" + E.Name).str(), true);
      if (GV == NULL) {
        GV = new GlobalVariable(M, NameStrConstant->getType(),
                                true, GlobalValue::PrivateLinkage,
                                NameStrConstant,
                                "__csi_unit_object_name_" + E.Name,
                                nullptr,
                                GlobalVariable::NotThreadLocal, 0);
        GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      }
      assert(GV);
      Name =
        ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
    }
    // The order of arguments to ConstantStruct::get() must match the
    // source_loc_t type in csi.h.
    TableEntries.push_back(ConstantStruct::get(TableType, Name, Line, File));
  }

  ArrayType *TableArrayType = ArrayType::get(TableType, TableEntries.size());
  Constant *Table = ConstantArray::get(TableArrayType, TableEntries);
  GlobalVariable *GV =
    new GlobalVariable(M, TableArrayType, false, GlobalValue::InternalLinkage,
                       Table, CsiUnitObjTableName);
  return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
}

bool CilkSanitizerImpl::run() {
  initializeCsi();
  initializeCsanFEDTables();
  initializeCsanObjectTables();
  initializeCsanHooks();

  for (Function &F : M) {
    DEBUG(dbgs() << "Instrumenting " << F.getName() << "\n");
    instrumentFunction(F);
  }

  collectUnitFEDTables();
  collectUnitObjectTables();
  finalizeCsi();
  return true;
}

void CilkSanitizerImpl::initializeCsanFEDTables() {
  DetachFED = FrontEndDataTable(M, CsanDetachBaseIdName);
  TaskFED = FrontEndDataTable(M, CsanTaskBaseIdName);
  TaskExitFED = FrontEndDataTable(M, CsanTaskExitBaseIdName);
  DetachContinueFED = FrontEndDataTable(M, CsanDetachContinueBaseIdName);
  SyncFED = FrontEndDataTable(M, CsanSyncBaseIdName);
}

void CilkSanitizerImpl::initializeCsanObjectTables() {
  LoadObj = ObjectTable(M, CsiLoadBaseIdName);
  StoreObj = ObjectTable(M, CsiStoreBaseIdName);
}

void CilkSanitizerImpl::collectUnitFEDTables() {
  CSIImpl::collectUnitFEDTables();
  LLVMContext &C = M.getContext();
  StructType *UnitFedTableType =
      getUnitFedTableType(C, FrontEndDataTable::getPointerType(C));

  // The order of the FED tables here must match the enum in csanrt.c and the
  // csan_instrumentation_counts_t in csan.h.
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, DetachFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, TaskFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, TaskExitFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, DetachContinueFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, SyncFED));
}

// Create a struct type to match the unit_obj_entry_t type in csanrt.c.
StructType *CilkSanitizerImpl::getUnitObjTableType(LLVMContext &C,
                                                   PointerType *EntryPointerType) {
  return StructType::get(IntegerType::get(C, 64),
                         EntryPointerType);
}

Constant *CilkSanitizerImpl::objTableToUnitObjTable(
    Module &M, StructType *UnitObjTableType, ObjectTable &ObjTable) {
  Constant *NumEntries =
    ConstantInt::get(IntegerType::get(M.getContext(), 64), ObjTable.size());
  // Constant *BaseIdPtr =
  //   ConstantExpr::getPointerCast(FedTable.baseId(),
  //                                Type::getInt8PtrTy(M.getContext(), 0));
  Constant *InsertedTable = ObjTable.insertIntoModule(M);
  return ConstantStruct::get(UnitObjTableType, NumEntries,
                             InsertedTable);
}

void CilkSanitizerImpl::collectUnitObjectTables() {
  LLVMContext &C = M.getContext();
  StructType *UnitObjTableType =
      getUnitObjTableType(C, ObjectTable::getPointerType(C));

  UnitObjTables.push_back(
      objTableToUnitObjTable(M, UnitObjTableType, LoadObj));
  UnitObjTables.push_back(
      objTableToUnitObjTable(M, UnitObjTableType, StoreObj));
}

CallInst *CilkSanitizerImpl::createRTUnitInitCall(IRBuilder<> &IRB) {
  LLVMContext &C = M.getContext();

  StructType *UnitFedTableType =
      getUnitFedTableType(C, FrontEndDataTable::getPointerType(C));
  StructType *UnitObjTableType =
      getUnitObjTableType(C, ObjectTable::getPointerType(C));

  // Lookup __csirt_unit_init
  SmallVector<Type *, 4> InitArgTypes({IRB.getInt8PtrTy(),
                                       PointerType::get(UnitFedTableType, 0),
                                       PointerType::get(UnitObjTableType, 0),
                                       InitCallsiteToFunction->getType()});
  FunctionType *InitFunctionTy =
      FunctionType::get(IRB.getVoidTy(), InitArgTypes, false);
  RTUnitInit = checkCsiInterfaceFunction(
      M.getOrInsertFunction(CsiRtUnitInitName, InitFunctionTy));
  assert(RTUnitInit);

  ArrayType *UnitFedTableArrayType =
      ArrayType::get(UnitFedTableType, UnitFedTables.size());
  Constant *FEDTable = ConstantArray::get(UnitFedTableArrayType, UnitFedTables);
  GlobalVariable *FEDGV = new GlobalVariable(M, UnitFedTableArrayType, false,
                                             GlobalValue::InternalLinkage, FEDTable,
                                             CsiUnitFedTableArrayName);

  ArrayType *UnitObjTableArrayType =
      ArrayType::get(UnitObjTableType, UnitObjTables.size());
  Constant *ObjTable = ConstantArray::get(UnitObjTableArrayType, UnitObjTables);
  GlobalVariable *ObjGV = new GlobalVariable(M, UnitObjTableArrayType, false,
                                             GlobalValue::InternalLinkage, ObjTable,
                                             CsiUnitObjTableArrayName);

  Constant *Zero = ConstantInt::get(IRB.getInt32Ty(), 0);
  Value *GepArgs[] = {Zero, Zero};

  // Insert call to __csirt_unit_init
  return IRB.CreateCall(
      RTUnitInit,
      {IRB.CreateGlobalStringPtr(M.getName()),
          ConstantExpr::getGetElementPtr(FEDGV->getValueType(), FEDGV, GepArgs),
          ConstantExpr::getGetElementPtr(ObjGV->getValueType(), ObjGV, GepArgs),
          InitCallsiteToFunction});
}

void CilkSanitizerImpl::initializeCsanHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *FuncPropertyTy = CsiFuncProperty::getType(C);
  Type *FuncExitPropertyTy = CsiFuncExitProperty::getType(C);
  Type *LoadPropertyTy = CsiLoadStoreProperty::getType(C);
  Type *StorePropertyTy = CsiLoadStoreProperty::getType(C);
  Type *RetType = IRB.getVoidTy();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *NumBytesType = IRB.getInt32Ty();
  Type *IDType = IRB.getInt64Ty();

  CsanFuncEntry = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csan_func_entry", RetType,
                            /* func_id */ IDType,
                            /* stack_ptr */ AddrType,
                            FuncPropertyTy));
  CsanFuncExit = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csan_func_exit", RetType,
                            /* func_exit_id */ IDType,
                            /* func_id */ IDType,
                            FuncExitPropertyTy));

  CsanRead = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csan_load", RetType, IDType,
                            AddrType, NumBytesType, LoadPropertyTy));
  CsanWrite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csan_store", RetType, IDType,
                            AddrType, NumBytesType, StorePropertyTy));
  // CsanWrite = checkCsiInterfaceFunction(
  //     M.getOrInsertFunction("__csan_atomic_exchange", RetType, IDType,
  //                           AddrType, NumBytesType, StorePropertyTy));

  CsanDetach = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csan_detach", RetType,
                            /* detach_id */ IDType));
  CsanTaskEntry = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csan_task", RetType,
                            /* task_id */ IDType,
                            /* detach_id */ IDType,
                            /* stack_ptr */ AddrType));
  CsanTaskExit = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csan_task_exit", RetType,
                            /* task_exit_id */ IDType,
                            /* task_id */ IDType,
                            /* detach_id */ IDType));
  CsanDetachContinue = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csan_detach_continue", RetType,
                            /* detach_continue_id */ IDType,
                            /* detach_id */ IDType));
  CsanSync = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csan_sync", RetType, IDType));
}

// Do not instrument known races/"benign races" that come from compiler
// instrumentatin. The user has no way of suppressing them.
static bool shouldInstrumentReadWriteFromAddress(const Module *M, Value *Addr) {
  // Peel off GEPs and BitCasts.
  Addr = Addr->stripInBoundsOffsets();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->hasSection()) {
      StringRef SectionName = GV->getSection();
      // Check if the global is in the PGO counters section.
      auto OF = Triple(M->getTargetTriple()).getObjectFormat();
      if (SectionName.endswith(
              getInstrProfSectionName(IPSK_cnts, OF, /*AddSegmentInfo=*/false)))
        return false;
    }

    // Check if the global is private gcov data.
    if (GV->getName().startswith("__llvm_gcov") ||
        GV->getName().startswith("__llvm_gcda"))
      return false;
  }

  // Do not instrument acesses from different address spaces; we cannot deal
  // with them.
  if (Addr) {
    Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
    if (PtrTy->getPointerAddressSpace() != 0)
      return false;
  }

  return true;
}

void CilkSanitizerImpl::chooseInstructionsToInstrument(
    SmallVectorImpl<Instruction *> &Local, SmallVectorImpl<Instruction *> &All,
    const DataLayout &DL) {
  SmallSet<Value*, 8> WriteTargets;
  // Iterate from the end.
  for (Instruction *I : reverse(Local)) {
    if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
      Value *Addr = Store->getPointerOperand();
      if (!shouldInstrumentReadWriteFromAddress(I->getModule(), Addr))
        continue;
      WriteTargets.insert(Addr);
    } else {
      LoadInst *Load = cast<LoadInst>(I);
      Value *Addr = Load->getPointerOperand();
      if (!shouldInstrumentReadWriteFromAddress(I->getModule(), Addr))
        continue;
      if (WriteTargets.count(Addr)) {
        // We will write to this temp, so no reason to analyze the read.
        NumOmittedReadsBeforeWrite++;
        continue;
      }
      if (addrPointsToConstantData(Addr)) {
        // Addr points to some constant data -- it can not race with any writes.
        NumOmittedReadsFromConstants++;
        continue;
      }
    }
    // Value *Addr = isa<StoreInst>(*I)
    //     ? cast<StoreInst>(I)->getPointerOperand()
    //     : cast<LoadInst>(I)->getPointerOperand();
    // if (isa<AllocaInst>(GetUnderlyingObject(Addr, DL)) &&
    //     !PointerMayBeCaptured(Addr, true, true)) {
    //   // The variable is addressable but not captured, so it cannot be
    //   // referenced from a different thread and participate in a data race
    //   // (see llvm/Analysis/CaptureTracking.h for details).
    //   NumOmittedNonCaptured++;
    //   continue;
    // }
    All.push_back(I);
  }
  Local.clear();
}

bool CilkSanitizerImpl::instrumentFunction(Function &F) {
  if (F.empty() || shouldNotInstrumentFunction(F))
    return false;

  DominatorTree *DT = &GetDomTree(F);
  // DetachSSA &DSSA = GetDSSA(F);
  // MemorySSA &MSSA = GetMSSA(F);

  SmallVector<Instruction*, 8> AllLoadsAndStores;
  SmallVector<Instruction*, 8> LocalLoadsAndStores;
  SmallVector<Instruction*, 8> AtomicAccesses;
  SmallVector<Instruction*, 8> MemIntrinCalls;
  SmallVector<Instruction *, 8> Callsites;
  SmallVector<DetachInst*, 8> Detaches;
  SmallVector<SyncInst*, 8> Syncs;
  bool Res = false;
  bool HasCalls = false;
  bool MaySpawn = false;

  // TODO: Consider modifying this to choose instrumentation to insert based on
  // fibrils, not basic blocks.
  for (BasicBlock &BB : F) {
    // Record the Tapir instructions found
    if (DetachInst *DI = dyn_cast<DetachInst>(BB.getTerminator())) {
      MaySpawn = true;
      Detaches.push_back(DI);
    } else if (SyncInst *SI = dyn_cast<SyncInst>(BB.getTerminator()))
      Syncs.push_back(SI);

    // Record the memory accesses in the basic block
    for (Instruction &Inst : BB) {
      if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
        LocalLoadsAndStores.push_back(&Inst);
      else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
        if (CallInst *CI = dyn_cast<CallInst>(&Inst))
          maybeMarkSanitizerLibraryCallNoBuiltin(CI, TLI);
        if (isa<MemIntrinsic>(Inst))
          MemIntrinCalls.push_back(&Inst);
        if (!isa<DbgInfoIntrinsic>(Inst)) {
          if (!isa<MemIntrinsic>(Inst))
            Callsites.push_back(&Inst);
          HasCalls = true;
          chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores,
                                         DL);
        }
      }
    }
    chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores, DL);
  }

  uint64_t LocalId = getLocalFunctionID(F);

  for (auto Inst : AllLoadsAndStores)
    Res |= instrumentLoadOrStore(Inst, DL);

  for (auto Inst : MemIntrinCalls)
    Res |= instrumentMemIntrinsic(Inst);

  for (auto Inst : Callsites)
    Res |= instrumentCallsite(Inst, DT);

  for (auto Inst : Detaches)
    Res |= instrumentDetach(Inst, DT);

  for (auto Inst : Syncs)
    Res |= instrumentSync(Inst);

  if ((Res || HasCalls)) {
    IRBuilder<> IRB(&*F.getEntryBlock().getFirstInsertionPt());
    CsiFuncProperty FuncEntryProp;
    FuncEntryProp.setMaySpawn(MaySpawn);
    Value *FuncId = FunctionFED.localToGlobalId(LocalId, IRB);
    // TODO: Determine if we actually want the frame pointer, not the stack
    // pointer.
    // Value *StackSave = IRB.CreateCall(
    //     Intrinsic::getDeclaration(&M, Intrinsic::stacksave));
    // IRB.CreateCall(CsanFuncEntry, {FuncId, StackSave, FuncEntryProp.getValue(IRB)});
    Value *FrameAddr = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::frameaddress),
        {IRB.getInt32(0)});
    IRB.CreateCall(CsanFuncEntry, {FuncId, FrameAddr, FuncEntryProp.getValue(IRB)});

    EscapeEnumerator EE(F, "csan_cleanup", true);
    while (IRBuilder<> *AtExit = EE.Next()) {
      // uint64_t ExitLocalId = FunctionExitFED.add(F);
      uint64_t ExitLocalId = FunctionExitFED.add(*AtExit->GetInsertPoint());
      Value *ExitCsiId = FunctionExitFED.localToGlobalId(ExitLocalId, *AtExit);
      CsiFuncExitProperty FuncExitProp;
      FuncExitProp.setMaySpawn(MaySpawn);
      AtExit->CreateCall(CsanFuncExit,
                         {ExitCsiId, FuncId, FuncExitProp.getValue(*AtExit)});
    }
  }
  return Res;
}

bool CilkSanitizerImpl::instrumentLoadOrStore(Instruction *I,
                                              const DataLayout &DL) {
  IRBuilder<> IRB(I);
  bool IsWrite = isa<StoreInst>(*I);
  Value *Addr = IsWrite
      ? cast<StoreInst>(I)->getPointerOperand()
      : cast<LoadInst>(I)->getPointerOperand();

  // swifterror memory addresses are mem2reg promoted by instruction selection.
  // As such they cannot have regular uses like an instrumentation function and
  // it makes no sense to track them as memory.
  if (Addr->isSwiftError())
    return false;

  const unsigned Alignment = IsWrite
      ? cast<StoreInst>(I)->getAlignment()
      : cast<LoadInst>(I)->getAlignment();
  CsiLoadStoreProperty Prop;
  Prop.setAlignment(Alignment);
  if (IsWrite) {
    uint64_t LocalId = StoreFED.add(*I);
    uint64_t StoreObjId = StoreObj.add(*I, DL);
    assert(LocalId == StoreObjId &&
           "Store received different ID's in FED and object tables.");
    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    Value *Args[] = {CsiId,
                     IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                     IRB.getInt32(getNumBytesAccessed(Addr, DL)),
                     Prop.getValue(IRB)};
    Instruction *Call = IRB.CreateCall(CsanWrite, Args);
    IRB.SetInstDebugLocation(Call);
    NumInstrumentedWrites++;
  } else {
    uint64_t LocalId = LoadFED.add(*I);
    uint64_t LoadObjId = LoadObj.add(*I, DL);
    assert(LocalId == LoadObjId &&
           "Load received different ID's in FED and object tables.");
    Value *CsiId = LoadFED.localToGlobalId(LocalId, IRB);
    Value *Args[] = {CsiId,
                     IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                     IRB.getInt32(getNumBytesAccessed(Addr, DL)),
                     Prop.getValue(IRB)};
    Instruction *Call = IRB.CreateCall(CsanRead, Args);
    IRB.SetInstDebugLocation(Call);
    NumInstrumentedReads++;
  }
  return true;
}

bool CilkSanitizerImpl::instrumentCallsite(Instruction *I, DominatorTree *DT) {
  // Exclude calls to the syncregion.start intrinsic.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
    if (Intrinsic::syncregion_start == II->getIntrinsicID() ||
        Intrinsic::lifetime_start == II->getIntrinsicID() ||
        Intrinsic::lifetime_end == II->getIntrinsicID())
      return false;

  bool IsInvoke = isa<InvokeInst>(I);

  Function *Called = NULL;
  if (CallInst *CI = dyn_cast<CallInst>(I))
    Called = CI->getCalledFunction();
  else if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    Called = II->getCalledFunction();

  IRBuilder<> IRB(I);
  uint64_t LocalId = CallsiteFED.add(*I);
  Value *CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
  Value *FuncId = NULL;
  GlobalVariable *FuncIdGV = NULL;
  if (Called) {
    Module *M = I->getParent()->getParent()->getParent();
    std::string GVName =
      CsiFuncIdVariablePrefix + Called->getName().str();
    FuncIdGV = dyn_cast<GlobalVariable>(M->getOrInsertGlobal(GVName,
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
  CsiCallProperty Prop;
  Prop.setIsIndirect(!Called);
  Value *PropVal = Prop.getValue(IRB);
  insertConditionalHookCall(I, CsiBeforeCallsite,
                            {CallsiteId, FuncId, PropVal});

  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block
    // and the exception block. This also means we have to recompute
    // the callsite and function IDs in each basic block so that we
    // can use it for the after hook.

    // TODO: Do we want the "after" hook for this callsite to come
    // before or after the BB entry hook? Currently it is inserted
    // before BB entry because instrumentCallsite is called after
    // instrumentBasicBlock.

    // TODO: If a destination of an invoke has multiple predecessors, then we
    // must split that destination.
    InvokeInst *II = dyn_cast<InvokeInst>(I);
    BasicBlock *NormalBB = II->getNormalDest();
    unsigned SuccNum = GetSuccessorNumber(II->getParent(), NormalBB);
    if (isCriticalEdge(II, SuccNum))
      NormalBB = SplitCriticalEdge(II, SuccNum,
                                   CriticalEdgeSplittingOptions(DT));
    IRB.SetInsertPoint(&*NormalBB->getFirstInsertionPt());
    CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
    if (FuncIdGV != NULL) FuncId = IRB.CreateLoad(FuncIdGV);
    PropVal = Prop.getValue(IRB);
    insertConditionalHookCall(&*IRB.GetInsertPoint(), CsiAfterCallsite,
                              {CallsiteId, FuncId, PropVal});

    BasicBlock *UnwindBB = II->getUnwindDest();
    IRB.SetInsertPoint(&*UnwindBB->getFirstInsertionPt());
    CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
    if (FuncIdGV != NULL) FuncId = IRB.CreateLoad(FuncIdGV);
    PropVal = Prop.getValue(IRB);
    insertConditionalHookCall(&*IRB.GetInsertPoint(), CsiAfterCallsite,
                              {CallsiteId, FuncId, PropVal});
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    PropVal = Prop.getValue(IRB);
    insertConditionalHookCall(&*Iter, CsiAfterCallsite,
                              {CallsiteId, FuncId, PropVal});
  }

  return true;
}

bool CilkSanitizerImpl::instrumentDetach(DetachInst *DI,
                                         DominatorTree *DT) {
  // Instrument the detach instruction itself
  Value *DetachID;
  {
    IRBuilder<> IRB(DI);
    uint64_t LocalID = DetachFED.add(*DI);
    DetachID = DetachFED.localToGlobalId(LocalID, IRB);
    Instruction *Call = IRB.CreateCall(CsanDetach, {DetachID});
    IRB.SetInstDebugLocation(Call);
  }
  NumInstrumentedDetaches++;

  // Find the detached block, continuation, and associated reattaches.
  BasicBlock *DetachedBlock = DI->getDetached();
  BasicBlock *ContinueBlock = DI->getContinue();
  SmallVector<BasicBlock *, 8> TaskExits;
  // TODO: Extend this loop to find EH exits of the detached task.
  for (BasicBlock *Pred : predecessors(ContinueBlock))
    if (isa<ReattachInst>(Pred->getTerminator()))
      TaskExits.push_back(Pred);

  // Instrument the entry and exit points of the detached task.
  {
    // Instrument the entry point of the detached task.
    IRBuilder<> IRB(&*DetachedBlock->getFirstInsertionPt());
    uint64_t LocalID = TaskFED.add(*DetachedBlock);
    Value *TaskID = TaskFED.localToGlobalId(LocalID, IRB);
    // TODO: Determine if we actually want the frame pointer, not the stack
    // pointer.
    // Value *StackSave = IRB.CreateCall(
    //     Intrinsic::getDeclaration(&M, Intrinsic::stacksave));
    // Instruction *Call = IRB.CreateCall(CsanTaskEntry,
    //                                    {TaskID, DetachID, StackSave});
    Value *FrameAddr = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::frameaddress),
        {IRB.getInt32(0)});
    Instruction *Call = IRB.CreateCall(CsanTaskEntry,
                                       {TaskID, DetachID, FrameAddr});
    IRB.SetInstDebugLocation(Call);

    // Instrument the exit points of the detached tasks.
    for (BasicBlock *TaskExit : TaskExits) {
      IRBuilder<> IRB(TaskExit->getTerminator());
      uint64_t LocalID = TaskExitFED.add(*TaskExit->getTerminator());
      Value *TaskExitID = TaskExitFED.localToGlobalId(LocalID, IRB);
      Instruction *Call = IRB.CreateCall(CsanTaskExit,
                                         {TaskExitID, TaskID, DetachID});
      IRB.SetInstDebugLocation(Call);
      NumInstrumentedDetachExits++;
    }
  }

  // Instrument the continuation of the detach.
  {
    if (isCriticalContinueEdge(DI, 1))
      ContinueBlock = SplitCriticalEdge(
          DI, 1,
          CriticalEdgeSplittingOptions(DT).setSplitDetachContinue());

    IRBuilder<> IRB(&*ContinueBlock->getFirstInsertionPt());
    uint64_t LocalID = DetachContinueFED.add(*ContinueBlock);
    Value *ContinueID = DetachContinueFED.localToGlobalId(LocalID, IRB);
    Instruction *Call = IRB.CreateCall(CsanDetachContinue,
                                       {ContinueID, DetachID});
    IRB.SetInstDebugLocation(Call);
  }
  return true;
}

bool CilkSanitizerImpl::instrumentSync(SyncInst *SI) {
  IRBuilder<> IRB(SI);
  // Get the ID of this sync.
  uint64_t LocalID = SyncFED.add(*SI);
  Value *SyncID = SyncFED.localToGlobalId(LocalID, IRB);
  // Insert instrumentation before the sync.
  Instruction *Call = IRB.CreateCall(CsanSync, {SyncID});
  IRB.SetInstDebugLocation(Call);
  NumInstrumentedSyncs++;
  return true;
}

bool CilkSanitizer::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  // auto GetDSSA = [this](Function &F) -> DetachSSA & {
  //   return this->getAnalysis<DetachSSAWrapperPass>(F).getDSSA();
  // };
  // auto GetMSSA = [this](Function &F) -> MemorySSA & {
  //   return this->getAnalysis<MemorySSAWrapperPass>(F).getMSSA();
  // };

  CallGraph *CG = &getAnalysis<CallGraphWrapperPass>().getCallGraph();
  const TargetLibraryInfo *TLI =
      &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  auto GetDomTree = [this](Function &F) -> DominatorTree & {
    return this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
  };

  // return CilkSanitizerImpl(M, CG, GetDSSA, GetMSSA).run();
  return CilkSanitizerImpl(M, CG, GetDomTree, TLI).run();
}
