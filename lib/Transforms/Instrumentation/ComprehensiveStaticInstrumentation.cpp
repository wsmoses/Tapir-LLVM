#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace {
const char *const CsiRtUnitInitName = "__csirt_unit_init";
const char *const CsiRtUnitCtorName = "csirt.unit_ctor";
const char *const CsiFunctionBaseIdName = "__csi_unit_func_base_id";
const char *const CsiFunctionExitBaseIdName = "__csi_unit_func_exit_base_id";
const char *const CsiBasicBlockBaseIdName = "__csi_unit_bb_base_id";
const char *const CsiCallsiteBaseIdName = "__csi_unit_callsite_base_id";
const char *const CsiLoadBaseIdName = "__csi_unit_load_base_id";
const char *const CsiStoreBaseIdName = "__csi_unit_store_base_id";
const char *const CsiUnitFedTableName = "__csi_unit_fed_table";
const char *const CsiFuncIdVariablePrefix = "__csi_func_id_";
const char *const CsiUnitFedTableArrayName = "__csi_unit_fed_tables";
const char *const CsiInitCallsiteToFunctionName =
    "__csi_init_callsite_to_function";
const char *const CsiDisableInstrumentationName =
    "__csi_disable_instrumentation";

const int64_t CsiCallsiteUnknownTargetId = -1;
// See llvm/tools/clang/lib/CodeGen/CodeGenModule.h:
const int CsiUnitCtorPriority = 65535;

/// Return the first DILocation in the given basic block, or nullptr
/// if none exists.
DILocation *getFirstDebugLoc(BasicBlock &BB) {
  for (Instruction &Inst : BB) {
    if (DILocation *Loc = Inst.getDebugLoc()) {
      return Loc;
    }
  }
  return nullptr;
}

/// Set DebugLoc on the call instruction to a CSI hook, based on the
/// debug information of the instrumented instruction.
void setInstrumentationDebugLoc(Instruction *Instrumented, Instruction *Call) {
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
void setInstrumentationDebugLoc(BasicBlock &Instrumented, Instruction *Call) {
  DISubprogram *Subprog = Instrumented.getParent()->getSubprogram();
  if (Subprog) {
    LLVMContext &C = Instrumented.getParent()->getParent()->getContext();
    Call->setDebugLoc(DILocation::get(C, 0, 0, Subprog));
  }
}

/// Set DebugLoc on the call instruction to a CSI hook, based on the
/// debug information of the instrumented instruction.
void setInstrumentationDebugLoc(Function &Instrumented, Instruction *Call) {
  DISubprogram *Subprog = Instrumented.getSubprogram();
  if (Subprog) {
    LLVMContext &C = Instrumented.getParent()->getContext();
    Call->setDebugLoc(DILocation::get(C, 0, 0, Subprog));
  }
}

/// Maintains a mapping from CSI ID to front-end data for that ID.
///
/// The front-end data currently is the source location that a given
/// CSI ID corresponds to.
class FrontEndDataTable {
public:
  FrontEndDataTable() : BaseId(nullptr), IdCounter(0) {}
  FrontEndDataTable(Module &M, StringRef BaseIdName);

  /// The number of entries in this FED table
  uint64_t size() const { return LocalIdToSourceLocationMap.size(); }

  /// The GlobalVariable holding the base ID for this FED table.
  GlobalVariable *baseId() const { return BaseId; }

  /// Add the given Function to this FED table.
  /// \returns The local ID of the Function.
  uint64_t add(Function &F);

  /// Add the given BasicBlock to this FED table.
  /// \returns The local ID of the BasicBlock.
  uint64_t add(BasicBlock &BB);

  /// Add the given Instruction to this FED table.
  /// \returns The local ID of the Instruction.
  uint64_t add(Instruction &I);

  /// Get the local ID of the given Value.
  uint64_t getId(Value *V);

  /// Converts a local to global ID conversion.
  ///
  /// This is done by using the given IRBuilder to insert a load to
  /// the base ID global variable followed by an add of the base value
  /// and the local ID.
  ///
  /// \returns A Value holding the global ID corresponding to the
  /// given local ID.
  Value *localToGlobalId(uint64_t LocalId, IRBuilder<> IRB) const;

  /// Get the Type for a pointer to a FED table entry.
  ///
  /// A FED table entry is just a source location.
  static PointerType *getPointerType(LLVMContext &C);

  /// Insert this FED table into the given Module.
  ///
  /// The FED table is constructed as a ConstantArray indexed by local
  /// IDs.  The runtime is responsible for performing the mapping that
  /// allows the table to be indexed by global ID.
  Constant *insertIntoModule(Module &M) const;

private:
  struct SourceLocation {
    StringRef Name;
    int32_t Line;
    StringRef File;
  };

  /// The GlobalVariable holding the base ID for this FED table.
  GlobalVariable *BaseId;
  /// Counter of local IDs used so far.
  uint64_t IdCounter;
  /// Map of local ID to SourceLocation.
  std::map<uint64_t, SourceLocation> LocalIdToSourceLocationMap;
  /// Map of Value to Local ID.
  std::map<Value *, uint64_t> ValueToLocalIdMap;

  /// Create a struct type to match the "struct SourceLocation" type.
  /// (and the source_loc_t type in csi.h).
  static StructType *getSourceLocStructType(LLVMContext &C);

  /// Append the debug information to the table, assigning it the next
  /// available ID.
  ///
  /// \returns The local ID of the appended information.
  /// @{
  uint64_t add(DILocation *Loc);
  uint64_t add(DISubprogram *Subprog);
  /// @}

  /// Append the line and file information to the table, assigning it
  /// the next available ID.
  ///
  /// \returns The new local ID of the DILocation.
  uint64_t add(int32_t Line, StringRef File, StringRef Name = "");
};

/// Represents a csi_prop_t value passed to hooks.
class CsiProperty {
public:
  CsiProperty() {
    PropValue.LoadReadBeforeWriteInBB = false;
  }

  /// Return the Type of a property.
  static StructType *getType(LLVMContext &C);

  /// Return a Value holding this property.
  Value *getValue(IRBuilder<> IRB) const;

  /// Set the value of the LoadReadBeforeWriteInBB property.
  void setLoadReadBeforeWriteInBB(bool v);
private:
  typedef struct {
    bool LoadReadBeforeWriteInBB;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;
};

/// The Comprehensive Static Instrumentation pass.
/// Inserts calls to user-defined hooks at predefined points in the IR.
struct ComprehensiveStaticInstrumentation : public ModulePass {
  static char ID;

  ComprehensiveStaticInstrumentation() : ModulePass(ID) {}
  StringRef getPassName() const override;
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  /// Initialize llvm::Functions for the CSI hooks.
  /// @{
  void initializeLoadStoreHooks(Module &M);
  void initializeFuncHooks(Module &M);
  void initializeBasicBlockHooks(Module &M);
  void initializeCallsiteHooks(Module &M);
  /// @}

  /// Initialize the front-end data table structures.
  void initializeFEDTables(Module &M);

  /// Generate a function that stores global function IDs into a set
  /// of externally-visible global variables.
  void generateInitCallsiteToFunction(Module &M);

  /// Get the number of bytes accessed via the given address.
  int getNumBytesAccessed(Value *Addr, const DataLayout &DL);

  /// Compute CSI properties on the given ordered list of loads and stores.
  void computeLoadAndStoreProperties(
      SmallVectorImpl<std::pair<Instruction *, CsiProperty>>
          &LoadAndStoreProperties,
      SmallVectorImpl<Instruction *> &BBLoadsAndStores);

  /// Insert calls to the instrumentation hooks.
  /// @{
  void addLoadStoreInstrumentation(Instruction *I, Function *BeforeFn,
                                   Function *AfterFn, Value *CsiId,
                                   Type *AddrType, Value *Addr, int NumBytes,
                                   CsiProperty Prop);
  void instrumentLoadOrStore(Instruction *I, CsiProperty Prop,
                             const DataLayout &DL);
  void instrumentMemIntrinsic(Instruction *I);
  void instrumentCallsite(Instruction *I);
  void instrumentBasicBlock(BasicBlock &BB);
  void instrumentFunction(Function &F);
  /// @}

  /// Insert a conditional call to the given hook function before the
  /// given instruction. The condition is based on the value of
  /// __csi_disable_instrumentation.
  void insertConditionalHookCall(Instruction *I, Function *HookFunction,
                                 ArrayRef<Value *> HookArgs);

  /// Return true if the given function should not be instrumented.
  bool shouldNotInstrumentFunction(Function &F);

  /// Initialize the CSI pass.
  void initializeCsi(Module &M);
  /// Finalize the CSI pass.
  void finalizeCsi(Module &M);

  FrontEndDataTable FunctionFED, FunctionExitFED, BasicBlockFED, CallsiteFED,
      LoadFED, StoreFED;

  Function *CsiBeforeCallsite, *CsiAfterCallsite;
  Function *CsiFuncEntry, *CsiFuncExit;
  Function *CsiBBEntry, *CsiBBExit;
  Function *CsiBeforeRead, *CsiAfterRead;
  Function *CsiBeforeWrite, *CsiAfterWrite;

  CallGraph *CG;
  Function *MemmoveFn, *MemcpyFn, *MemsetFn;
  Function *InitCallsiteToFunction;
  GlobalVariable *DisableInstrGV;
  Type *IntptrTy;
  std::map<std::string, uint64_t> FuncOffsetMap;
}; // struct ComprehensiveStaticInstrumentation
} // anonymous namespace

char ComprehensiveStaticInstrumentation::ID = 0;

INITIALIZE_PASS(ComprehensiveStaticInstrumentation, "csi",
                "ComprehensiveStaticInstrumentation pass", false, false)

StringRef ComprehensiveStaticInstrumentation::getPassName() const {
  return "ComprehensiveStaticInstrumentation";
}

ModulePass *llvm::createComprehensiveStaticInstrumentationPass() {
  return new ComprehensiveStaticInstrumentation();
}

FrontEndDataTable::FrontEndDataTable(Module &M, StringRef BaseIdName) {
  LLVMContext &C = M.getContext();
  IntegerType *Int64Ty = IntegerType::get(C, 64);
  IdCounter = 0;
  BaseId = new GlobalVariable(M, Int64Ty, false, GlobalValue::InternalLinkage,
                              ConstantInt::get(Int64Ty, 0), BaseIdName);
  assert(BaseId);
}

uint64_t FrontEndDataTable::add(Function &F) {
  uint64_t Id = add(F.getSubprogram());
  ValueToLocalIdMap[&F] = Id;
  return Id;
}

uint64_t FrontEndDataTable::add(BasicBlock &BB) {
  uint64_t Id = add(getFirstDebugLoc(BB));
  ValueToLocalIdMap[&BB] = Id;
  return Id;
}

uint64_t FrontEndDataTable::add(Instruction &I) {
  uint64_t Id = add(I.getDebugLoc());
  ValueToLocalIdMap[&I] = Id;
  return Id;
}

uint64_t FrontEndDataTable::getId(Value *V) {
  assert(ValueToLocalIdMap.find(V) != ValueToLocalIdMap.end() &&
         "Value not in ID map.");
  return ValueToLocalIdMap[V];
}

Value *FrontEndDataTable::localToGlobalId(uint64_t LocalId,
                                          IRBuilder<> IRB) const {
  assert(BaseId);
  Value *Base = IRB.CreateLoad(BaseId);
  Value *Offset = IRB.getInt64(LocalId);
  return IRB.CreateAdd(Base, Offset);
}

PointerType *FrontEndDataTable::getPointerType(LLVMContext &C) {
  return PointerType::get(getSourceLocStructType(C), 0);
}

StructType *FrontEndDataTable::getSourceLocStructType(LLVMContext &C) {
  return StructType::get(/* Name */ PointerType::get(IntegerType::get(C, 8), 0),
                         /* Line */ IntegerType::get(C, 32),
                         /* File */ PointerType::get(IntegerType::get(C, 8), 0),
                         nullptr);
}

uint64_t FrontEndDataTable::add(DILocation *Loc) {
  if (Loc) {
    return add((int32_t)Loc->getLine(), Loc->getFilename());
  } else {
    return add(-1, "");
  }
}

uint64_t FrontEndDataTable::add(DISubprogram *Subprog) {
  if (Subprog) {
    return add((int32_t)Subprog->getLine(),
               (Subprog->getDirectory() + Subprog->getFilename()).str(),
               Subprog->getName());
  } else {
    return add(-1, "", "");
  }
}

uint64_t FrontEndDataTable::add(int32_t Line, StringRef File, StringRef Name) {
  uint64_t Id = IdCounter++;
  assert(LocalIdToSourceLocationMap.find(Id) ==
             LocalIdToSourceLocationMap.end() &&
         "Id already exists in FED table.");
  LocalIdToSourceLocationMap[Id] = {Name, Line, File};
  return Id;
}

Constant *FrontEndDataTable::insertIntoModule(Module &M) const {
  LLVMContext &C = M.getContext();
  StructType *FedType = getSourceLocStructType(C);
  IntegerType *Int32Ty = IntegerType::get(C, 32);
  Constant *Zero = ConstantInt::get(Int32Ty, 0);
  Value *GepArgs[] = {Zero, Zero};
  SmallVector<Constant *, 4> FEDEntries;

  for (const auto it : LocalIdToSourceLocationMap) {
    const SourceLocation &E = it.second;
    Value *Line = ConstantInt::get(Int32Ty, E.Line);
    Constant *File;
    {
      Constant *FileStrConstant = ConstantDataArray::getString(C, E.File);
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
    {
      Constant *NameStrConstant = ConstantDataArray::getString(C, E.Name);
      GlobalVariable *GV =
        M.getGlobalVariable(("__csi_unit_function_name_" + E.Name).str(), true);
      if (GV == NULL) {
        GV = new GlobalVariable(M, NameStrConstant->getType(),
                                true, GlobalValue::PrivateLinkage,
                                NameStrConstant,
                                "__csi_unit_function_name_" + E.Name,
                                nullptr,
                                GlobalVariable::NotThreadLocal, 0);
        GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      }
      assert(GV);
      Name =
        ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
    }
    FEDEntries.push_back(ConstantStruct::get(FedType, Name, Line, File,
                                             nullptr));
  }

  ArrayType *FedArrayType = ArrayType::get(FedType, FEDEntries.size());
  Constant *Table = ConstantArray::get(FedArrayType, FEDEntries);
  GlobalVariable *GV =
    new GlobalVariable(M, FedArrayType, false, GlobalValue::InternalLinkage,
                       Table, CsiUnitFedTableName);
  return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
}

StructType *CsiProperty::getType(LLVMContext &C) {
  // Must match the definition of csi_prop_t in csi.h
  return StructType::get(IntegerType::get(C, 1), IntegerType::get(C, 63), nullptr);
}

Value *CsiProperty::getValue(IRBuilder<> IRB) const {
  LLVMContext &C = IRB.getContext();
  Constant *Value = ConstantStruct::get(getType(C),
        ConstantInt::get(IntegerType::get(C, 1), PropValue.LoadReadBeforeWriteInBB),
        ConstantInt::get(IntegerType::get(C, 63), 0),
        nullptr);
  Type *StructTy = getType(C);
  Type *Int64PtrTy = PointerType::get(IntegerType::get(C, 64), 0);
  AllocaInst *AI = IRB.CreateAlloca(StructTy);
  IRB.CreateStore(Value, AI);
  return IRB.CreateLoad(IRB.CreateBitCast(IRB.CreateInBoundsGEP(AI,
                                                                {IRB.getInt32(0), IRB.getInt32(0)}),
                                          Int64PtrTy));
}

void CsiProperty::setLoadReadBeforeWriteInBB(bool v) {
  PropValue.LoadReadBeforeWriteInBB = v;
}

void ComprehensiveStaticInstrumentation::initializeFuncHooks(Module &M) {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  CsiFuncEntry = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_func_entry", IRB.getVoidTy(), IRB.getInt64Ty(), IRB.getInt64Ty(), nullptr));
  CsiFuncExit = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_func_exit", IRB.getVoidTy(),
                            IRB.getInt64Ty(), IRB.getInt64Ty(), IRB.getInt64Ty(), nullptr));
}

void ComprehensiveStaticInstrumentation::initializeBasicBlockHooks(Module &M) {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  CsiBBEntry = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_bb_entry", IRB.getVoidTy(), IRB.getInt64Ty(), IRB.getInt64Ty(), nullptr));
  CsiBBExit = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_bb_exit", IRB.getVoidTy(), IRB.getInt64Ty(), IRB.getInt64Ty(), nullptr));
}

void ComprehensiveStaticInstrumentation::initializeCallsiteHooks(Module &M) {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  CsiBeforeCallsite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_call", IRB.getVoidTy(),
                            IRB.getInt64Ty(), IRB.getInt64Ty(), IRB.getInt64Ty(), nullptr));
  CsiAfterCallsite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_call", IRB.getVoidTy(),
                            IRB.getInt64Ty(), IRB.getInt64Ty(), IRB.getInt64Ty(), nullptr));
}

void ComprehensiveStaticInstrumentation::initializeLoadStoreHooks(Module &M) {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *RetType = IRB.getVoidTy();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *NumBytesType = IRB.getInt32Ty();

  CsiBeforeRead = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_load", RetType, IRB.getInt64Ty(),
                            AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));
  CsiAfterRead = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_load", RetType, IRB.getInt64Ty(),
                            AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  CsiBeforeWrite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_store", RetType, IRB.getInt64Ty(),
                            AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));
  CsiAfterWrite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_store", RetType, IRB.getInt64Ty(),
                            AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  MemmoveFn = checkCsiInterfaceFunction(
      M.getOrInsertFunction("memmove", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemcpyFn = checkCsiInterfaceFunction(
      M.getOrInsertFunction("memcpy", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemsetFn = checkCsiInterfaceFunction(
      M.getOrInsertFunction("memset", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt32Ty(), IntptrTy, nullptr));
}

int ComprehensiveStaticInstrumentation::getNumBytesAccessed(
    Value *Addr, const DataLayout &DL) {
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize != 8 && TypeSize != 16 && TypeSize != 32 && TypeSize != 64 &&
      TypeSize != 128) {
    return -1;
  }
  return TypeSize / 8;
}

void ComprehensiveStaticInstrumentation::addLoadStoreInstrumentation(
    Instruction *I, Function *BeforeFn, Function *AfterFn, Value *CsiId,
    Type *AddrType, Value *Addr, int NumBytes, CsiProperty Prop) {
  IRBuilder<> IRB(I);
  Value *PropVal = Prop.getValue(IRB);
  insertConditionalHookCall(I, BeforeFn, {CsiId, IRB.CreatePointerCast(Addr, AddrType),
        IRB.getInt32(NumBytes), PropVal});

  BasicBlock::iterator Iter(I);
  Iter++;
  IRB.SetInsertPoint(&*Iter);
  PropVal = Prop.getValue(IRB);
  insertConditionalHookCall(&*Iter, AfterFn, {CsiId, IRB.CreatePointerCast(Addr, AddrType),
        IRB.getInt32(NumBytes), PropVal});
}

void ComprehensiveStaticInstrumentation::instrumentLoadOrStore(
    Instruction *I, CsiProperty Prop, const DataLayout &DL) {
  IRBuilder<> IRB(I);
  bool IsWrite = isa<StoreInst>(I);
  Value *Addr = IsWrite ? cast<StoreInst>(I)->getPointerOperand()
                        : cast<LoadInst>(I)->getPointerOperand();
  int NumBytes = getNumBytesAccessed(Addr, DL);
  Type *AddrType = IRB.getInt8PtrTy();

  if (NumBytes == -1)
    return; // size that we don't recognize

  if (IsWrite) {
    uint64_t LocalId = StoreFED.add(*I);
    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    addLoadStoreInstrumentation(I, CsiBeforeWrite, CsiAfterWrite, CsiId,
                                AddrType, Addr, NumBytes, Prop);
  } else { // is read
    uint64_t LocalId = LoadFED.add(*I);
    Value *CsiId = LoadFED.localToGlobalId(LocalId, IRB);
    addLoadStoreInstrumentation(I, CsiBeforeRead, CsiAfterRead, CsiId, AddrType,
                                Addr, NumBytes, Prop);
  }
}

// If a memset intrinsic gets inlined by the code gen, we will miss races on it.
// So, we either need to ensure the intrinsic is not inlined, or instrument it.
// We do not instrument memset/memmove/memcpy intrinsics (too complicated),
// instead we simply replace them with regular function calls, which are then
// intercepted by the run-time.
// Since our pass runs after everyone else, the calls should not be
// replaced back with intrinsics. If that becomes wrong at some point,
// we will need to call e.g. __csi_memset to avoid the intrinsics.
void ComprehensiveStaticInstrumentation::instrumentMemIntrinsic(
    Instruction *I) {
  IRBuilder<> IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    Instruction *Call = IRB.CreateCall(
                                       MemsetFn,
                                       {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
                                           IRB.CreateIntCast(M->getArgOperand(1), IRB.getInt32Ty(), false),
                                           IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    setInstrumentationDebugLoc(I, Call);
    I->eraseFromParent();
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    Instruction *Call = IRB.CreateCall(
                                       isa<MemCpyInst>(M) ? MemcpyFn : MemmoveFn,
                                       {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
                                           IRB.CreatePointerCast(M->getArgOperand(1), IRB.getInt8PtrTy()),
                                           IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    setInstrumentationDebugLoc(I, Call);
    I->eraseFromParent();
  }
}

void ComprehensiveStaticInstrumentation::instrumentBasicBlock(BasicBlock &BB) {
  IRBuilder<> IRB(&*BB.getFirstInsertionPt());
  //LLVMContext &C = IRB.getContext();
  uint64_t LocalId = BasicBlockFED.add(BB);
  Value *CsiId = BasicBlockFED.localToGlobalId(LocalId, IRB);
  CsiProperty Prop;
  TerminatorInst *TI = BB.getTerminator();
  Value *PropVal = Prop.getValue(IRB);

  insertConditionalHookCall(&*IRB.GetInsertPoint(), CsiBBEntry, {CsiId, PropVal});
  insertConditionalHookCall(TI, CsiBBExit, {CsiId, PropVal});
}

void ComprehensiveStaticInstrumentation::instrumentCallsite(Instruction *I) {
  Function *Called = NULL;
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    Called = CI->getCalledFunction();
  } else if (InvokeInst *II = dyn_cast<InvokeInst>(I)) {
    Called = II->getCalledFunction();
  }

  if (Called && Called->getName().startswith("llvm.dbg")) {
    return;
  }

  IRBuilder<> IRB(I);
  //LLVMContext &C = IRB.getContext();
  uint64_t LocalId = CallsiteFED.add(*I);
  Value *CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
  Value *FuncId = NULL;
  if (Called) {
    Module *M = I->getParent()->getParent()->getParent();
    std::string GVName = CsiFuncIdVariablePrefix + Called->getName().str();
    GlobalVariable *FuncIdGV =
      dyn_cast<GlobalVariable>(M->getOrInsertGlobal(GVName, IRB.getInt64Ty()));
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
  CsiProperty Prop;
  Value *PropVal = Prop.getValue(IRB);
  insertConditionalHookCall(I, CsiBeforeCallsite, {CallsiteId, FuncId, PropVal});

  BasicBlock::iterator Iter(I);
  Iter++;
  IRB.SetInsertPoint(&*Iter);
  PropVal = Prop.getValue(IRB);
  insertConditionalHookCall(&*Iter, CsiAfterCallsite, {CallsiteId, FuncId, PropVal});
}

void ComprehensiveStaticInstrumentation::insertConditionalHookCall(Instruction *I, Function *HookFunction, ArrayRef<Value *> HookArgs) {
  IRBuilder<> IRB(I);
  // Value *Cond = IRB.CreateICmpEQ(IRB.CreateLoad(DisableInstrGV), IRB.getInt1(false));
  // TerminatorInst *TI = SplitBlockAndInsertIfThen(Cond, I, false);
  // IRB.SetInsertPoint(TI);
  // IRB.CreateStore(IRB.getInt1(true), DisableInstrGV);
  Instruction *Call = IRB.CreateCall(HookFunction, HookArgs);
  setInstrumentationDebugLoc(I, Call);
  // IRB.CreateStore(IRB.getInt1(false), DisableInstrGV);
}


void ComprehensiveStaticInstrumentation::initializeFEDTables(Module &M) {
  FunctionFED = FrontEndDataTable(M, CsiFunctionBaseIdName);
  FunctionExitFED = FrontEndDataTable(M, CsiFunctionExitBaseIdName);
  BasicBlockFED = FrontEndDataTable(M, CsiBasicBlockBaseIdName);
  CallsiteFED = FrontEndDataTable(M, CsiCallsiteBaseIdName);
  LoadFED = FrontEndDataTable(M, CsiLoadBaseIdName);
  StoreFED = FrontEndDataTable(M, CsiStoreBaseIdName);
}

void ComprehensiveStaticInstrumentation::generateInitCallsiteToFunction(
    Module &M) {
  LLVMContext &C = M.getContext();
  BasicBlock *EntryBB = BasicBlock::Create(C, "", InitCallsiteToFunction);
  IRBuilder<> IRB(ReturnInst::Create(C, EntryBB));

  GlobalVariable *Base = FunctionFED.baseId();
  LoadInst *LI = IRB.CreateLoad(Base);
  // Traverse the map of function name -> function local id. Generate
  // a store of each function's global ID to the corresponding weak
  // global variable.
  for (const auto &it : FuncOffsetMap) {
    std::string GVName = CsiFuncIdVariablePrefix + it.first;
    GlobalVariable *GV = nullptr;
    if ((GV = M.getGlobalVariable(GVName)) == nullptr) {
      GV = new GlobalVariable(M, IRB.getInt64Ty(), false,
                              GlobalValue::WeakAnyLinkage,
                              IRB.getInt64(CsiCallsiteUnknownTargetId), GVName);
    }
    assert(GV);
    IRB.CreateStore(IRB.CreateAdd(LI, IRB.getInt64(it.second)), GV);
  }
}

void ComprehensiveStaticInstrumentation::initializeCsi(Module &M) {
  IntptrTy = M.getDataLayout().getIntPtrType(M.getContext());

  initializeFEDTables(M);
  initializeFuncHooks(M);
  initializeLoadStoreHooks(M);
  initializeBasicBlockHooks(M);
  initializeCallsiteHooks(M);

  FunctionType *FnType =
      FunctionType::get(Type::getVoidTy(M.getContext()), {}, false);
  InitCallsiteToFunction = checkCsiInterfaceFunction(
      M.getOrInsertFunction(CsiInitCallsiteToFunctionName, FnType));
  assert(InitCallsiteToFunction);
  InitCallsiteToFunction->setLinkage(GlobalValue::InternalLinkage);

  CG = &getAnalysis<CallGraphWrapperPass>().getCallGraph();

  DisableInstrGV = new GlobalVariable(M, IntegerType::get(M.getContext(), 1), false,
                                      GlobalValue::ExternalLinkage, nullptr,
                                      CsiDisableInstrumentationName, nullptr,
                                      GlobalValue::GeneralDynamicTLSModel, 0, true);
}

// Create a struct type to match the unit_fed_entry_t type in csirt.c.
StructType *getUnitFedTableType(LLVMContext &C, PointerType *EntryPointerType) {
  return StructType::get(IntegerType::get(C, 64),
                         PointerType::get(IntegerType::get(C, 64), 0),
                         EntryPointerType, nullptr);
}

Constant *fedTableToUnitFedTable(Module &M, StructType *UnitFedTableType,
                                 FrontEndDataTable &FedTable) {
  Constant *NumEntries =
      ConstantInt::get(IntegerType::get(M.getContext(), 64), FedTable.size());
  Constant *InsertedTable = FedTable.insertIntoModule(M);
  return ConstantStruct::get(UnitFedTableType, NumEntries, FedTable.baseId(),
                             InsertedTable, nullptr);
}

void ComprehensiveStaticInstrumentation::finalizeCsi(Module &M) {
  LLVMContext &C = M.getContext();

  // Add CSI global constructor, which calls unit init.
  Function *Ctor =
      Function::Create(FunctionType::get(Type::getVoidTy(C), false),
                       GlobalValue::InternalLinkage, CsiRtUnitCtorName, &M);
  BasicBlock *CtorBB = BasicBlock::Create(C, "", Ctor);
  IRBuilder<> IRB(ReturnInst::Create(C, CtorBB));

  StructType *UnitFedTableType =
      getUnitFedTableType(C, FrontEndDataTable::getPointerType(C));

  // Lookup __csirt_unit_init
  SmallVector<Type *, 4> InitArgTypes({IRB.getInt8PtrTy(),
                                       PointerType::get(UnitFedTableType, 0),
                                       InitCallsiteToFunction->getType()});
  FunctionType *InitFunctionTy =
      FunctionType::get(IRB.getVoidTy(), InitArgTypes, false);
  Function *InitFunction = checkCsiInterfaceFunction(
      M.getOrInsertFunction(CsiRtUnitInitName, InitFunctionTy));
  assert(InitFunction);

  // Insert __csi_func_id_<f> weak symbols for all defined functions
  // and generate the runtime code that stores to all of them.
  generateInitCallsiteToFunction(M);

  SmallVector<Constant *, 4> UnitFedTables({
      fedTableToUnitFedTable(M, UnitFedTableType, BasicBlockFED),
      fedTableToUnitFedTable(M, UnitFedTableType, FunctionFED),
      fedTableToUnitFedTable(M, UnitFedTableType, FunctionExitFED),
      fedTableToUnitFedTable(M, UnitFedTableType, CallsiteFED),
      fedTableToUnitFedTable(M, UnitFedTableType, LoadFED),
      fedTableToUnitFedTable(M, UnitFedTableType, StoreFED),
  });

  ArrayType *UnitFedTableArrayType =
      ArrayType::get(UnitFedTableType, UnitFedTables.size());
  Constant *Table = ConstantArray::get(UnitFedTableArrayType, UnitFedTables);
  GlobalVariable *GV = new GlobalVariable(M, UnitFedTableArrayType, false,
                                          GlobalValue::InternalLinkage, Table,
                                          CsiUnitFedTableArrayName);

  Constant *Zero = ConstantInt::get(IRB.getInt32Ty(), 0);
  Value *GepArgs[] = {Zero, Zero};

  // Insert call to __csirt_unit_init
  CallInst *Call = IRB.CreateCall(
      InitFunction,
      {IRB.CreateGlobalStringPtr(M.getName()),
       ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs),
       InitCallsiteToFunction});

  // Add the constructor to the global list
  appendToGlobalCtors(M, Ctor, CsiUnitCtorPriority);

  CallGraphNode *CNCtor = CG->getOrInsertFunction(Ctor);
  CallGraphNode *CNFunc = CG->getOrInsertFunction(InitFunction);
  CNCtor->addCalledFunction(Call, CNFunc);
}

void ComprehensiveStaticInstrumentation::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<CallGraphWrapperPass>();
}

bool ComprehensiveStaticInstrumentation::shouldNotInstrumentFunction(
    Function &F) {
  Module &M = *F.getParent();
  // Never instrument the CSI ctor.
  if (F.hasName() && F.getName() == CsiRtUnitCtorName) {
    return true;
  }
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

void ComprehensiveStaticInstrumentation::computeLoadAndStoreProperties(
    SmallVectorImpl<std::pair<Instruction *, CsiProperty>> &LoadAndStoreProperties,
    SmallVectorImpl<Instruction *> &BBLoadsAndStores) {
  SmallSet<Value *, 8> WriteTargets;

  for (SmallVectorImpl<Instruction *>::reverse_iterator
           It = BBLoadsAndStores.rbegin(),
           E = BBLoadsAndStores.rend();
       It != E; ++It) {
    Instruction *I = *It;
    CsiProperty Prop;
    if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
      WriteTargets.insert(Store->getPointerOperand());
      LoadAndStoreProperties.push_back(std::make_pair(I, Prop));
    } else {
      LoadInst *Load = cast<LoadInst>(I);
      Value *Addr = Load->getPointerOperand();
      bool HasBeenSeen = WriteTargets.count(Addr) > 0;
      Prop.setLoadReadBeforeWriteInBB(HasBeenSeen);
      LoadAndStoreProperties.push_back(std::make_pair(I, Prop));
    }
  }
  BBLoadsAndStores.clear();
}

bool ComprehensiveStaticInstrumentation::runOnModule(Module &M) {
  initializeCsi(M);

  for (Function &F : M) {
    instrumentFunction(F);
  }

  finalizeCsi(M);
  return true; // We always insert the unit constructor.
}

void ComprehensiveStaticInstrumentation::instrumentFunction(Function &F) {
  // This is required to prevent instrumenting the call to
  // __csi_module_init from within the module constructor.
  if (F.empty() || shouldNotInstrumentFunction(F)) {
    return;
  }

  SmallVector<std::pair<Instruction *, CsiProperty>, 8> LoadAndStoreProperties;
  SmallVector<Instruction *, 8> ReturnInstructions;
  SmallVector<Instruction *, 8> MemIntrinsics;
  SmallVector<Instruction *, 8> Callsites;
  std::vector<BasicBlock *> BasicBlocks;
  const DataLayout &DL = F.getParent()->getDataLayout();

  // Compile lists of all instrumentation points before anything is modified.
  for (BasicBlock &BB : F) {
    SmallVector<Instruction *, 8> BBLoadsAndStores;
    for (Instruction &I : BB) {
      if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
        BBLoadsAndStores.push_back(&I);
      } else if (isa<ReturnInst>(I)) {
        ReturnInstructions.push_back(&I);
      } else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
        if (isa<MemIntrinsic>(I)) {
          MemIntrinsics.push_back(&I);
        } else {
          Callsites.push_back(&I);
        }
      }
    }
    computeLoadAndStoreProperties(LoadAndStoreProperties, BBLoadsAndStores);
    BasicBlocks.push_back(&BB);
  }


  // Instrument basic blocks Note that we do this before other
  // instrumentation so that we put this at the beginning of the basic
  // block, and then the function entry call goes before the call to
  // basic block entry.
  uint64_t LocalId = FunctionFED.add(F);
  FuncOffsetMap[F.getName()] = LocalId;
  for (BasicBlock *BB : BasicBlocks) {
    instrumentBasicBlock(*BB);
  }

  // Do this work in a separate loop after copying the iterators so that we
  // aren't modifying the list as we're iterating.
  for (std::pair<Instruction *, CsiProperty> p : LoadAndStoreProperties) {
    instrumentLoadOrStore(p.first, p.second, DL);
  }

  for (Instruction *I : MemIntrinsics) {
    instrumentMemIntrinsic(I);
  }

  for (Instruction *I : Callsites) {
    instrumentCallsite(I);
  }

  // Instrument function entry/exit points.
  IRBuilder<> IRB(&*F.getEntryBlock().getFirstInsertionPt());
  //LLVMContext &C = IRB.getContext();
  CsiProperty FuncEntryProp, FuncExitProp;
  Value *FuncId = FunctionFED.localToGlobalId(LocalId, IRB);
  Value *PropVal = FuncEntryProp.getValue(IRB);
  insertConditionalHookCall(&*IRB.GetInsertPoint(), CsiFuncEntry, {FuncId, PropVal});

  for (Instruction *I : ReturnInstructions) {
    IRBuilder<> IRBRet(I);
    uint64_t ExitLocalId = FunctionExitFED.add(F);
    Value *ExitCsiId = FunctionExitFED.localToGlobalId(ExitLocalId, IRBRet);
    PropVal = FuncExitProp.getValue(IRBRet);
    insertConditionalHookCall(I, CsiFuncExit, {ExitCsiId, FuncId, PropVal});
  }
}
