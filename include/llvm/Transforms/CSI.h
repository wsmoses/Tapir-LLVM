//===-- CSI.h -- CSI implementation structures and hooks -----*- C++ -*----===//
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

#ifndef LLVM_TRANSFORMS_CSI_H
#define LLVM_TRANSFORMS_CSI_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Instrumentation/SurgicalInstrumentationConfig.h"

namespace llvm {

class LoopInfo;
class Spindle;
class Task;
class TaskInfo;

static const char *const CsiRtUnitInitName = "__csirt_unit_init";
static const char *const CsiRtUnitCtorName = "csirt.unit_ctor";
static const char *const CsiFunctionBaseIdName = "__csi_unit_func_base_id";
static const char *const CsiFunctionExitBaseIdName =
    "__csi_unit_func_exit_base_id";
static const char *const CsiBasicBlockBaseIdName = "__csi_unit_bb_base_id";
static const char *const CsiCallsiteBaseIdName = "__csi_unit_callsite_base_id";
static const char *const CsiLoadBaseIdName = "__csi_unit_load_base_id";
static const char *const CsiStoreBaseIdName = "__csi_unit_store_base_id";
static const char *const CsiAllocaBaseIdName = "__csi_unit_alloca_base_id";
static const char *const CsiDetachBaseIdName = "__csi_unit_detach_base_id";
static const char *const CsiTaskBaseIdName = "__csi_unit_task_base_id";
static const char *const CsiTaskExitBaseIdName = "__csi_unit_task_exit_base_id";
static const char *const CsiDetachContinueBaseIdName =
    "__csi_unit_detach_continue_base_id";
static const char *const CsiSyncBaseIdName = "__csi_unit_sync_base_id";
static const char *const CsiAllocFnBaseIdName = "__csi_unit_allocfn_base_id";
static const char *const CsiFreeBaseIdName = "__csi_unit_free_base_id";
static const char *const CsiArithmeticBaseIdName = "__csi_unit_arithmetic_base_id";
static const char *const CsiParameterBaseIdName = "__csi_unit_parameter_base_id";
static const char *const CsiGlobalBaseIdName = "__csi_unit_global_base_id";

static const char *const CsiDefaultDebugNamePrefix = "__csi_unit_function_name_";

static const char *const CsiUnitSizeTableName = "__csi_unit_size_table";
static const char *const CsiUnitFedTableName = "__csi_unit_fed_table";
static const char *const CsiFuncIdVariablePrefix = "__csi_func_id_";
static const char *const CsiGlobalIdVariablePrefix = "__csi_global_id_";
static const char *const CsiUnitFedTableArrayName = "__csi_unit_fed_tables";
static const char *const CsiUnitSizeTableArrayName = "__csi_unit_size_tables";
static const char *const CsiInitCallsiteToFunctionName =
    "__csi_init_callsite_to_function";
static const char *const CsiDisableInstrumentationName =
    "__csi_disable_instrumentation";

using csi_id_t = int64_t;
static const csi_id_t CsiUnknownId = -1;
static const csi_id_t CsiCallsiteUnknownTargetId = CsiUnknownId;
// See llvm/tools/clang/lib/CodeGen/CodeGenModule.h:
static const int CsiUnitCtorPriority = 0;

/// Maintains a mapping from CSI ID to static data for that ID.
class ForensicTable {
public:
  ForensicTable() : BaseId(nullptr), IdCounter(0) {}
  ForensicTable(Module &M, StringRef BaseIdName,
                StringRef TableName = "");

  /// The number of entries in this forensic table
  uint64_t size() const { return IdCounter; }

  /// Return true if this FED has an ID associated with the given Value.
  bool hasId(const Value *V) const {
    return ValueToLocalIdMap.count(V);
  }

  /// Lookup the local ID of the given Value.
  csi_id_t lookupId(const Value *V) const {
    if (!hasId(V))
      return CsiUnknownId;
    return ValueToLocalIdMap.lookup(V);
  }

  /// Get the local ID of the given Value, creating an ID if necessary.
  csi_id_t getId(const Value *V);

  /// The GlobalVariable holding the base ID for this forensic table.
  GlobalVariable *baseId() const { return BaseId; }

  /// Converts a local to global ID conversion.
  ///
  /// This is done by using the given IRBuilder to insert a load to the base ID
  /// global variable followed by an add of the base value and the local ID.
  ///
  /// \returns A Value holding the global ID corresponding to the
  /// given local ID.
  Value *localToGlobalId(csi_id_t LocalId, IRBuilder<> &IRB) const;

  /// Helper function to get or create a string for a forensic-table entry.
  static Constant *getObjectStrGV(Module &M, StringRef Str, const Twine GVName);

protected:
  /// The GlobalVariable holding the base ID for this FED table.
  GlobalVariable *BaseId;
  /// Counter of local IDs used so far.
  csi_id_t IdCounter;
  /// Map of Value to Local ID.
  DenseMap<const Value *, csi_id_t> ValueToLocalIdMap;
  StringRef TableName;
};

/// Maintains a mapping from CSI ID to front-end data for that ID.
///
/// The front-end data currently is the source location that a given
/// CSI ID corresponds to.
class FrontEndDataTable : public ForensicTable {
public:
  FrontEndDataTable() : ForensicTable() {}
  FrontEndDataTable(Module &M, StringRef BaseIdName,
                    StringRef TableName = CsiUnitFedTableName,
                    StringRef DebugNamePrefix = CsiDefaultDebugNamePrefix)
      : ForensicTable(M, BaseIdName, TableName),
        DebugNamePrefix(DebugNamePrefix) {}

  /// The number of entries in this FED table
  uint64_t size() const { return LocalIdToSourceLocationMap.size(); }

  /// Add the given Function to this FED table.
  /// \returns The local ID of the Function.
  csi_id_t add(const Function &F);

  /// Add the given BasicBlock to this FED table.
  /// \returns The local ID of the BasicBlock.
  csi_id_t add(const BasicBlock &BB);

  /// Add the given Instruction to this FED table.
  /// \returns The local ID of the Instruction.
  csi_id_t add(const Instruction &I, const StringRef &RealName = "");

  /// Add the given Value to this FED table.
  /// \returns The local ID of the Value.
  csi_id_t add(Value &V);

  /// Add the given Global Value to this FED table.
  /// \returns The local ID of the Global.
  csi_id_t add(const GlobalValue &Val);

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
    int32_t Column;
    StringRef Filename;
    StringRef Directory;
  };
  StringRef DebugNamePrefix;

  /// Map of local ID to SourceLocation.
  DenseMap<csi_id_t, SourceLocation> LocalIdToSourceLocationMap;

  /// Create a struct type to match the "struct SourceLocation" type.
  /// (and the source_loc_t type in csi.h).
  static StructType *getSourceLocStructType(LLVMContext &C);

  /// Append the debug information to the table, assigning it the next
  /// available ID.
  ///
  /// \returns The local ID of the appended information.
  /// @{
  void add(csi_id_t ID, const DILocation *Loc, const StringRef &RealName = "");
  void add(csi_id_t ID, const DISubprogram *Subprog);
  /// @}

  /// Append the line and file information to the table, assigning it
  /// the next available ID.
  ///
  /// \returns The new local ID of the DILocation.
  void add(csi_id_t ID, int32_t Line = -1, int32_t Column = -1,
           StringRef Filename = "", StringRef Directory = "",
           StringRef Name = "");
};

/// Maintains a mapping from CSI ID of a basic block to the size of that basic
/// block in LLVM IR instructions.
class SizeTable : public ForensicTable {
public:
  SizeTable() : ForensicTable() {}
  SizeTable(Module &M, StringRef BaseIdName,
            StringRef TableName = CsiUnitSizeTableName)
      : ForensicTable(M, BaseIdName, TableName) {}

  /// The number of entries in this table
  uint64_t size() const { return LocalIdToSizeMap.size(); }

  /// Add the given basic block  to this table.
  /// \returns The local ID of the basic block.
  csi_id_t add(const BasicBlock &BB);

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
  struct SizeInformation {
    // This count includes every IR instruction.
    int32_t FullIRSize;
    // This count excludes IR instructions that don't lower to any real
    // instructions, e.g., PHI instructions, debug intrinsics, and lifetime
    // intrinsics.
    int32_t NonEmptyIRSize;
  };

  /// Map of local ID to size.
  DenseMap<csi_id_t, SizeInformation> LocalIdToSizeMap;

  /// Create a struct type to match the "struct SourceLocation" type.
  /// (and the source_loc_t type in csi.h).
  static StructType *getSizeStructType(LLVMContext &C);

  /// Append the size information to the table.
  void add(csi_id_t ID, int32_t FullIRSize = 0, int32_t NonEmptyIRSize = 0);
};

/// Represents a property value passed to hooks.
class CsiProperty {
public:
  CsiProperty() {}

  virtual ~CsiProperty() {}

  /// Return the coerced type of a property.
  ///
  /// TODO: Right now, this function simply returns a 64-bit integer.  Although
  /// this solution works for x86_64, it should be generalized to handle other
  /// architectures in the future.
  static Type *getCoercedType(LLVMContext &C, StructType *Ty) {
    // Must match the definition of property type in csi.h
    // return StructType::get(IntegerType::get(C, 64),
    //                        nullptr);
    // We return an integer type, rather than a struct type, to deal with x86_64
    // type coercion on struct bit fields.
    return IntegerType::get(C, 64);
  }

  /// Return a constant value holding this property.
  virtual Constant *getValueImpl(LLVMContext &C) const = 0;

  Constant *getValue(IRBuilder<> &IRB) const {
    return getValueImpl(IRB.getContext());
  }
};

class CsiFuncProperty : public CsiProperty {
public:
  CsiFuncProperty() { PropValue.Bits = 0; }

  /// Return the Type of a property.
  static StructType *getStructType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return StructType::get(IntegerType::get(C, PropBits.MaySpawn),
                           IntegerType::get(C, PropBits.Padding));
  }
  static Type *getType(LLVMContext &C) {
    return getCoercedType(C, getStructType(C));
  }

  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // StructType *StructTy = getType(C);
    // return ConstantStruct::get(StructTy,
    //                            ConstantInt::get(IntegerType::get(C, 64), 0),
    //                            nullptr);
    // TODO: This solution works for x86, but should be generalized to support
    // other architectures in the future.
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the MaySpawn property.
  void setMaySpawn(bool v) { PropValue.Fields.MaySpawn = v; }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned MaySpawn : 1;
      uint64_t Padding : 63;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int MaySpawn;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = {1, (64 - 1)};
};

class CsiFuncExitProperty : public CsiProperty {
public:
  CsiFuncExitProperty() { PropValue.Bits = 0; }

  /// Return the Type of a property.
  static StructType *getStructType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return StructType::get(IntegerType::get(C, PropBits.MaySpawn),
                           IntegerType::get(C, PropBits.EHReturn),
                           IntegerType::get(C, PropBits.Padding));
  }
  static Type *getType(LLVMContext &C) {
    return getCoercedType(C, getStructType(C));
  }

  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // StructType *StructTy = getType(C);
    // return ConstantStruct::get(StructTy,
    //                            ConstantInt::get(IntegerType::get(C, 64), 0),
    //                            nullptr);
    // TODO: This solution works for x86, but should be generalized to support
    // other architectures in the future.
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the MaySpawn property.
  void setMaySpawn(bool v) { PropValue.Fields.MaySpawn = v; }

  /// Set the value of the EHReturn property.
  void setEHReturn(bool v) { PropValue.Fields.EHReturn = v; }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned MaySpawn : 1;
      unsigned EHReturn : 1;
      uint64_t Padding : 62;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int MaySpawn;
    int EHReturn;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = {1, 1, (64 - 1 - 1)};
};

class CsiBBProperty : public CsiProperty {
public:
  CsiBBProperty() { PropValue.Bits = 0; }

  /// Return the Type of a property.
  static StructType *getStructType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return StructType::get(IntegerType::get(C, PropBits.IsLandingPad),
                           IntegerType::get(C, PropBits.IsEHPad),
                           IntegerType::get(C, PropBits.Padding));
  }
  static Type *getType(LLVMContext &C) {
    return getCoercedType(C, getStructType(C));
  }

  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // StructType *StructTy = getType(C);
    // return ConstantStruct::get(StructTy,
    //                            ConstantInt::get(IntegerType::get(C, 64), 0),
    //                            nullptr);
    // TODO: This solution works for x86, but should be generalized to support
    // other architectures in the future.
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the IsLandingPad property.
  void setIsLandingPad(bool v) { PropValue.Fields.IsLandingPad = v; }

  /// Set the value of the IsEHPad property.
  void setIsEHPad(bool v) { PropValue.Fields.IsEHPad = v; }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned IsLandingPad : 1;
      unsigned IsEHPad : 1;
      uint64_t Padding : 62;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int IsLandingPad;
    int IsEHPad;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = {1, 1, (64 - 1 - 1)};
};

class CsiCallProperty : public CsiProperty {
public:
  CsiCallProperty() { PropValue.Bits = 0; }

  /// Return the Type of a property.
  static StructType *getStructType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return StructType::get(IntegerType::get(C, PropBits.IsIndirect),
                           IntegerType::get(C, PropBits.HasOneUse),
                           IntegerType::get(C, PropBits.Padding));
  }
  static Type *getType(LLVMContext &C) {
    return getCoercedType(C, getStructType(C));
  }

  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // StructType *StructTy = getType(C);
    // return ConstantStruct::get(
    //     StructTy,
    //     ConstantInt::get(IntegerType::get(C, PropBits.IsIndirect),
    //                      PropValue.IsIndirect),
    //     ConstantInt::get(IntegerType::get(C, PropBits.Padding), 0),
    //     nullptr);
    // TODO: This solution works for x86, but should be generalized to support
    // other architectures in the future.
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the IsIndirect property.
  void setIsIndirect(bool v) { PropValue.Fields.IsIndirect = v; }
  /// Set the value of the HasOneUse property.
  void setHasOneUse(bool v) { PropValue.Fields.HasOneUse = v; }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned IsIndirect : 1;
      unsigned HasOneUse : 1;
      uint64_t Padding : 62;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int IsIndirect;
    int HasOneUse;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = {1, 1, (64 - 1 - 1)};
};

class CsiLoadStoreProperty : public CsiProperty {
public:
  CsiLoadStoreProperty() { PropValue.Bits = 0; }
  /// Return the Type of a property.
  static StructType *getStructType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return StructType::get(
        IntegerType::get(C, PropBits.Alignment),
        IntegerType::get(C, PropBits.IsVtableAccess),
        IntegerType::get(C, PropBits.IsConstant),
        IntegerType::get(C, PropBits.IsOnStack),
        IntegerType::get(C, PropBits.MayBeCaptured),
        IntegerType::get(C, PropBits.IsVolatile),
        IntegerType::get(C, PropBits.LoadReadBeforeWriteInBB),
        IntegerType::get(C, PropBits.HasOneUse),
        IntegerType::get(C, PropBits.Padding));
  }
  static Type *getType(LLVMContext &C) {
    return getCoercedType(C, getStructType(C));
  }

  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // return ConstantStruct::get(
    //     StructTy,
    //     ConstantInt::get(IntegerType::get(C, PropBits.Alignment),
    //                      PropValue.Alignment),
    //     ConstantInt::get(IntegerType::get(C, PropBits.IsVtableAccess),
    //                      PropValue.IsVtableAccess),
    //     ConstantInt::get(IntegerType::get(C, PropBits.IsConstant),
    //                      PropValue.IsVtableAccess),
    //     ConstantInt::get(IntegerType::get(C, PropBits.IsOnStack),
    //                      PropValue.IsVtableAccess),
    //     ConstantInt::get(IntegerType::get(C, PropBits.MayBeCaptured),
    //                      PropValue.IsVtableAccess),
    //     ConstantInt::get(IntegerType::get(C,
    //     PropBits.LoadReadBeforeWriteInBB),
    //                      PropValue.LoadReadBeforeWriteInBB),
    //     ConstantInt::get(IntegerType::get(C, PropBits.Padding), 0),
    //     nullptr);
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the Alignment property.
  void setAlignment(char v) { PropValue.Fields.Alignment = v; }
  /// Set the value of the IsVtableAccess property.
  void setIsVtableAccess(bool v) { PropValue.Fields.IsVtableAccess = v; }
  /// Set the value of the IsConstant property.
  void setIsConstant(bool v) { PropValue.Fields.IsConstant = v; }
  /// Set the value of the IsOnStack property.
  void setIsOnStack(bool v) { PropValue.Fields.IsOnStack = v; }
  /// Set the value of the MayBeCaptured property.
  void setMayBeCaptured(bool v) { PropValue.Fields.MayBeCaptured = v; }
  /// Set the value of the IsVolatile property.
  void setIsVolatile(bool v) { PropValue.Fields.IsVolatile = v; }
  /// Set the value of the HasOneUse property.
  void setHasOneUse(bool v) { PropValue.Fields.HasOneUse = v; }
  /// Set the value of the LoadReadBeforeWriteInBB property.
  void setLoadReadBeforeWriteInBB(bool v) {
    PropValue.Fields.LoadReadBeforeWriteInBB = v;
  }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned Alignment : 8;
      unsigned IsVtableAccess : 1;
      unsigned IsConstant : 1;
      unsigned IsOnStack : 1;
      unsigned MayBeCaptured : 1;
      unsigned IsVolatile : 1;
      unsigned LoadReadBeforeWriteInBB : 1;
      unsigned HasOneUse : 1;
      uint64_t Padding : 49;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int Alignment;
    int IsVtableAccess;
    int IsConstant;
    int IsOnStack;
    int MayBeCaptured;
    int IsVolatile;
    int LoadReadBeforeWriteInBB;
    int HasOneUse;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = {
      8, 1, 1, 1, 1, 1, 1, 1, (64 - 8 - 1 - 1 - 1 - 1 - 1 - 1 - 1)};
};

class CsiAllocaProperty : public CsiProperty {
public:
  CsiAllocaProperty() { PropValue.Bits = 0; }

  /// Return the Type of a property.
  static StructType *getStructType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return StructType::get(IntegerType::get(C, PropBits.IsStatic),
                           IntegerType::get(C, PropBits.MayBeCaptured),
                           IntegerType::get(C, PropBits.Padding));
  }
  static Type *getType(LLVMContext &C) {
    return getCoercedType(C, getStructType(C));
  }

  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // TODO: This solution works for x86, but should be generalized to support
    // other architectures in the future.
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the IsIndirect property.
  void setIsStatic(bool v) { PropValue.Fields.IsStatic = v; }
  /// Set the value of the MayBeCaptured property.
  void setMayBeCaptured(bool v) { PropValue.Fields.MayBeCaptured = v; }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned IsStatic : 1;
      unsigned MayBeCaptured : 1;
      uint64_t Padding : 62;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int IsStatic;
    int MayBeCaptured;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = {1, 1, (64 - 1 - 1)};
};

class CsiAllocFnProperty : public CsiProperty {
public:
  CsiAllocFnProperty() { PropValue.Bits = 0; }
  /// Return the Type of a property.
  static StructType *getStructType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return StructType::get(IntegerType::get(C, PropBits.AllocFnTy),
                           IntegerType::get(C, PropBits.MayBeCaptured),
                           IntegerType::get(C, PropBits.Padding));
  }
  static Type *getType(LLVMContext &C) {
    return getCoercedType(C, getStructType(C));
  }

  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csan.h
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the allocation function type (e.g., malloc, calloc, new).
  void setAllocFnTy(unsigned v) { PropValue.Fields.AllocFnTy = v; }
  /// Set the value of the MayBeCaptured property.
  void setMayBeCaptured(bool v) { PropValue.Fields.MayBeCaptured = v; }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned AllocFnTy : 8;
      unsigned MayBeCaptured : 1;
      uint64_t Padding : 55;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int AllocFnTy;
    int MayBeCaptured;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = {8, 1, (64 - 8 - 1)};
};

class CsiFreeProperty : public CsiProperty {
public:
  CsiFreeProperty() { PropValue.Bits = 0; }
  /// Return the Type of a property.
  static StructType *getStructType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return StructType::get(IntegerType::get(C, PropBits.FreeTy),
                           IntegerType::get(C, PropBits.Padding));
  }
  static Type *getType(LLVMContext &C) {
    return getCoercedType(C, getStructType(C));
  }

  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csan.h
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the allocation function type (e.g., malloc, calloc, new).
  void setFreeTy(unsigned v) { PropValue.Fields.FreeTy = v; }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned FreeTy : 8;
      uint64_t Padding : 56;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int FreeTy;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = {8, (64 - 8)};
};

class CsiArithmeticFlags : public CsiProperty {
public:
  CsiArithmeticFlags() { PropValue.Bits = 0; }

  /// Return the Type of a property.
  static StructType *getStructType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return StructType::get(IntegerType::get(C, PropBits.NoSignedWrap),
                           IntegerType::get(C, PropBits.NoUnsignedWrap),
                           IntegerType::get(C, PropBits.IsExact),
                           IntegerType::get(C, PropBits.NoNaNs),
                           IntegerType::get(C, PropBits.NoInfs),
                           IntegerType::get(C, PropBits.NoSignedZeros),
                           IntegerType::get(C, PropBits.AllowReciprocal),
                           IntegerType::get(C, PropBits.AllowContract),
                           IntegerType::get(C, PropBits.ApproxFunc),
                           IntegerType::get(C, PropBits.IsInBounds),
                           IntegerType::get(C, PropBits.HasOneUse),
                           IntegerType::get(C, PropBits.Padding));
  }
  static Type *getType(LLVMContext &C) {
    return getCoercedType(C, getStructType(C));
  }

  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // TODO: This solution works for x86, but should be generalized to support
    // other architectures in the future.
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  // Set the property based on the IR flags of the given Value.
  void copyIRFlags(const Value *V) {
    if (auto *OB = dyn_cast<OverflowingBinaryOperator>(V)) {
      PropValue.Fields.NoSignedWrap = OB->hasNoSignedWrap();
      PropValue.Fields.NoUnsignedWrap = OB->hasNoUnsignedWrap();
    }

    if (auto *PE = dyn_cast<PossiblyExactOperator>(V)) {
      PropValue.Fields.IsExact = PE->isExact();
    }

    if (auto *FP = dyn_cast<FPMathOperator>(V)) {
      PropValue.Fields.AllowReassoc = FP->hasAllowReassoc();
      PropValue.Fields.NoNaNs = FP->hasNoNaNs();
      PropValue.Fields.NoInfs = FP->hasNoInfs();
      PropValue.Fields.NoSignedZeros = FP->hasNoSignedZeros();
      PropValue.Fields.AllowReciprocal = FP->hasAllowReciprocal();
      PropValue.Fields.ApproxFunc = FP->hasApproxFunc();
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
      PropValue.Fields.IsInBounds = GEP->isInBounds();
    }
  }
  /// Set the value of the HasOneUse property.
  void setHasOneUse(bool v) { PropValue.Fields.HasOneUse = v; }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned NoSignedWrap : 1;
      unsigned NoUnsignedWrap : 1;
      unsigned IsExact : 1;
      unsigned AllowReassoc : 1;
      unsigned NoNaNs : 1;
      unsigned NoInfs : 1;
      unsigned NoSignedZeros : 1;
      unsigned AllowReciprocal : 1;
      unsigned AllowContract : 1;
      unsigned ApproxFunc : 1;
      unsigned IsInBounds : 1;
      unsigned HasOneUse : 1;
      uint64_t Padding : 52;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int NoSignedWrap;
    int NoUnsignedWrap;
    int IsExact;
    int AllowReassoc;
    int NoNaNs;
    int NoInfs;
    int NoSignedZeros;
    int AllowReciprocal;
    int AllowContract;
    int ApproxFunc;
    int IsInBounds;
    int HasOneUse;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits =
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     (64 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1)};
};

struct CSIImpl {
public:
  CSIImpl(Module &M, CallGraph *CG,
          function_ref<DominatorTree &(Function &)> GetDomTree,
          function_ref<LoopInfo &(Function &)> GetLoopInfo,
          function_ref<TaskInfo &(Function &)> GetTaskInfo,
          const TargetLibraryInfo *TLI,
          const CSIOptions &Options = CSIOptions())
      : M(M), DL(M.getDataLayout()), CG(CG), GetDomTree(GetDomTree),
        GetLoopInfo(GetLoopInfo), GetTaskInfo(GetTaskInfo), TLI(TLI),
        Options(Options) {
    loadConfiguration();
  }

  virtual ~CSIImpl() {}

  bool run();

  /// Get the number of bytes accessed via the given address.
  static int getNumBytesAccessed(Value *Addr, const DataLayout &DL);

  /// Members to extract properties of loads/stores.
  static bool isVtableAccess(Instruction *I);
  static bool addrPointsToConstantData(Value *Addr);
  static bool isAtomic(Instruction *I);
  static void getAllocFnArgs(const Instruction *I,
                             SmallVectorImpl<Value *> &AllocFnArgs,
                             Type *SizeTy, Type *AddrTy,
                             const TargetLibraryInfo &TLI);

  /// Helper functions to deal with calls to functions that can throw.
  static void setupCalls(Function &F);
  static void setupBlocks(Function &F, const TargetLibraryInfo *TLI,
                          DominatorTree *DT = nullptr);

  /// Helper function that identifies calls or invokes of placeholder functions,
  /// such as debug-info intrinsics or lifetime intrinsics.
  static bool callsPlaceholderFunction(const Instruction &I);

  static Constant *getDefaultID(IRBuilder<> &IRB) {
    return IRB.getInt64(CsiUnknownId);
  }

protected:
  /// Initialize the CSI pass.
  void initializeCsi();
  /// Finalize the CSI pass.
  void finalizeCsi();

  /// Initialize llvm::Functions for the CSI hooks.
  /// @{
  void initializeLoadStoreHooks();
  void initializeFuncHooks();
  void initializeBasicBlockHooks();
  void initializeCallsiteHooks();
  void initializeAllocaHooks();
  void initializeMemIntrinsicsHooks();
  void initializeTapirHooks();
  void initializeAllocFnHooks();
  void initializeArithmeticHooks();
  /// @}

  static StructType *getUnitFedTableType(LLVMContext &C,
                                         PointerType *EntryPointerType);
  static Constant *fedTableToUnitFedTable(Module &M,
                                          StructType *UnitFedTableType,
                                          FrontEndDataTable &FedTable);
  static StructType *getUnitSizeTableType(LLVMContext &C,
                                          PointerType *EntryPointerType);
  static Constant *sizeTableToUnitSizeTable(Module &M,
                                            StructType *UnitSizeTableType,
                                            SizeTable &SzTable);
  /// Initialize the front-end data table structures.
  void initializeFEDTables();
  /// Collect unit front-end data table structures for finalization.
  void collectUnitFEDTables();
  /// Initialize the size table structures.
  void initializeSizeTables();
  /// Collect unit size table structures for finalization.
  void collectUnitSizeTables();

  virtual CallInst *createRTUnitInitCall(IRBuilder<> &IRB);

  // Get the local ID of the given function.
  csi_id_t getLocalFunctionID(Function &F);
  /// Generate a function that stores global function IDs into a set
  /// of externally-visible global variables.
  void generateInitCallsiteToFunction();

  /// Compute CSI properties on the given ordered list of loads and stores.
  void computeLoadAndStoreProperties(
      SmallVectorImpl<std::pair<Instruction *, CsiLoadStoreProperty>>
          &LoadAndStoreProperties,
      SmallVectorImpl<Instruction *> &BBLoadsAndStores, const DataLayout &DL,
      LoopInfo &LI);

  /// Insert calls to the instrumentation hooks.
  /// @{
  void addLoadStoreInstrumentation(Instruction *I, Function *BeforeFn,
                                   Function *AfterFn, Value *CsiId,
                                   Type *AddrType, Value *Addr, int NumBytes,
                                   Value *StoreValCat, Value *StoreValID,
                                   CsiLoadStoreProperty &Prop);
  void assignLoadOrStoreID(Instruction *I);
  void instrumentLoadOrStore(Instruction *I, CsiLoadStoreProperty &Prop,
                             const DataLayout &DL);
  void instrumentVectorMemBuiltin(Instruction *I);
  void assignAtomicID(Instruction *I);
  void instrumentAtomic(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(Instruction *I);
  void assignCallsiteID(Instruction *I);
  bool handleFPBuiltinCall(CallInst *I, Function *F, LoopInfo &LI);
  void instrumentCallsite(Instruction *I, DominatorTree *DT, LoopInfo &LI);
  void instrumentBasicBlock(BasicBlock &BB);

  void instrumentDetach(DetachInst *DI, DominatorTree *DT, TaskInfo &TI,
                        const DenseMap<Value *, Value *> &TrackVars);
  void instrumentSync(SyncInst *SI,
                      const DenseMap<Value *, Value *> &TrackVars);
  void assignAllocaID(Instruction *I);
  void instrumentAlloca(Instruction *I);
  void assignAllocFnID(Instruction *I);
  void instrumentAllocFn(Instruction *I, DominatorTree *DT);
  void instrumentFree(Instruction *I);
  void assignArithmeticID(Instruction *I);
  void instrumentArithmetic(Instruction *I, LoopInfo &LI);

  void interposeCall(Instruction *I);

  void instrumentFunction(Function &F);
  /// @}

  DenseMap<Value *, Value *>
  keepTrackOfSpawns(Function &F,
                    const SmallVectorImpl<DetachInst *> &Detaches,
                    const SmallVectorImpl<SyncInst *> &Syncs);

  /// Obtain the signature for the interposition function given the
  /// original function that needs interpositioning.
  Function *getInterpositionFunction(Function *F);

  /// Insert a call to the given hook function before the given instruction.
  CallInst* insertHookCall(Instruction *I, Function *HookFunction,
                      ArrayRef<Value *> HookArgs);
  bool updateArgPHIs(Instruction *I, BasicBlock *Succ, BasicBlock *BB,
                     ArrayRef<Value *> HookArgs,
                     ArrayRef<Value *> DefaultHookArgs);
  CallInst *insertHookCallInSuccessorBB(Instruction *I, BasicBlock *Succ,
                                        BasicBlock *BB, Function *HookFunction,
                                        ArrayRef<Value *> HookArgs,
                                        ArrayRef<Value *> DefaultHookArgs);
  void insertHookCallAtSharedEHSpindleExits(Instruction *I,
                                            Spindle *SharedEHSpindle, Task *T,
                                            Function *HookFunction,
                                            FrontEndDataTable &FED,
                                            ArrayRef<Value *> HookArgs,
                                            ArrayRef<Value *> DefaultArgs);

  /// Return true if the given function should not be instrumented.
  bool shouldNotInstrumentFunction(Function &F);

  // Update the attributes on the instrumented function that might be
  // invalidated by the inserted instrumentation.
  void updateInstrumentedFnAttrs(Function &F);
  // List of all allocation function types.  This list needs to remain
  // consistent with TargetLibraryInfo and with csan.h.
  enum class AllocFnTy {
    malloc = 0,
    valloc,
    calloc,
    realloc,
    reallocf,
    Znwj,
    ZnwjRKSt9nothrow_t,
    Znwm,
    ZnwmRKSt9nothrow_t,
    Znaj,
    ZnajRKSt9nothrow_t,
    Znam,
    ZnamRKSt9nothrow_t,
    msvc_new_int,
    msvc_new_int_nothrow,
    msvc_new_longlong,
    msvc_new_longlong_nothrow,
    msvc_new_array_int,
    msvc_new_array_int_nothrow,
    msvc_new_array_longlong,
    msvc_new_array_longlong_nothrow,
    LAST_ALLOCFNTY
  };

  static AllocFnTy getAllocFnTy(const LibFunc &F) {
    switch (F) {
    default:
      return AllocFnTy::LAST_ALLOCFNTY;
    case LibFunc_malloc:
      return AllocFnTy::malloc;
    case LibFunc_valloc:
      return AllocFnTy::valloc;
    case LibFunc_calloc:
      return AllocFnTy::calloc;
    case LibFunc_realloc:
      return AllocFnTy::realloc;
    case LibFunc_reallocf:
      return AllocFnTy::reallocf;
    case LibFunc_Znwj:
      return AllocFnTy::Znwj;
    case LibFunc_ZnwjRKSt9nothrow_t:
      return AllocFnTy::ZnwjRKSt9nothrow_t;
    case LibFunc_Znwm:
      return AllocFnTy::Znwm;
    case LibFunc_ZnwmRKSt9nothrow_t:
      return AllocFnTy::ZnwmRKSt9nothrow_t;
    case LibFunc_Znaj:
      return AllocFnTy::Znaj;
    case LibFunc_ZnajRKSt9nothrow_t:
      return AllocFnTy::ZnajRKSt9nothrow_t;
    case LibFunc_Znam:
      return AllocFnTy::Znam;
    case LibFunc_ZnamRKSt9nothrow_t:
      return AllocFnTy::ZnamRKSt9nothrow_t;
    case LibFunc_msvc_new_int:
      return AllocFnTy::msvc_new_int;
    case LibFunc_msvc_new_int_nothrow:
      return AllocFnTy::msvc_new_int_nothrow;
    case LibFunc_msvc_new_longlong:
      return AllocFnTy::msvc_new_longlong;
    case LibFunc_msvc_new_longlong_nothrow:
      return AllocFnTy::msvc_new_longlong_nothrow;
    case LibFunc_msvc_new_array_int:
      return AllocFnTy::msvc_new_array_int;
    case LibFunc_msvc_new_array_int_nothrow:
      return AllocFnTy::msvc_new_array_int_nothrow;
    case LibFunc_msvc_new_array_longlong:
      return AllocFnTy::msvc_new_array_longlong;
    case LibFunc_msvc_new_array_longlong_nothrow:
      return AllocFnTy::msvc_new_array_longlong_nothrow;
    }
  }

  // List of all free function types.  This list needs to remain consistent with
  // TargetLibraryInfo and with csi.h.
  enum class FreeTy {
    free = 0,
    ZdlPv,
    ZdlPvRKSt9nothrow_t,
    ZdlPvj,
    ZdlPvm,
    ZdaPv,
    ZdaPvRKSt9nothrow_t,
    ZdaPvj,
    ZdaPvm,
    msvc_delete_ptr32,
    msvc_delete_ptr32_nothrow,
    msvc_delete_ptr32_int,
    msvc_delete_ptr64,
    msvc_delete_ptr64_nothrow,
    msvc_delete_ptr64_longlong,
    msvc_delete_array_ptr32,
    msvc_delete_array_ptr32_nothrow,
    msvc_delete_array_ptr32_int,
    msvc_delete_array_ptr64,
    msvc_delete_array_ptr64_nothrow,
    msvc_delete_array_ptr64_longlong,
    LAST_FREETY
  };

  static FreeTy getFreeTy(const LibFunc &F) {
    switch (F) {
    default:
      return FreeTy::LAST_FREETY;
    case LibFunc_free:
      return FreeTy::free;
    case LibFunc_ZdlPv:
      return FreeTy::ZdlPv;
    case LibFunc_ZdlPvRKSt9nothrow_t:
      return FreeTy::ZdlPvRKSt9nothrow_t;
    case LibFunc_ZdlPvj:
      return FreeTy::ZdlPvj;
    case LibFunc_ZdlPvm:
      return FreeTy::ZdlPvm;
    case LibFunc_ZdaPv:
      return FreeTy::ZdaPv;
    case LibFunc_ZdaPvRKSt9nothrow_t:
      return FreeTy::ZdaPvRKSt9nothrow_t;
    case LibFunc_ZdaPvj:
      return FreeTy::ZdaPvj;
    case LibFunc_ZdaPvm:
      return FreeTy::ZdaPvm;
    case LibFunc_msvc_delete_ptr32:
      return FreeTy::msvc_delete_ptr32;
    case LibFunc_msvc_delete_ptr32_nothrow:
      return FreeTy::msvc_delete_ptr32_nothrow;
    case LibFunc_msvc_delete_ptr32_int:
      return FreeTy::msvc_delete_ptr32_int;
    case LibFunc_msvc_delete_ptr64:
      return FreeTy::msvc_delete_ptr64;
    case LibFunc_msvc_delete_ptr64_nothrow:
      return FreeTy::msvc_delete_ptr64_nothrow;
    case LibFunc_msvc_delete_ptr64_longlong:
      return FreeTy::msvc_delete_ptr64_longlong;
    case LibFunc_msvc_delete_array_ptr32:
      return FreeTy::msvc_delete_array_ptr32;
    case LibFunc_msvc_delete_array_ptr32_nothrow:
      return FreeTy::msvc_delete_array_ptr32_nothrow;
    case LibFunc_msvc_delete_array_ptr32_int:
      return FreeTy::msvc_delete_array_ptr32_int;
    case LibFunc_msvc_delete_array_ptr64:
      return FreeTy::msvc_delete_array_ptr64;
    case LibFunc_msvc_delete_array_ptr64_nothrow:
      return FreeTy::msvc_delete_array_ptr64_nothrow;
    case LibFunc_msvc_delete_array_ptr64_longlong:
      return FreeTy::msvc_delete_array_ptr64_longlong;
    }
  }
  enum class CSIOpcode
    {
     Add = 0,
     FAdd,
     Sub,
     FSub,
     Mul,
     FMul,
     UDiv,
     SDiv,
     FDiv,
     URem,
     SRem,
     FRem,
     Shl,
     LShr,
     AShr,
     And,
     Or,
     Xor,
     LAST_CSIOpcode
    };
  Value *getOpcodeID(unsigned Opcode, IRBuilder<> &IRB) const {
    switch(Opcode) {
    case Instruction::Add:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::Add));
    case Instruction::FAdd:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::FAdd));
    case Instruction::Sub:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::Sub));
    case Instruction::FSub:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::FSub));
    case Instruction::Mul:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::Mul));
    case Instruction::FMul:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::FMul));
    case Instruction::UDiv:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::UDiv));
    case Instruction::SDiv:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::SDiv));
    case Instruction::FDiv:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::FDiv));
    case Instruction::URem:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::URem));
    case Instruction::SRem:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::SRem));
    case Instruction::FRem:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::FRem));
    case Instruction::Shl:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::Shl));
    case Instruction::LShr:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::LShr));
    case Instruction::AShr:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::AShr));
    case Instruction::And:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::And));
    case Instruction::Or:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::Or));
    case Instruction::Xor:
      return IRB.getInt8(static_cast<unsigned>(CSIOpcode::Xor));
    default:
      llvm_unreachable("Invalid opcode");
      break;
    }
  }

  enum class CSIOperandCategory
    {
     None = 0,
     Constant,
     Parameter,
     Global,
     // Relevant types of IR objects to appear as arguments
     Callsite,
     Load,
     Alloca,
     AllocFn,
     Arithmetic,
     LAST_CSIOperandCategory
    };
  std::pair<Value *, Value *> getOperandID(const Value *Operand,
                                           IRBuilder<> &IRB) const {
    std::pair<Value *, Value *> OperandID;
    if (!Operand) {
      OperandID.first = IRB.getInt8(
          static_cast<unsigned>(CSIOperandCategory::None));
      OperandID.second = getDefaultID(IRB);
    } else if (isa<Constant>(Operand)) {
      OperandID.first = IRB.getInt8(
          static_cast<unsigned>(CSIOperandCategory::Constant));
      OperandID.second = getDefaultID(IRB);
    } else if (isa<Argument>(Operand)) {
      OperandID.first = IRB.getInt8(
          static_cast<unsigned>(CSIOperandCategory::Parameter));
      OperandID.second = ParameterFED.localToGlobalId(
          ParameterFED.lookupId(Operand), IRB);
    } else if (auto *GV = dyn_cast<GlobalValue>(Operand)) {
      OperandID.first = IRB.getInt8(
          static_cast<unsigned>(CSIOperandCategory::Global));
      // Because we're referencing this global variable in the
      // program-under-test, we must ensure that the CSI global storing the CSI
      // ID of the program's global is available.  Create the CSI global in this
      // module if necessary.
      std::string GVName = CsiGlobalIdVariablePrefix + GV->getName().str();
      GlobalVariable *GlobIdGV = dyn_cast<GlobalVariable>(
          M.getOrInsertGlobal(GVName, IRB.getInt64Ty()));
      GlobIdGV->setConstant(false);
      GlobIdGV->setLinkage(GlobalValue::WeakAnyLinkage);
      GlobIdGV->setInitializer(getDefaultID(IRB));
      OperandID.second = IRB.CreateLoad(GlobIdGV);
    } else if (isa<CallInst>(Operand) || isa<InvokeInst>(Operand)) {
      OperandID.first = IRB.getInt8(
          static_cast<unsigned>(CSIOperandCategory::Callsite));
      OperandID.second = CallsiteFED.localToGlobalId(
          CallsiteFED.lookupId(Operand),IRB);
    } else if (isa<LoadInst>(Operand)) {
      OperandID.first = IRB.getInt8(
          static_cast<unsigned>(CSIOperandCategory::Load));
      OperandID.second = LoadFED.localToGlobalId(LoadFED.lookupId(Operand),
                                                 IRB);
    } else if (isa<AllocaInst>(Operand)) {
      OperandID.first = IRB.getInt8(
          static_cast<unsigned>(CSIOperandCategory::Alloca));
      OperandID.second = AllocaFED.localToGlobalId(AllocaFED.lookupId(Operand),
                                                   IRB);
    } else if (isAllocationFn(Operand, TLI)) {
      OperandID.first = IRB.getInt8(
          static_cast<unsigned>(CSIOperandCategory::AllocFn));
      OperandID.second = AllocFnFED.localToGlobalId(
          AllocFnFED.lookupId(Operand), IRB);
    } else {
      OperandID.first = IRB.getInt8(
          static_cast<unsigned>(CSIOperandCategory::Arithmetic));
      OperandID.second = ArithmeticFED.localToGlobalId(
          ArithmeticFED.lookupId(Operand), IRB);
    }
    return OperandID;
  }

  // TODO: Determine how to handle the "finite" variants of some operations.
  enum class CSIBuiltinFuncOp
    {
     F_Fma = 0,
     F_Sqrt,
     F_Cbrt,
     F_Sin,
     F_Cos,
     F_Tan,
     F_SinPi,
     F_CosPi,
     F_SinCosPi,
     F_ASin,
     F_ACos,
     F_ATan,
     F_ATan2,
     F_Sinh,
     F_Cosh,
     F_Tanh,
     F_ASinh,
     F_ACosh,
     F_ATanh,
     F_Log,
     F_Log2,
     F_Log10,
     F_Logb,
     F_Log1p,
     F_Exp,
     F_Exp2,
     F_Exp10,
     F_Expm1,
     F_Fabs,
     F_Floor,
     F_Ceil,
     F_Fmod,
     F_Trunc,
     F_Rint,
     F_NearbyInt,
     F_Round,
     F_Canonicalize,
     F_Pow,
     F_CopySign,
     F_MinNum,  // Same as FMin
     F_MaxNum,  // Same as FMax
     F_Ldexp,
     LAST_CSIBuiltinFuncOp
    };
  CSIBuiltinFuncOp getBuiltinFuncOp(CallSite &CS);

  void linkInToolFromBitcode(const std::string &bitcodePath);
  void loadConfiguration();

  Module &M;
  const DataLayout &DL;
  CallGraph *CG;
  function_ref<DominatorTree &(Function &)> GetDomTree;
  function_ref<LoopInfo &(Function &)> GetLoopInfo;
  function_ref<TaskInfo &(Function &)> GetTaskInfo;
  const TargetLibraryInfo *TLI;
  CSIOptions Options;

  FrontEndDataTable FunctionFED, FunctionExitFED, BasicBlockFED, CallsiteFED,
      LoadFED, StoreFED, AllocaFED, DetachFED, TaskFED, TaskExitFED,
      DetachContinueFED, SyncFED, AllocFnFED, FreeFED, ArithmeticFED,
      ParameterFED, GlobalFED;

  SmallVector<Constant *, 17> UnitFedTables;

  SizeTable BBSize;
  SmallVector<Constant *, 1> UnitSizeTables;

  // Instrumentation hooks
  Function *CsiFuncEntry = nullptr, *CsiFuncExit = nullptr;
  Function *CsiBBEntry = nullptr, *CsiBBExit = nullptr;
  Function *CsiBeforeCallsite = nullptr, *CsiAfterCallsite = nullptr;
  Function *CsiBeforeRead = nullptr, *CsiAfterRead = nullptr;
  Function *CsiBeforeWrite = nullptr, *CsiAfterWrite = nullptr;
  Function *CsiBeforeAlloca = nullptr, *CsiAfterAlloca = nullptr;
  Function *CsiDetach = nullptr, *CsiDetachContinue = nullptr;
  Function *CsiTaskEntry = nullptr, *CsiTaskExit = nullptr;
  Function *CsiBeforeSync = nullptr, *CsiAfterSync = nullptr;
  Function *CsiBeforeAllocFn = nullptr, *CsiAfterAllocFn = nullptr;
  Function *CsiBeforeFree = nullptr, *CsiAfterFree = nullptr;

  // Function *CsiBeforeArithmeticH = nullptr, *CsiAfterArithmeticH = nullptr;
  Function *CsiBeforeArithmeticF = nullptr, *CsiAfterArithmeticF = nullptr;
  Function *CsiBeforeArithmeticD = nullptr, *CsiAfterArithmeticD = nullptr;
  Function *CsiBeforeArithmeticI8 = nullptr, *CsiAfterArithmeticI8 = nullptr;
  Function *CsiBeforeArithmeticI16 = nullptr, *CsiAfterArithmeticI16 = nullptr;
  Function *CsiBeforeArithmeticI32 = nullptr, *CsiAfterArithmeticI32 = nullptr;
  Function *CsiBeforeArithmeticI64 = nullptr, *CsiAfterArithmeticI64 = nullptr;
  Function *CsiBeforeArithmeticI128 = nullptr,
    *CsiAfterArithmeticI128 = nullptr;

  Function *CsiBeforeArithmetic4F = nullptr, *CsiAfterArithmetic4F = nullptr;
  Function *CsiBeforeArithmetic8F = nullptr, *CsiAfterArithmetic8F = nullptr;
  Function *CsiBeforeArithmetic16F = nullptr, *CsiAfterArithmetic16F = nullptr;
  Function *CsiBeforeArithmetic2D = nullptr, *CsiAfterArithmetic2D = nullptr;
  Function *CsiBeforeArithmetic4D = nullptr, *CsiAfterArithmetic4D = nullptr;
  Function *CsiBeforeArithmetic8D = nullptr, *CsiAfterArithmetic8D = nullptr;

  // Function *CsiBeforeExtendHF = nullptr, *CsiAfterExtendHF = nullptr;
  // Function *CsiBeforeExtendHD = nullptr, *CsiAfterExtendHD = nullptr;
  Function *CsiBeforeExtendFD = nullptr, *CsiAfterExtendFD = nullptr;
  Function *CsiBeforeTruncateDF = nullptr, *CsiAfterTruncateDF = nullptr;
  // Function *CsiBeforeTruncateDH = nullptr, *CsiAfterTruncateDH = nullptr;
  // Function *CsiBeforeTruncateFH = nullptr, *CsiAfterTruncateFH = nullptr;

  Function *CsiBeforeTruncateI128I8 = nullptr,
    *CsiAfterTruncateI128I8 = nullptr;
  Function *CsiBeforeTruncateI128I16 = nullptr,
    *CsiAfterTruncateI128I16 = nullptr;
  Function *CsiBeforeTruncateI128I32 = nullptr,
    *CsiAfterTruncateI128I32 = nullptr;
  Function *CsiBeforeTruncateI128I64 = nullptr,
    *CsiAfterTruncateI128I64 = nullptr;
  Function *CsiBeforeTruncateI64I8 = nullptr, *CsiAfterTruncateI64I8 = nullptr;
  Function *CsiBeforeTruncateI64I16 = nullptr,
    *CsiAfterTruncateI64I16 = nullptr;
  Function *CsiBeforeTruncateI64I32 = nullptr,
    *CsiAfterTruncateI64I32 = nullptr;
  Function *CsiBeforeTruncateI32I8 = nullptr, *CsiAfterTruncateI32I8 = nullptr;
  Function *CsiBeforeTruncateI32I16 = nullptr,
    *CsiAfterTruncateI32I16 = nullptr;
  Function *CsiBeforeTruncateI16I8 = nullptr, *CsiAfterTruncateI16I8 = nullptr;

  Function *CsiBeforeZeroExtendI8I16 = nullptr,
    *CsiAfterZeroExtendI8I16 = nullptr;
  Function *CsiBeforeZeroExtendI8I32 = nullptr,
    *CsiAfterZeroExtendI8I32 = nullptr;
  Function *CsiBeforeZeroExtendI8I64 = nullptr,
    *CsiAfterZeroExtendI8I64 = nullptr;
  Function *CsiBeforeZeroExtendI8I128 = nullptr,
    *CsiAfterZeroExtendI8I128 = nullptr;
  Function *CsiBeforeZeroExtendI16I32 = nullptr,
    *CsiAfterZeroExtendI16I32 = nullptr;
  Function *CsiBeforeZeroExtendI16I64 = nullptr,
    *CsiAfterZeroExtendI16I64 = nullptr;
  Function *CsiBeforeZeroExtendI16I128 = nullptr,
    *CsiAfterZeroExtendI16I128 = nullptr;
  Function *CsiBeforeZeroExtendI32I64 = nullptr,
    *CsiAfterZeroExtendI32I64 = nullptr;
  Function *CsiBeforeZeroExtendI32I128 = nullptr,
    *CsiAfterZeroExtendI32I128 = nullptr;
  Function *CsiBeforeZeroExtendI64I128 = nullptr,
    *CsiAfterZeroExtendI64I128 = nullptr;

  Function *CsiBeforeSignExtendI8I16 = nullptr,
    *CsiAfterSignExtendI8I16 = nullptr;
  Function *CsiBeforeSignExtendI8I32 = nullptr,
    *CsiAfterSignExtendI8I32 = nullptr;
  Function *CsiBeforeSignExtendI8I64 = nullptr,
    *CsiAfterSignExtendI8I64 = nullptr;
  Function *CsiBeforeSignExtendI8I128 = nullptr,
    *CsiAfterSignExtendI8I128 = nullptr;
  Function *CsiBeforeSignExtendI16I32 = nullptr,
    *CsiAfterSignExtendI16I32 = nullptr;
  Function *CsiBeforeSignExtendI16I64 = nullptr,
    *CsiAfterSignExtendI16I64 = nullptr;
  Function *CsiBeforeSignExtendI16I128 = nullptr,
    *CsiAfterSignExtendI16I128 = nullptr;
  Function *CsiBeforeSignExtendI32I64 = nullptr,
    *CsiAfterSignExtendI32I64 = nullptr;
  Function *CsiBeforeSignExtendI32I128 = nullptr,
    *CsiAfterSignExtendI32I128 = nullptr;
  Function *CsiBeforeSignExtendI64I128 = nullptr,
    *CsiAfterSignExtendI64I128 = nullptr;

  // Function *CsiBeforeConvertHUI8 = nullptr, *CsiAfterConvertHUI8 = nullptr;
  // Function *CsiBeforeConvertHUI16 = nullptr, *CsiAfterConvertHUI16 = nullptr;
  // Function *CsiBeforeConvertHUI32 = nullptr, *CsiAfterConvertHUI32 = nullptr;
  // Function *CsiBeforeConvertHUI64 = nullptr, *CsiAfterConvertHUI64 = nullptr;
  // Function *CsiBeforeConvertHUI128 = nullptr, *CsiAfterConvertHUI128 = nullptr;
  Function *CsiBeforeConvertFUI8 = nullptr, *CsiAfterConvertFUI8 = nullptr;
  Function *CsiBeforeConvertFUI16 = nullptr, *CsiAfterConvertFUI16 = nullptr;
  Function *CsiBeforeConvertFUI32 = nullptr, *CsiAfterConvertFUI32 = nullptr;
  Function *CsiBeforeConvertFUI64 = nullptr, *CsiAfterConvertFUI64 = nullptr;
  Function *CsiBeforeConvertFUI128 = nullptr, *CsiAfterConvertFUI128 = nullptr;
  Function *CsiBeforeConvertDUI8 = nullptr, *CsiAfterConvertDUI8 = nullptr;
  Function *CsiBeforeConvertDUI16 = nullptr, *CsiAfterConvertDUI16 = nullptr;
  Function *CsiBeforeConvertDUI32 = nullptr, *CsiAfterConvertDUI32 = nullptr;
  Function *CsiBeforeConvertDUI64 = nullptr, *CsiAfterConvertDUI64 = nullptr;
  Function *CsiBeforeConvertDUI128 = nullptr, *CsiAfterConvertDUI128 = nullptr;

  Function *CsiBeforeConvert4FUI8 = nullptr, *CsiAfterConvert4FUI8 = nullptr;
  Function *CsiBeforeConvert4FUI16 = nullptr, *CsiAfterConvert4FUI16 = nullptr;
  Function *CsiBeforeConvert4FUI32 = nullptr, *CsiAfterConvert4FUI32 = nullptr;
  Function *CsiBeforeConvert4FUI64 = nullptr, *CsiAfterConvert4FUI64 = nullptr;
  Function *CsiBeforeConvert4FUI128 = nullptr,
    *CsiAfterConvert4FUI128 = nullptr;
  Function *CsiBeforeConvert8FUI8 = nullptr, *CsiAfterConvert8FUI8 = nullptr;
  Function *CsiBeforeConvert8FUI16 = nullptr, *CsiAfterConvert8FUI16 = nullptr;
  Function *CsiBeforeConvert8FUI32 = nullptr, *CsiAfterConvert8FUI32 = nullptr;
  Function *CsiBeforeConvert8FUI64 = nullptr, *CsiAfterConvert8FUI64 = nullptr;
  Function *CsiBeforeConvert8FUI128 = nullptr,
    *CsiAfterConvert8FUI128 = nullptr;
  Function *CsiBeforeConvert16FUI8 = nullptr, *CsiAfterConvert16FUI8 = nullptr;
  Function *CsiBeforeConvert16FUI16 = nullptr,
    *CsiAfterConvert16FUI16 = nullptr;
  Function *CsiBeforeConvert16FUI32 = nullptr,
    *CsiAfterConvert16FUI32 = nullptr;
  Function *CsiBeforeConvert16FUI64 = nullptr,
    *CsiAfterConvert16FUI64 = nullptr;
  Function *CsiBeforeConvert16FUI128 = nullptr,
    *CsiAfterConvert16FUI128 = nullptr;
  Function *CsiBeforeConvert2DUI8 = nullptr, *CsiAfterConvert2DUI8 = nullptr;
  Function *CsiBeforeConvert2DUI16 = nullptr, *CsiAfterConvert2DUI16 = nullptr;
  Function *CsiBeforeConvert2DUI32 = nullptr, *CsiAfterConvert2DUI32 = nullptr;
  Function *CsiBeforeConvert2DUI64 = nullptr, *CsiAfterConvert2DUI64 = nullptr;
  Function *CsiBeforeConvert2DUI128 = nullptr,
    *CsiAfterConvert2DUI128 = nullptr;
  Function *CsiBeforeConvert4DUI8 = nullptr, *CsiAfterConvert4DUI8 = nullptr;
  Function *CsiBeforeConvert4DUI16 = nullptr, *CsiAfterConvert4DUI16 = nullptr;
  Function *CsiBeforeConvert4DUI32 = nullptr, *CsiAfterConvert4DUI32 = nullptr;
  Function *CsiBeforeConvert4DUI64 = nullptr, *CsiAfterConvert4DUI64 = nullptr;
  Function *CsiBeforeConvert4DUI128 = nullptr,
    *CsiAfterConvert4DUI128 = nullptr;
  Function *CsiBeforeConvert8DUI8 = nullptr, *CsiAfterConvert8DUI8 = nullptr;
  Function *CsiBeforeConvert8DUI16 = nullptr, *CsiAfterConvert8DUI16 = nullptr;
  Function *CsiBeforeConvert8DUI32 = nullptr, *CsiAfterConvert8DUI32 = nullptr;
  Function *CsiBeforeConvert8DUI64 = nullptr, *CsiAfterConvert8DUI64 = nullptr;
  Function *CsiBeforeConvert8DUI128 = nullptr,
    *CsiAfterConvert8DUI128 = nullptr;

  // Function *CsiBeforeConvertHSI8 = nullptr, *CsiAfterConvertHSI8 = nullptr;
  // Function *CsiBeforeConvertHSI16 = nullptr, *CsiAfterConvertHSI16 = nullptr;
  // Function *CsiBeforeConvertHSI32 = nullptr, *CsiAfterConvertHSI32 = nullptr;
  // Function *CsiBeforeConvertHSI64 = nullptr, *CsiAfterConvertHSI64 = nullptr;
  // Function *CsiBeforeConvertHSI128 = nullptr, *CsiAfterConvertHSI128 = nullptr;
  Function *CsiBeforeConvertFSI8 = nullptr, *CsiAfterConvertFSI8 = nullptr;
  Function *CsiBeforeConvertFSI16 = nullptr, *CsiAfterConvertFSI16 = nullptr;
  Function *CsiBeforeConvertFSI32 = nullptr, *CsiAfterConvertFSI32 = nullptr;
  Function *CsiBeforeConvertFSI64 = nullptr, *CsiAfterConvertFSI64 = nullptr;
  Function *CsiBeforeConvertFSI128 = nullptr, *CsiAfterConvertFSI128 = nullptr;
  Function *CsiBeforeConvertDSI8 = nullptr, *CsiAfterConvertDSI8 = nullptr;
  Function *CsiBeforeConvertDSI16 = nullptr, *CsiAfterConvertDSI16 = nullptr;
  Function *CsiBeforeConvertDSI32 = nullptr, *CsiAfterConvertDSI32 = nullptr;
  Function *CsiBeforeConvertDSI64 = nullptr, *CsiAfterConvertDSI64 = nullptr;
  Function *CsiBeforeConvertDSI128 = nullptr, *CsiAfterConvertDSI128 = nullptr;

  Function *CsiBeforeConvert4FSI8 = nullptr, *CsiAfterConvert4FSI8 = nullptr;
  Function *CsiBeforeConvert4FSI16 = nullptr, *CsiAfterConvert4FSI16 = nullptr;
  Function *CsiBeforeConvert4FSI32 = nullptr, *CsiAfterConvert4FSI32 = nullptr;
  Function *CsiBeforeConvert4FSI64 = nullptr, *CsiAfterConvert4FSI64 = nullptr;
  Function *CsiBeforeConvert4FSI128 = nullptr,
    *CsiAfterConvert4FSI128 = nullptr;
  Function *CsiBeforeConvert8FSI8 = nullptr, *CsiAfterConvert8FSI8 = nullptr;
  Function *CsiBeforeConvert8FSI16 = nullptr, *CsiAfterConvert8FSI16 = nullptr;
  Function *CsiBeforeConvert8FSI32 = nullptr, *CsiAfterConvert8FSI32 = nullptr;
  Function *CsiBeforeConvert8FSI64 = nullptr, *CsiAfterConvert8FSI64 = nullptr;
  Function *CsiBeforeConvert16FSI8 = nullptr, *CsiAfterConvert16FSI8 = nullptr;
  Function *CsiBeforeConvert16FSI16 = nullptr,
    *CsiAfterConvert16FSI16 = nullptr;
  Function *CsiBeforeConvert16FSI32 = nullptr,
    *CsiAfterConvert16FSI32 = nullptr;
  Function *CsiBeforeConvert2DSI8 = nullptr, *CsiAfterConvert2DSI8 = nullptr;
  Function *CsiBeforeConvert2DSI16 = nullptr, *CsiAfterConvert2DSI16 = nullptr;
  Function *CsiBeforeConvert2DSI32 = nullptr, *CsiAfterConvert2DSI32 = nullptr;
  Function *CsiBeforeConvert2DSI64 = nullptr, *CsiAfterConvert2DSI64 = nullptr;
  Function *CsiBeforeConvert2DSI128 = nullptr,
    *CsiAfterConvert2DSI128 = nullptr;
  Function *CsiBeforeConvert4DSI8 = nullptr, *CsiAfterConvert4DSI8 = nullptr;
  Function *CsiBeforeConvert4DSI16 = nullptr, *CsiAfterConvert4DSI16 = nullptr;
  Function *CsiBeforeConvert4DSI32 = nullptr, *CsiAfterConvert4DSI32 = nullptr;
  Function *CsiBeforeConvert4DSI64 = nullptr, *CsiAfterConvert4DSI64 = nullptr;
  Function *CsiBeforeConvert4DSI128 = nullptr,
    *CsiAfterConvert4DSI128 = nullptr;
  Function *CsiBeforeConvert8DSI8 = nullptr, *CsiAfterConvert8DSI8 = nullptr;
  Function *CsiBeforeConvert8DSI16 = nullptr, *CsiAfterConvert8DSI16 = nullptr;
  Function *CsiBeforeConvert8DSI32 = nullptr, *CsiAfterConvert8DSI32 = nullptr;
  Function *CsiBeforeConvert8DSI64 = nullptr, *CsiAfterConvert8DSI64 = nullptr;

  // Function *CsiBeforeConvertUI8H = nullptr, *CsiAfterConvertUI8H = nullptr;
  // Function *CsiBeforeConvertUI16H = nullptr, *CsiAfterConvertUI16H = nullptr;
  // Function *CsiBeforeConvertUI32H = nullptr, *CsiAfterConvertUI32H = nullptr;
  // Function *CsiBeforeConvertUI64H = nullptr, *CsiAfterConvertUI64H = nullptr;
  // Function *CsiBeforeConvertUI128H = nullptr, *CsiAfterConvertUI128H = nullptr;
  Function *CsiBeforeConvertUI8F = nullptr, *CsiAfterConvertUI8F = nullptr;
  Function *CsiBeforeConvertUI16F = nullptr, *CsiAfterConvertUI16F = nullptr;
  Function *CsiBeforeConvertUI32F = nullptr, *CsiAfterConvertUI32F = nullptr;
  Function *CsiBeforeConvertUI64F = nullptr, *CsiAfterConvertUI64F = nullptr;
  Function *CsiBeforeConvertUI128F = nullptr, *CsiAfterConvertUI128F = nullptr;
  Function *CsiBeforeConvertUI8D = nullptr, *CsiAfterConvertUI8D = nullptr;
  Function *CsiBeforeConvertUI16D = nullptr, *CsiAfterConvertUI16D = nullptr;
  Function *CsiBeforeConvertUI32D = nullptr, *CsiAfterConvertUI32D = nullptr;
  Function *CsiBeforeConvertUI64D = nullptr, *CsiAfterConvertUI64D = nullptr;
  Function *CsiBeforeConvertUI128D = nullptr, *CsiAfterConvertUI128D = nullptr;

  Function *CsiBeforeConvert4UI8F = nullptr, *CsiAfterConvert4UI8F = nullptr;
  Function *CsiBeforeConvert4UI16F = nullptr, *CsiAfterConvert4UI16F = nullptr;
  Function *CsiBeforeConvert4UI32F = nullptr, *CsiAfterConvert4UI32F = nullptr;
  Function *CsiBeforeConvert4UI64F = nullptr, *CsiAfterConvert4UI64F = nullptr;
  Function *CsiBeforeConvert4UI128F = nullptr,
    *CsiAfterConvert4UI128F = nullptr;
  Function *CsiBeforeConvert8UI8F = nullptr, *CsiAfterConvert8UI8F = nullptr;
  Function *CsiBeforeConvert8UI16F = nullptr, *CsiAfterConvert8UI16F = nullptr;
  Function *CsiBeforeConvert8UI32F = nullptr, *CsiAfterConvert8UI32F = nullptr;
  Function *CsiBeforeConvert8UI64F = nullptr, *CsiAfterConvert8UI64F = nullptr;
  Function *CsiBeforeConvert16UI8F = nullptr, *CsiAfterConvert16UI8F = nullptr;
  Function *CsiBeforeConvert16UI16F = nullptr,
    *CsiAfterConvert16UI16F = nullptr;
  Function *CsiBeforeConvert16UI32F = nullptr,
    *CsiAfterConvert16UI32F = nullptr;
  Function *CsiBeforeConvert2UI8D = nullptr, *CsiAfterConvert2UI8D = nullptr;
  Function *CsiBeforeConvert2UI16D = nullptr, *CsiAfterConvert2UI16D = nullptr;
  Function *CsiBeforeConvert2UI32D = nullptr, *CsiAfterConvert2UI32D = nullptr;
  Function *CsiBeforeConvert2UI64D = nullptr, *CsiAfterConvert2UI64D = nullptr;
  Function *CsiBeforeConvert2UI128D = nullptr,
    *CsiAfterConvert2UI128D = nullptr;
  Function *CsiBeforeConvert4UI8D = nullptr, *CsiAfterConvert4UI8D = nullptr;
  Function *CsiBeforeConvert4UI16D = nullptr, *CsiAfterConvert4UI16D = nullptr;
  Function *CsiBeforeConvert4UI32D = nullptr, *CsiAfterConvert4UI32D = nullptr;
  Function *CsiBeforeConvert4UI64D = nullptr, *CsiAfterConvert4UI64D = nullptr;
  Function *CsiBeforeConvert4UI128D = nullptr,
    *CsiAfterConvert4UI128D = nullptr;
  Function *CsiBeforeConvert8UI8D = nullptr, *CsiAfterConvert8UI8D = nullptr;
  Function *CsiBeforeConvert8UI16D = nullptr, *CsiAfterConvert8UI16D = nullptr;
  Function *CsiBeforeConvert8UI32D = nullptr, *CsiAfterConvert8UI32D = nullptr;
  Function *CsiBeforeConvert8UI64D = nullptr, *CsiAfterConvert8UI64D = nullptr;

  // Function *CsiBeforeConvertSI8H = nullptr, *CsiAfterConvertSI8H = nullptr;
  // Function *CsiBeforeConvertSI16H = nullptr, *CsiAfterConvertSI16H = nullptr;
  // Function *CsiBeforeConvertSI32H = nullptr, *CsiAfterConvertSI32H = nullptr;
  // Function *CsiBeforeConvertSI64H = nullptr, *CsiAfterConvertSI64H = nullptr;
  // Function *CsiBeforeConvertSI128H = nullptr, *CsiAfterConvertSI128H = nullptr;
  Function *CsiBeforeConvertSI8F = nullptr, *CsiAfterConvertSI8F = nullptr;
  Function *CsiBeforeConvertSI16F = nullptr, *CsiAfterConvertSI16F = nullptr;
  Function *CsiBeforeConvertSI32F = nullptr, *CsiAfterConvertSI32F = nullptr;
  Function *CsiBeforeConvertSI64F = nullptr, *CsiAfterConvertSI64F = nullptr;
  Function *CsiBeforeConvertSI128F = nullptr, *CsiAfterConvertSI128F = nullptr;
  Function *CsiBeforeConvertSI8D = nullptr, *CsiAfterConvertSI8D = nullptr;
  Function *CsiBeforeConvertSI16D = nullptr, *CsiAfterConvertSI16D = nullptr;
  Function *CsiBeforeConvertSI32D = nullptr, *CsiAfterConvertSI32D = nullptr;
  Function *CsiBeforeConvertSI64D = nullptr, *CsiAfterConvertSI64D = nullptr;
  Function *CsiBeforeConvertSI128D = nullptr, *CsiAfterConvertSI128D = nullptr;

  Function *CsiBeforeConvert4SI8F = nullptr, *CsiAfterConvert4SI8F = nullptr;
  Function *CsiBeforeConvert4SI16F = nullptr, *CsiAfterConvert4SI16F = nullptr;
  Function *CsiBeforeConvert4SI32F = nullptr, *CsiAfterConvert4SI32F = nullptr;
  Function *CsiBeforeConvert4SI64F = nullptr, *CsiAfterConvert4SI64F = nullptr;
  Function *CsiBeforeConvert4SI128F = nullptr,
    *CsiAfterConvert4SI128F = nullptr;
  Function *CsiBeforeConvert8SI8F = nullptr, *CsiAfterConvert8SI8F = nullptr;
  Function *CsiBeforeConvert8SI16F = nullptr, *CsiAfterConvert8SI16F = nullptr;
  Function *CsiBeforeConvert8SI32F = nullptr, *CsiAfterConvert8SI32F = nullptr;
  Function *CsiBeforeConvert8SI64F = nullptr, *CsiAfterConvert8SI64F = nullptr;
  Function *CsiBeforeConvert16SI8F = nullptr, *CsiAfterConvert16SI8F = nullptr;
  Function *CsiBeforeConvert16SI16F = nullptr,
    *CsiAfterConvert16SI16F = nullptr;
  Function *CsiBeforeConvert16SI32F = nullptr,
    *CsiAfterConvert16SI32F = nullptr;
  Function *CsiBeforeConvert2SI8D = nullptr, *CsiAfterConvert2SI8D = nullptr;
  Function *CsiBeforeConvert2SI16D = nullptr, *CsiAfterConvert2SI16D = nullptr;
  Function *CsiBeforeConvert2SI32D = nullptr, *CsiAfterConvert2SI32D = nullptr;
  Function *CsiBeforeConvert2SI64D = nullptr, *CsiAfterConvert2SI64D = nullptr;
  Function *CsiBeforeConvert2SI128D = nullptr,
    *CsiAfterConvert2SI128D = nullptr;
  Function *CsiBeforeConvert4SI8D = nullptr, *CsiAfterConvert4SI8D = nullptr;
  Function *CsiBeforeConvert4SI16D = nullptr, *CsiAfterConvert4SI16D = nullptr;
  Function *CsiBeforeConvert4SI32D = nullptr, *CsiAfterConvert4SI32D = nullptr;
  Function *CsiBeforeConvert4SI64D = nullptr, *CsiAfterConvert4SI64D = nullptr;
  Function *CsiBeforeConvert4SI128D = nullptr,
    *CsiAfterConvert4SI128D = nullptr;
  Function *CsiBeforeConvert8SI8D = nullptr, *CsiAfterConvert8SI8D = nullptr;
  Function *CsiBeforeConvert8SI16D = nullptr, *CsiAfterConvert8SI16D = nullptr;
  Function *CsiBeforeConvert8SI32D = nullptr, *CsiAfterConvert8SI32D = nullptr;
  Function *CsiBeforeConvert8SI64D = nullptr, *CsiAfterConvert8SI64D = nullptr;

  Function /**CsiPhiH = nullptr,*/ *CsiPhiF = nullptr, *CsiPhiD = nullptr;
  Function *CsiPhiI8 = nullptr, *CsiPhiI16 = nullptr, *CsiPhiI32 = nullptr;
  Function *CsiPhiI64 = nullptr, *CsiPhiI128 = nullptr, *CsiPhi4F = nullptr;
  Function *CsiPhi8F = nullptr, *CsiPhi16F = nullptr, *CsiPhi2D = nullptr;
  Function *CsiPhi4D = nullptr, *CsiPhi8D = nullptr;

  // Vector operations
  Function *CsiBeforeInsertEl4F = nullptr, *CsiAfterInsertEl4F = nullptr;
  Function *CsiBeforeInsertEl8F = nullptr, *CsiAfterInsertEl8F = nullptr;
  Function *CsiBeforeInsertEl16F = nullptr, *CsiAfterInsertEl16F = nullptr;
  Function *CsiBeforeInsertEl2D = nullptr, *CsiAfterInsertEl2D = nullptr;
  Function *CsiBeforeInsertEl4D = nullptr, *CsiAfterInsertEl4D = nullptr;
  Function *CsiBeforeInsertEl8D = nullptr, *CsiAfterInsertEl8D = nullptr;

  Function *CsiBeforeExtractEl4F = nullptr, *CsiAfterExtractEl4F = nullptr;
  Function *CsiBeforeExtractEl8F = nullptr, *CsiAfterExtractEl8F = nullptr;
  Function *CsiBeforeExtractEl16F = nullptr, *CsiAfterExtractEl16F = nullptr;
  Function *CsiBeforeExtractEl2D = nullptr, *CsiAfterExtractEl2D = nullptr;
  Function *CsiBeforeExtractEl4D = nullptr, *CsiAfterExtractEl4D = nullptr;
  Function *CsiBeforeExtractEl8D = nullptr, *CsiAfterExtractEl8D = nullptr;

  Function *CsiBeforeShuffle4F4F = nullptr, *CsiAfterShuffle4F4F = nullptr;
  Function *CsiBeforeShuffle4F8F = nullptr, *CsiAfterShuffle4F8F = nullptr;
  Function *CsiBeforeShuffle4F16F = nullptr, *CsiAfterShuffle4F16F = nullptr;
  Function *CsiBeforeShuffle8F4F = nullptr, *CsiAfterShuffle8F4F = nullptr;
  Function *CsiBeforeShuffle8F8F = nullptr, *CsiAfterShuffle8F8F = nullptr;
  Function *CsiBeforeShuffle8F16F = nullptr, *CsiAfterShuffle8F16F = nullptr;
  Function *CsiBeforeShuffle16F4F = nullptr, *CsiAfterShuffle16F4F = nullptr;
  Function *CsiBeforeShuffle16F8F = nullptr, *CsiAfterShuffle16F8F = nullptr;
  Function *CsiBeforeShuffle16F16F = nullptr, *CsiAfterShuffle16F16F = nullptr;

  Function *CsiBeforeShuffle2D2D = nullptr, *CsiAfterShuffle2D2D = nullptr;
  Function *CsiBeforeShuffle2D4D = nullptr, *CsiAfterShuffle2D4D = nullptr;
  Function *CsiBeforeShuffle2D8D = nullptr, *CsiAfterShuffle2D8D = nullptr;
  Function *CsiBeforeShuffle4D2D = nullptr, *CsiAfterShuffle4D2D = nullptr;
  Function *CsiBeforeShuffle4D4D = nullptr, *CsiAfterShuffle4D4D = nullptr;
  Function *CsiBeforeShuffle4D8D = nullptr, *CsiAfterShuffle4D8D = nullptr;
  Function *CsiBeforeShuffle8D2D = nullptr, *CsiAfterShuffle8D2D = nullptr;
  Function *CsiBeforeShuffle8D4D = nullptr, *CsiAfterShuffle8D4D = nullptr;
  Function *CsiBeforeShuffle8D8D = nullptr, *CsiAfterShuffle8D8D = nullptr;

  // Built-in vector loads, stores, gathers, scatters
  Function *CsiBeforeVMaskedLoad4F = nullptr, *CsiAfterVMaskedLoad4F = nullptr;
  Function *CsiBeforeVMaskedLoad8F = nullptr, *CsiAfterVMaskedLoad8F = nullptr;
  Function *CsiBeforeVMaskedLoad16F = nullptr,
    *CsiAfterVMaskedLoad16F = nullptr;
  Function *CsiBeforeVMaskedLoad2D = nullptr, *CsiAfterVMaskedLoad2D = nullptr;
  Function *CsiBeforeVMaskedLoad4D = nullptr, *CsiAfterVMaskedLoad4D = nullptr;
  Function *CsiBeforeVMaskedLoad8D = nullptr, *CsiAfterVMaskedLoad8D = nullptr;

  Function *CsiBeforeVMaskedStore4F = nullptr,
    *CsiAfterVMaskedStore4F = nullptr;
  Function *CsiBeforeVMaskedStore8F = nullptr,
    *CsiAfterVMaskedStore8F = nullptr;
  Function *CsiBeforeVMaskedStore16F = nullptr,
    *CsiAfterVMaskedStore16F = nullptr;
  Function *CsiBeforeVMaskedStore2D = nullptr,
    *CsiAfterVMaskedStore2D = nullptr;
  Function *CsiBeforeVMaskedStore4D = nullptr,
    *CsiAfterVMaskedStore4D = nullptr;
  Function *CsiBeforeVMaskedStore8D = nullptr,
    *CsiAfterVMaskedStore8D = nullptr;

  Function *CsiBeforeVMaskedGather4F = nullptr,
    *CsiAfterVMaskedGather4F = nullptr;
  Function *CsiBeforeVMaskedGather8F = nullptr,
    *CsiAfterVMaskedGather8F = nullptr;
  Function *CsiBeforeVMaskedGather16F = nullptr,
    *CsiAfterVMaskedGather16F = nullptr;
  Function *CsiBeforeVMaskedGather2D = nullptr,
    *CsiAfterVMaskedGather2D = nullptr;
  Function *CsiBeforeVMaskedGather4D = nullptr,
    *CsiAfterVMaskedGather4D = nullptr;
  Function *CsiBeforeVMaskedGather8D = nullptr,
    *CsiAfterVMaskedGather8D = nullptr;

  Function *CsiBeforeVMaskedScatter4F = nullptr,
    *CsiAfterVMaskedScatter4F = nullptr;
  Function *CsiBeforeVMaskedScatter8F = nullptr,
    *CsiAfterVMaskedScatter8F = nullptr;
  Function *CsiBeforeVMaskedScatter16F = nullptr,
    *CsiAfterVMaskedScatter16F = nullptr;
  Function *CsiBeforeVMaskedScatter2D = nullptr,
    *CsiAfterVMaskedScatter2D = nullptr;
  Function *CsiBeforeVMaskedScatter4D = nullptr,
    *CsiAfterVMaskedScatter4D = nullptr;
  Function *CsiBeforeVMaskedScatter8D = nullptr,
    *CsiAfterVMaskedScatter8D = nullptr;

  // Hooks for builtins
  Function *CsiBeforeBuiltinFF = nullptr, *CsiBeforeBuiltinDD = nullptr;
  Function *CsiBeforeBuiltinFFF = nullptr, *CsiBeforeBuiltinDDD = nullptr;
  Function *CsiBeforeBuiltinFFI = nullptr, *CsiBeforeBuiltinDDI = nullptr;
  Function *CsiBeforeBuiltinFFFF = nullptr, *CsiBeforeBuiltinDDDD = nullptr;

  Function *CsiAfterBuiltinFF = nullptr, *CsiAfterBuiltinDD = nullptr;
  Function *CsiAfterBuiltinFFF = nullptr, *CsiAfterBuiltinDDD = nullptr;
  Function *CsiAfterBuiltinFFI = nullptr, *CsiAfterBuiltinDDI = nullptr;
  Function *CsiAfterBuiltinFFFF = nullptr, *CsiAfterBuiltinDDDD = nullptr;

  Function *CsiBeforeMemset = nullptr, *CsiAfterMemset = nullptr;
  Function *CsiBeforeMemcpy = nullptr, *CsiAfterMemcpy = nullptr;
  Function *CsiBeforeMemmove = nullptr, *CsiAfterMemmove = nullptr;

  Function *InitCallsiteToFunction = nullptr;
  // GlobalVariable *DisableInstrGV;

  // Runtime unit initialization
  Function *RTUnitInit = nullptr;

  Type *IntptrTy;
  DenseMap<StringRef, csi_id_t> FuncOffsetMap, GlobalOffsetMap;

  DenseMap<std::pair<Value *, BasicBlock *>, SmallVector<PHINode *, 4>> ArgPHIs;
  DenseMap<BasicBlock *, CallInst *> callsAfterSync;
  std::unique_ptr<InstrumentationConfig> Config;

  // Declarations of interposition functions.
  DenseMap<Function *, Function *> InterpositionFunctions;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_CSI_H
