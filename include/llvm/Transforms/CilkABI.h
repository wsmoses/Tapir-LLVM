//===- Detach2Cilk.cpp - The -detach2cilk pass, a wrapper around the Utils lib ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//
#ifndef CILK_ABI_H_
#define CILK_ABI_H_

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/ValueSymbolTable.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/Support/Debug.h"
#include <iostream>

namespace {

	typedef void *__CILK_JUMP_BUFFER[5];

	struct __cilkrts_pedigree {};
	struct __cilkrts_stack_frame {};
	struct __cilkrts_worker {};

	enum {
		__CILKRTS_ABI_VERSION = 1
	};

	enum {
		CILK_FRAME_STOLEN           =    0x01,
		CILK_FRAME_UNSYNCHED        =    0x02,
		CILK_FRAME_DETACHED         =    0x04,
		CILK_FRAME_EXCEPTION_PROBED =    0x08,
		CILK_FRAME_EXCEPTING        =    0x10,
		CILK_FRAME_LAST             =    0x80,
		CILK_FRAME_EXITING          =  0x0100,
		CILK_FRAME_SUSPENDED        =  0x8000,
		CILK_FRAME_UNWINDING        = 0x10000
	};

const bool EXCEPTIONS = false;

#define CILK_FRAME_VERSION (__CILKRTS_ABI_VERSION << 24)
#define CILK_FRAME_VERSION_MASK  0xFF000000
#define CILK_FRAME_FLAGS_MASK    0x00FFFFFF
#define CILK_FRAME_VERSION_VALUE(_flags) (((_flags) & CILK_FRAME_VERSION_MASK) >> 24)
#define CILK_FRAME_MBZ  (~ (CILK_FRAME_STOLEN           | \
			CILK_FRAME_UNSYNCHED        | \
			CILK_FRAME_DETACHED         | \
			CILK_FRAME_EXCEPTION_PROBED | \
			CILK_FRAME_EXCEPTING        | \
			CILK_FRAME_LAST             | \
			CILK_FRAME_EXITING          | \
			CILK_FRAME_SUSPENDED        | \
			CILK_FRAME_UNWINDING        | \
			CILK_FRAME_VERSION_MASK))

	typedef uint32_t cilk32_t;
	typedef uint64_t cilk64_t;
	typedef void (*__cilk_abi_f32_t)(void *data, cilk32_t low, cilk32_t high);
	typedef void (*__cilk_abi_f64_t)(void *data, cilk64_t low, cilk64_t high);

	typedef void (__cilkrts_enter_frame)(__cilkrts_stack_frame *sf);
	typedef void (__cilkrts_enter_frame_1)(__cilkrts_stack_frame *sf);
	typedef void (__cilkrts_enter_frame_fast)(__cilkrts_stack_frame *sf);
	typedef void (__cilkrts_enter_frame_fast_1)(__cilkrts_stack_frame *sf);
	typedef void (__cilkrts_leave_frame)(__cilkrts_stack_frame *sf);
	typedef void (__cilkrts_sync)(__cilkrts_stack_frame *sf);
	typedef void (__cilkrts_return_exception)(__cilkrts_stack_frame *sf);
	typedef void (__cilkrts_rethrow)(__cilkrts_stack_frame *sf);
	typedef void (__cilkrts_detach)(__cilkrts_stack_frame *sf);
	typedef void (__cilkrts_pop_frame)(__cilkrts_stack_frame *sf);
	typedef __cilkrts_worker *(__cilkrts_get_tls_worker)();
	typedef __cilkrts_worker *(__cilkrts_get_tls_worker_fast)();
	typedef __cilkrts_worker *(__cilkrts_bind_thread_1)();
	typedef void (__cilkrts_cilk_for_32)(__cilk_abi_f32_t body, void *data,
			cilk32_t count, int grain);
	typedef void (__cilkrts_cilk_for_64)(__cilk_abi_f64_t body, void *data,
			cilk64_t count, int grain);

	typedef void (cilk_func)(__cilkrts_stack_frame *);

#define CILKRTS_FUNC(name, CGF) Get__cilkrts_##name(CGF)

#define DEFAULT_GET_CILKRTS_FUNC(name) \
	static llvm::Function *Get__cilkrts_##name(llvm::Module& M) { \
		return llvm::cast<llvm::Function>(M.getOrInsertFunction( \
					"__cilkrts_"#name,\
					llvm::TypeBuilder<__cilkrts_##name, false>::get(M.getContext()) \
					)); \
	}

DEFAULT_GET_CILKRTS_FUNC(sync)
DEFAULT_GET_CILKRTS_FUNC(rethrow)
DEFAULT_GET_CILKRTS_FUNC(leave_frame)
DEFAULT_GET_CILKRTS_FUNC(get_tls_worker)
DEFAULT_GET_CILKRTS_FUNC(bind_thread_1)
DEFAULT_GET_CILKRTS_FUNC(cilk_for_32)
DEFAULT_GET_CILKRTS_FUNC(cilk_for_64)

	typedef std::map<llvm::LLVMContext*, llvm::StructType*> TypeBuilderCache;
}
	namespace llvm {

		/// Specializations of llvm::TypeBuilder for:
		///   __cilkrts_pedigree,
		///   __cilkrts_worker,
		///   __cilkrts_stack_frame
		template <bool X>
			class TypeBuilder<__cilkrts_pedigree, X> {
				public:
					static StructType *get(LLVMContext &C) {
						static TypeBuilderCache cache;
						TypeBuilderCache::iterator I = cache.find(&C);
						if (I != cache.end())
							return I->second;
						StructType *Ty = StructType::create(C, "__cilkrts_pedigree");
						cache[&C] = Ty;
						Ty->setBody(
								TypeBuilder<uint64_t,            X>::get(C), // rank
								TypeBuilder<__cilkrts_pedigree*, X>::get(C), // next
								NULL);
						return Ty;
					}
					enum {
						rank,
						next
					};
			};

		template <bool X>
			class TypeBuilder<__cilkrts_worker, X> {
				public:
					static StructType *get(LLVMContext &C) {
						static TypeBuilderCache cache;
						TypeBuilderCache::iterator I = cache.find(&C);
						if (I != cache.end())
							return I->second;
						StructType *Ty = StructType::create(C, "__cilkrts_worker");
						cache[&C] = Ty;
						Ty->setBody(
								TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // tail
								TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // head
								TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // exc
								TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // protected_tail
								TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // ltq_limit
								TypeBuilder<int32_t,                 X>::get(C), // self
								TypeBuilder<void*,                   X>::get(C), // g
								TypeBuilder<void*,                   X>::get(C), // l
								TypeBuilder<void*,                   X>::get(C), // reducer_map
								TypeBuilder<__cilkrts_stack_frame*,  X>::get(C), // current_stack_frame
								TypeBuilder<__cilkrts_stack_frame**, X>::get(C), // saved_protected_tail
								TypeBuilder<void*,                   X>::get(C), // sysdep
								TypeBuilder<__cilkrts_pedigree,      X>::get(C), // pedigree
								NULL);
						return Ty;
					}
					enum {
						tail,
						head,
						exc,
						protected_tail,
						ltq_limit,
						self,
						g,
						l,
						reducer_map,
						current_stack_frame,
						saved_protected_tail,
						sysdep,
						pedigree
					};
			};

		template <bool X>
			class TypeBuilder<__cilkrts_stack_frame, X> {
				public:
					static StructType *get(LLVMContext &C) {
						static TypeBuilderCache cache;
						TypeBuilderCache::iterator I = cache.find(&C);
						if (I != cache.end())
							return I->second;
						StructType *Ty = StructType::create(C, "__cilkrts_stack_frame");
						cache[&C] = Ty;
						Ty->setBody(
								TypeBuilder<uint32_t,               X>::get(C), // flags
								TypeBuilder<int32_t,                X>::get(C), // size
								TypeBuilder<__cilkrts_stack_frame*, X>::get(C), // call_parent
								TypeBuilder<__cilkrts_worker*,      X>::get(C), // worker
								TypeBuilder<void*,                  X>::get(C), // except_data
								TypeBuilder<__CILK_JUMP_BUFFER,     X>::get(C), // ctx
								TypeBuilder<uint32_t,               X>::get(C), // mxcsr
								TypeBuilder<uint16_t,               X>::get(C), // fpcsr
								TypeBuilder<uint16_t,               X>::get(C), // reserved
								TypeBuilder<__cilkrts_pedigree,     X>::get(C), // parent_pedigree
								NULL);
						return Ty;
					}
					enum {
						flags,
						size,
						call_parent,
						worker,
						except_data,
						ctx,
						mxcsr,
						fpcsr,
						reserved,
						parent_pedigree
					};
			};

	} // namespace llvm

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace llvm {
namespace cilk {

/// Helper typedefs for cilk struct TypeBuilders.
typedef llvm::TypeBuilder<__cilkrts_stack_frame, false> StackFrameBuilder;
typedef llvm::TypeBuilder<__cilkrts_worker, false> WorkerBuilder;
typedef llvm::TypeBuilder<__cilkrts_pedigree, false> PedigreeBuilder;

static Value *GEP(IRBuilder<> &B, Value *Base, int field) {
	return B.CreateConstInBoundsGEP2_32(nullptr, Base, 0, field);
}

static void StoreField(IRBuilder<> &B, Value *Val, Value *Dst, int field) {
	B.CreateStore(Val, GEP(B, Dst, field));
}

static Value *LoadField(IRBuilder<> &B, Value *Src, int field) {
	return B.CreateLoad(GEP(B, Src, field));
}

/// \brief Emit inline assembly code to save the floating point
/// state, for x86 Only.
static void EmitSaveFloatingPointState(IRBuilder<> &B, Value *SF) {
  typedef void (AsmPrototype)(uint32_t*, uint16_t*);
  llvm::FunctionType *FTy =
    TypeBuilder<AsmPrototype, false>::get(B.getContext());

  Value *Asm = InlineAsm::get(FTy,
                              "stmxcsr $0\n\t" "fnstcw $1",
                              "*m,*m,~{dirflag},~{fpsr},~{flags}",
                              /*sideeffects*/ true);

  Value * args[2] = {
    GEP(B, SF, StackFrameBuilder::mxcsr),
    GEP(B, SF, StackFrameBuilder::fpcsr)
  };
  
  B.CreateCall(Asm, args);
}

/// \brief Helper to find a function with the given name, creating it if it
/// doesn't already exist. If the function needed to be created then return
/// false, signifying that the caller needs to add the function body.
template <typename T>
static bool GetOrCreateFunction(const char *FnName, Module& M,
                                Function *&Fn, Function::LinkageTypes Linkage =
                                               Function::InternalLinkage,
                                bool DoesNotThrow = true) {
  LLVMContext &Ctx = M.getContext();

  Fn = M.getFunction(FnName);

  // if the function already exists then let the
  // caller know that it is complete
  if (Fn)
    return true;

  // Otherwise we have to create it
  llvm::FunctionType *FTy = TypeBuilder<T, false>::get(Ctx);
  Fn = Function::Create(FTy, Linkage, FnName, &M);

  // Set nounwind if it does not throw.
  if (DoesNotThrow)
    Fn->setDoesNotThrow();

  // and let the caller know that the function is incomplete
  // and the body still needs to be added
  return false;
}

/// \brief Register a sync function with a named metadata.
static void registerSyncFunction(Module &M, llvm::Function *Fn) {
//TODO?
//  LLVMContext &Context = M.getContext();
//  llvm::NamedMDNode *SyncMetadata = M.getOrInsertNamedMetadata("cilk.sync");

//  SyncMetadata->addOperand(llvm::MDNode::get(Context, Fn));
}

/// \brief Register a spawn helper function with a named metadata.
static void registerSpawnFunction( llvm::Function& Fn) {
//TODO?
//  LLVMContext &Context = Fn.getContext();
//  llvm::NamedMDNode *SpawnMetadata = Fn.getParent().getOrInsertNamedMetadata("cilk.spawn");

//  SpawnMetadata->addOperand(llvm::MDNode::get(Context, Fn));
}

/// \brief Emit a call to the CILK_SETJMP function.
static CallInst *EmitCilkSetJmp(IRBuilder<> &B, Value *SF, Module& M) {
  LLVMContext &Ctx = M.getContext();

  // We always want to save the floating point state too
  EmitSaveFloatingPointState(B, SF);

  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);
  llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(Ctx);

  // Get the buffer to store program state
  // Buffer is a void**.
  Value *Buf = GEP(B, SF, StackFrameBuilder::ctx);

  // Store the frame pointer in the 0th slot
  Value *FrameAddr =
    B.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::frameaddress),
                 ConstantInt::get(Int32Ty, 0));

  Value *FrameSaveSlot = GEP(B, Buf, 0);
  B.CreateStore(FrameAddr, FrameSaveSlot);

  // Store stack pointer in the 2nd slot
  Value *StackAddr = B.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::stacksave));

  Value *StackSaveSlot = GEP(B, Buf, 2);
  B.CreateStore(StackAddr, StackSaveSlot);

  // Call LLVM's EH setjmp, which is lightweight.
  
	Value* F = Intrinsic::getDeclaration(&M, Intrinsic::eh_sjlj_setjmp);
  Buf = B.CreateBitCast(Buf, Int8PtrTy);

  CallInst *SetjmpCall = B.CreateCall(F, Buf);
  SetjmpCall->setCanReturnTwice();
  return SetjmpCall;
}

/// \brief Get or create a LLVM function for __cilkrts_pop_frame.
/// It is equivalent to the following C code
///
/// __cilkrts_pop_frame(__cilkrts_stack_frame *sf) {
///   sf->worker->current_stack_frame = sf->call_parent;
///   sf->call_parent = 0;
/// }
static Function *Get__cilkrts_pop_frame(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_pop_frame", M, Fn))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  // sf->worker->current_stack_frame = sf.call_parent;
  StoreField(B,
    LoadField(B, SF, StackFrameBuilder::call_parent),
    LoadField(B, SF, StackFrameBuilder::worker),
    WorkerBuilder::current_stack_frame);

  // sf->call_parent = 0;
  StoreField(B,
    Constant::getNullValue(TypeBuilder<__cilkrts_stack_frame*, false>::get(Ctx)),
    SF, StackFrameBuilder::call_parent);

  B.CreateRetVoid();

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilkrts_detach.
/// It is equivalent to the following C code
///
/// void __cilkrts_detach(struct __cilkrts_stack_frame *sf) {
///   struct __cilkrts_worker *w = sf->worker;
///   struct __cilkrts_stack_frame *volatile *tail = w->tail;
///
///   sf->spawn_helper_pedigree = w->pedigree;
///   sf->call_parent->parent_pedigree = w->pedigree;
///
///   w->pedigree.rank = 0;
///   w->pedigree.next = &sf->spawn_helper_pedigree;
///
///   *tail++ = sf->call_parent;
///   w->tail = tail;
///
///   sf->flags |= CILK_FRAME_DETACHED;
/// }
static Function *Get__cilkrts_detach(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_detach", M, Fn))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  // struct __cilkrts_worker *w = sf->worker;
  Value *W = LoadField(B, SF, StackFrameBuilder::worker);

  // __cilkrts_stack_frame *volatile *tail = w->tail;
  Value *Tail = LoadField(B, W, WorkerBuilder::tail);

  // sf->spawn_helper_pedigree = w->pedigree;
  StoreField(B,
             LoadField(B, W, WorkerBuilder::pedigree),
             SF, StackFrameBuilder::parent_pedigree);

  // sf->call_parent->parent_pedigree = w->pedigree;
  StoreField(B,
             LoadField(B, W, WorkerBuilder::pedigree),
             LoadField(B, SF, StackFrameBuilder::call_parent),
             StackFrameBuilder::parent_pedigree);

  // w->pedigree.rank = 0;
  {
    StructType *STy = PedigreeBuilder::get(Ctx);
    llvm::Type *Ty = STy->getElementType(PedigreeBuilder::rank);
    StoreField(B,
               ConstantInt::get(Ty, 0),
               GEP(B, W, WorkerBuilder::pedigree),
               PedigreeBuilder::rank);
  }

  // w->pedigree.next = &sf->spawn_helper_pedigree;
  StoreField(B,
             GEP(B, SF, StackFrameBuilder::parent_pedigree),
             GEP(B, W, WorkerBuilder::pedigree),
             PedigreeBuilder::next);

  // *tail++ = sf->call_parent;
  B.CreateStore(LoadField(B, SF, StackFrameBuilder::call_parent), Tail);
  Tail = B.CreateConstGEP1_32(Tail, 1);

  // w->tail = tail;
  StoreField(B, Tail, W, WorkerBuilder::tail);

  // sf->flags |= CILK_FRAME_DETACHED;
  {
    Value *F = LoadField(B, SF, StackFrameBuilder::flags);
    F = B.CreateOr(F, ConstantInt::get(F->getType(), CILK_FRAME_DETACHED));
    StoreField(B, F, SF, StackFrameBuilder::flags);
  }

  B.CreateRetVoid();

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilk_excepting_sync.
/// This is a special sync to be inserted before processing a catch statement.
/// Calls to this function are always inlined.
///
/// It is equivalent to the following C code
///
/// void __cilk_excepting_sync(struct __cilkrts_stack_frame *sf, void **ExnSlot) {
///   if (sf->flags & CILK_FRAME_UNSYNCHED) {
///     if (!CILK_SETJMP(sf->ctx)) {
///       sf->except_data = *ExnSlot;
///       sf->flags |= CILK_FRAME_EXCEPTING;
///       __cilkrts_sync(sf);
///     }
///     sf->flags &= ~CILK_FRAME_EXCEPTING;
///     *ExnSlot = sf->except_data;
///   }
///   ++sf->worker->pedigree.rank;
/// }
static Function *GetCilkExceptingSyncFn(Module &M) {
  Function *Fn = 0;

  typedef void (cilk_func_1)(__cilkrts_stack_frame *, void **);
  if (GetOrCreateFunction<cilk_func_1>("__cilk_excepting_sync", M, Fn))
    return Fn;

  LLVMContext &Ctx = M.getContext();
  assert((Fn->arg_size() == 2) && "unexpected function type");
  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args++;
  Value *ExnSlot = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn),
             *JumpTest = BasicBlock::Create(Ctx, "setjmp.test", Fn),
             *JumpIf = BasicBlock::Create(Ctx, "setjmp.if", Fn),
             *JumpCont = BasicBlock::Create(Ctx, "setjmp.cont", Fn),
             *Exit = BasicBlock::Create(Ctx, "exit", Fn);

  // Entry
  {
    IRBuilder<> B(Entry);

    // if (sf->flags & CILK_FRAME_UNSYNCHED)
    Value *Flags = LoadField(B, SF, StackFrameBuilder::flags);
    Flags = B.CreateAnd(Flags,
                        ConstantInt::get(Flags->getType(),
                                         CILK_FRAME_UNSYNCHED));
    Value *Zero = Constant::getNullValue(Flags->getType());

    Value *Unsynced = B.CreateICmpEQ(Flags, Zero);
    B.CreateCondBr(Unsynced, Exit, JumpTest);
  }

  // JumpTest
  {
    IRBuilder<> B(JumpTest);
    // if (!CILK_SETJMP(sf.ctx))
    Value *C = EmitCilkSetJmp(B, SF, M);
    C = B.CreateICmpEQ(C, Constant::getNullValue(C->getType()));
    B.CreateCondBr(C, JumpIf, JumpCont);
  }

  // JumpIf
  {
    IRBuilder<> B(JumpIf);

    // sf->except_data = *ExnSlot;
    StoreField(B, B.CreateLoad(ExnSlot), SF, StackFrameBuilder::except_data);

    // sf->flags |= CILK_FRAME_EXCEPTING;
    Value *Flags = LoadField(B, SF, StackFrameBuilder::flags);
    Flags = B.CreateOr(Flags, ConstantInt::get(Flags->getType(),
                                               CILK_FRAME_EXCEPTING));
    StoreField(B, Flags, SF, StackFrameBuilder::flags);

    // __cilkrts_sync(&sf);
    B.CreateCall(CILKRTS_FUNC(sync, M), SF);
    B.CreateBr(JumpCont);
  }

  // JumpCont
  {
    IRBuilder<> B(JumpCont);

    // sf->flags &= ~CILK_FRAME_EXCEPTING;
    Value *Flags = LoadField(B, SF, StackFrameBuilder::flags);
    Flags = B.CreateAnd(Flags, ConstantInt::get(Flags->getType(),
                                                ~CILK_FRAME_EXCEPTING));
    StoreField(B, Flags, SF, StackFrameBuilder::flags);

    // Exn = sf->except_data;
    B.CreateStore(LoadField(B, SF, StackFrameBuilder::except_data), ExnSlot);
    B.CreateBr(Exit);
  }

  // Exit
  {
    IRBuilder<> B(Exit);

    // ++sf.worker->pedigree.rank;
    Value *Rank = LoadField(B, SF, StackFrameBuilder::worker);
    Rank = GEP(B, Rank, WorkerBuilder::pedigree);
    Rank = GEP(B, Rank, PedigreeBuilder::rank);
    B.CreateStore(B.CreateAdd(B.CreateLoad(Rank),
                  ConstantInt::get(Rank->getType()->getPointerElementType(),
                                   1)),
                  Rank);
    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::AlwaysInline);
  Fn->addFnAttr(Attribute::ReturnsTwice);
  //***INTEL
  // Special Intel-specific attribute for inliner.
  Fn->addFnAttr("INTEL_ALWAYS_INLINE");
  registerSyncFunction(M, Fn);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilk_sync.
/// Calls to this function is always inlined, as it saves
/// the current stack/frame pointer values. This function must be marked
/// as returns_twice to allow it to be inlined, since the call to setjmp
/// is marked returns_twice.
///
/// It is equivalent to the following C code
///
/// void __cilk_sync(struct __cilkrts_stack_frame *sf) {
///   if (sf->flags & CILK_FRAME_UNSYNCHED) {
///     sf->parent_pedigree = sf->worker->pedigree;
///     SAVE_FLOAT_STATE(*sf);
///     if (!CILK_SETJMP(sf->ctx))
///       __cilkrts_sync(sf);
///     else if (sf->flags & CILK_FRAME_EXCEPTING)
///       __cilkrts_rethrow(sf);
///   }
///   ++sf->worker->pedigree.rank;
/// }
///
/// With exceptions disabled in the compiler, the function
/// does not call __cilkrts_rethrow()
static Function *GetCilkSyncFn(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilk_sync", M, Fn, Function::InternalLinkage, /*doesNotThrow*/false))
    return Fn;

	// If we get here we need to add the function body
	LLVMContext &Ctx = M.getContext();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "cilk.sync.test", Fn),
             *SaveState = BasicBlock::Create(Ctx, "cilk.sync.savestate", Fn),
             *SyncCall = BasicBlock::Create(Ctx, "cilk.sync.runtimecall", Fn),
             *Excepting = BasicBlock::Create(Ctx, "cilk.sync.excepting", Fn),
             *Rethrow = EXCEPTIONS ?
                          BasicBlock::Create(Ctx, "cilk.sync.rethrow", Fn) : 0,
             *Exit = BasicBlock::Create(Ctx, "cilk.sync.end", Fn);

  // Entry
  {
    IRBuilder<> B(Entry);

    // if (sf->flags & CILK_FRAME_UNSYNCHED)
    Value *Flags = LoadField(B, SF, StackFrameBuilder::flags);
    Flags = B.CreateAnd(Flags,
                        ConstantInt::get(Flags->getType(),
                                         CILK_FRAME_UNSYNCHED));
    Value *Zero = ConstantInt::get(Flags->getType(), 0);
    Value *Unsynced = B.CreateICmpEQ(Flags, Zero);
    B.CreateCondBr(Unsynced, Exit, SaveState);
  }

  // SaveState
  {
    IRBuilder<> B(SaveState);

    // sf.parent_pedigree = sf.worker->pedigree;
    StoreField(B,
      LoadField(B, LoadField(B, SF, StackFrameBuilder::worker),
                WorkerBuilder::pedigree),
      SF, StackFrameBuilder::parent_pedigree);

    // if (!CILK_SETJMP(sf.ctx))
    Value *C = EmitCilkSetJmp(B, SF, M);
    C = B.CreateICmpEQ(C, ConstantInt::get(C->getType(), 0));
    B.CreateCondBr(C, SyncCall, Excepting);
  }

  // SyncCall
  {
    IRBuilder<> B(SyncCall);

    // __cilkrts_sync(&sf);
    B.CreateCall(CILKRTS_FUNC(sync, M), SF);
    B.CreateBr(Exit);
  }

  // Excepting
  {
    IRBuilder<> B(Excepting);
    if (EXCEPTIONS) {
      Value *Flags = LoadField(B, SF, StackFrameBuilder::flags);
      Flags = B.CreateAnd(Flags,
                          ConstantInt::get(Flags->getType(),
                                          CILK_FRAME_EXCEPTING));
      Value *Zero = ConstantInt::get(Flags->getType(), 0);
      Value *C = B.CreateICmpEQ(Flags, Zero);
      B.CreateCondBr(C, Exit, Rethrow);
    } else {
      B.CreateBr(Exit);
    }
  }

  // Rethrow
  if (EXCEPTIONS) {
    IRBuilder<> B(Rethrow);
    B.CreateCall(CILKRTS_FUNC(rethrow, M), SF)->setDoesNotReturn();
    B.CreateUnreachable();
  }

  // Exit
  {
    IRBuilder<> B(Exit);

    // ++sf.worker->pedigree.rank;
    Value *Rank = LoadField(B, SF, StackFrameBuilder::worker);
    Rank = GEP(B, Rank, WorkerBuilder::pedigree);
    Rank = GEP(B, Rank, PedigreeBuilder::rank);
    B.CreateStore(B.CreateAdd(B.CreateLoad(Rank),
                  ConstantInt::get(Rank->getType()->getPointerElementType(),
                                   1)),
                  Rank);
    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::AlwaysInline);
  Fn->addFnAttr(Attribute::ReturnsTwice);
  //***INTEL
  // Special Intel-specific attribute for inliner.
  Fn->addFnAttr("INTEL_ALWAYS_INLINE");
  
//TODO?
//  llvm::NamedMDNode *SyncMetadata = M.getOrInsertNamedMetadata("cilk.sync");

//  SyncMetadata->addOperand(llvm::MDNode::get(Ctx, Fn));

  return Fn;
}

/// \brief Get or create a LLVM function to set worker to null value.
/// It is equivalent to the following C code
///
/// This is a utility function to ensure that __cilk_helper_epilogue
/// skips uninitialized stack frames.
///
/// void __cilk_reset_worker(__cilkrts_stack_frame *sf) {
///   sf->worker = 0;
/// }
///
static Function *GetCilkResetWorkerFn(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilk_reset_worker", M, Fn))
    return Fn;

  LLVMContext &Ctx = M.getContext();
  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  // sf->worker = 0;
  StoreField(B,
    Constant::getNullValue(TypeBuilder<__cilkrts_worker*, false>::get(Ctx)),
    SF, StackFrameBuilder::worker);

  B.CreateRetVoid();

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilkrts_enter_frame.
/// It is equivalent to the following C code
///
/// void __cilkrts_enter_frame_1(struct __cilkrts_stack_frame *sf)
/// {
///     struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
///     if (w == 0) { /* slow path, rare */
///         w = __cilkrts_bind_thread_1();
///         sf->flags = CILK_FRAME_LAST | CILK_FRAME_VERSION;
///     } else {
///         sf->flags = CILK_FRAME_VERSION;
///     }
///     sf->call_parent = w->current_stack_frame;
///     sf->worker = w;
///     /* sf->except_data is only valid when CILK_FRAME_EXCEPTING is set */
///     w->current_stack_frame = sf;
/// }
static Function *Get__cilkrts_enter_frame_1(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_enter_frame_1", M, Fn, Function::AvailableExternallyLinkage))
    return Fn;

  LLVMContext &Ctx = M.getContext();
  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "", Fn);
  BasicBlock *SlowPath = BasicBlock::Create(Ctx, "", Fn);
  BasicBlock *FastPath = BasicBlock::Create(Ctx, "", Fn);
  BasicBlock *Cont = BasicBlock::Create(Ctx, "", Fn);

  llvm::PointerType *WorkerPtrTy = TypeBuilder<__cilkrts_worker*, false>::get(Ctx);
  StructType *SFTy = StackFrameBuilder::get(Ctx);

  // Block  (Entry)
  CallInst *W = 0;
  {
    IRBuilder<> B(Entry);
    W = B.CreateCall(CILKRTS_FUNC(get_tls_worker, M));
    Value *Cond = B.CreateICmpEQ(W, ConstantPointerNull::get(WorkerPtrTy));
    B.CreateCondBr(Cond, SlowPath, FastPath);
  }
  // Block  (SlowPath)
  CallInst *Wslow = 0;
  {
    IRBuilder<> B(SlowPath);
    Wslow = B.CreateCall(CILKRTS_FUNC(bind_thread_1, M));
    llvm::Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);
    StoreField(B,
      ConstantInt::get(Ty, CILK_FRAME_LAST | CILK_FRAME_VERSION),
      SF, StackFrameBuilder::flags);
    B.CreateBr(Cont);
  }
  // Block  (FastPath)
  {
    IRBuilder<> B(FastPath);
    llvm::Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);
    StoreField(B,
      ConstantInt::get(Ty, CILK_FRAME_VERSION),
      SF, StackFrameBuilder::flags);
    B.CreateBr(Cont);
  }
  // Block  (Cont)
  {
    IRBuilder<> B(Cont);
    Value *Wfast = W;
    PHINode *W  = B.CreatePHI(WorkerPtrTy, 2);
    W->addIncoming(Wslow, SlowPath);
    W->addIncoming(Wfast, FastPath);

    StoreField(B,
      LoadField(B, W, WorkerBuilder::current_stack_frame),
      SF, StackFrameBuilder::call_parent);

    StoreField(B, W, SF, StackFrameBuilder::worker);
    StoreField(B, SF, W, WorkerBuilder::current_stack_frame);

    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilkrts_enter_frame_fast.
/// It is equivalent to the following C code
///
/// void __cilkrts_enter_frame_fast_1(struct __cilkrts_stack_frame *sf)
/// {
///     struct __cilkrts_worker *w = __cilkrts_get_tls_worker();
///     sf->flags = CILK_FRAME_VERSION;
///     sf->call_parent = w->current_stack_frame;
///     sf->worker = w;
///     /* sf->except_data is only valid when CILK_FRAME_EXCEPTING is set */
///     w->current_stack_frame = sf;
/// }
static Function *Get__cilkrts_enter_frame_fast_1(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilkrts_enter_frame_fast_1", M, Fn, Function::AvailableExternallyLinkage))
    return Fn;

  LLVMContext &Ctx = M.getContext();
  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "", Fn);

  IRBuilder<> B(Entry);
  Value *W = B.CreateCall(CILKRTS_FUNC(get_tls_worker, M));
  StructType *SFTy = StackFrameBuilder::get(Ctx);
  llvm::Type *Ty = SFTy->getElementType(StackFrameBuilder::flags);

  StoreField(B,
    ConstantInt::get(Ty, CILK_FRAME_VERSION),
    SF, StackFrameBuilder::flags);
  StoreField(B,
    LoadField(B, W, WorkerBuilder::current_stack_frame),
    SF, StackFrameBuilder::call_parent);
  StoreField(B, W, SF, StackFrameBuilder::worker);
  StoreField(B, SF, W, WorkerBuilder::current_stack_frame);

  B.CreateRetVoid();

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilk_parent_prologue.
/// It is equivalent to the following C code
///
/// void __cilk_parent_prologue(__cilkrts_stack_frame *sf) {
///   __cilkrts_enter_frame_1(sf);
/// }
static Function *GetCilkParentPrologue(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilk_parent_prologue", M, Fn))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  // __cilkrts_enter_frame_1(sf)
  B.CreateCall(CILKRTS_FUNC(enter_frame_1, M), SF);

  B.CreateRetVoid();

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilk_parent_epilogue.
/// It is equivalent to the following C code
///
/// void __cilk_parent_epilogue(__cilkrts_stack_frame *sf) {
///   __cilkrts_pop_frame(sf);
///   if (sf->flags != CILK_FRAME_VERSION)
///     __cilkrts_leave_frame(sf);
/// }
static Function *GetCilkParentEpilogue(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilk_parent_epilogue", M, Fn))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn),
             *B1 = BasicBlock::Create(Ctx, "", Fn),
             *Exit  = BasicBlock::Create(Ctx, "exit", Fn);

  // Entry
  {
    IRBuilder<> B(Entry);

    // __cilkrts_pop_frame(sf)
    B.CreateCall(CILKRTS_FUNC(pop_frame, M), SF);

    // if (sf->flags != CILK_FRAME_VERSION)
    Value *Flags = LoadField(B, SF, StackFrameBuilder::flags);
    Value *Cond = B.CreateICmpNE(Flags,
      ConstantInt::get(Flags->getType(), CILK_FRAME_VERSION));
    B.CreateCondBr(Cond, B1, Exit);
  }

  // B1
  {
    IRBuilder<> B(B1);

    // __cilkrts_leave_frame(sf);
    B.CreateCall(CILKRTS_FUNC(leave_frame, M), SF);
    B.CreateBr(Exit);
  }

  // Exit
  {
    IRBuilder<> B(Exit);
    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilk_helper_prologue.
/// It is equivalent to the following C code
///
/// void __cilk_helper_prologue(__cilkrts_stack_frame *sf) {
///   __cilkrts_enter_frame_fast_1(sf);
///   __cilkrts_detach(sf);
/// }
static llvm::Function *GetCilkHelperPrologue(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilk_helper_prologue", M, Fn))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  // __cilkrts_enter_frame_fast_1(sf);
  B.CreateCall(CILKRTS_FUNC(enter_frame_fast_1, M), SF);

  // __cilkrts_detach(sf);
  B.CreateCall(CILKRTS_FUNC(detach, M), SF);

  B.CreateRetVoid();

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

/// \brief Get or create a LLVM function for __cilk_helper_epilogue.
/// It is equivalent to the following C code
///
/// void __cilk_helper_epilogue(__cilkrts_stack_frame *sf) {
///   if (sf->worker) {
///     __cilkrts_pop_frame(sf);
///     __cilkrts_leave_frame(sf);
///   }
/// }
static llvm::Function *GetCilkHelperEpilogue(Module &M) {
  Function *Fn = 0;

  if (GetOrCreateFunction<cilk_func>("__cilk_helper_epilogue", M, Fn))
    return Fn;

  // If we get here we need to add the function body
  LLVMContext &Ctx = M.getContext();

  Function::arg_iterator args = Fn->arg_begin();
  Value *SF = &*args;

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  BasicBlock *Body = BasicBlock::Create(Ctx, "body", Fn);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "exit", Fn);

  // Entry
  {
    IRBuilder<> B(Entry);

    // if (sf->worker)
    Value *C = B.CreateIsNotNull(LoadField(B, SF, StackFrameBuilder::worker));
    B.CreateCondBr(C, Body, Exit);
  }

  // Body
  {
    IRBuilder<> B(Body);

    // __cilkrts_pop_frame(sf);
    B.CreateCall(CILKRTS_FUNC(pop_frame, M), SF);

    // __cilkrts_leave_frame(sf);
    B.CreateCall(CILKRTS_FUNC(leave_frame, M), SF);

    B.CreateBr(Exit);
  }

  // Exit
  {
    IRBuilder<> B(Exit);
    B.CreateRetVoid();
  }

  Fn->addFnAttr(Attribute::InlineHint);

  return Fn;
}

static const char *stack_frame_name = "__cilkrts_sf";

static llvm::Value *LookupStackFrame(Function &F) {
  return F.getValueSymbolTable().lookup(stack_frame_name);
}

/// \brief Create the __cilkrts_stack_frame for the spawning function.
static llvm::AllocaInst *CreateStackFrame(Function &F) {
  assert(!LookupStackFrame(F) && "already created the stack frame");

  llvm::LLVMContext &Ctx = F.getContext();
  llvm::Type *SFTy = StackFrameBuilder::get(Ctx);
  
	Instruction* I = F.getEntryBlock().getFirstNonPHIOrDbgOrLifetime();

	AllocaInst* SF = new AllocaInst(SFTy, /*size*/nullptr, 8, /*name*/stack_frame_name, /*insert before*/I);
	if( I == nullptr ) {
		F.getEntryBlock().getInstList().push_back( SF );
	}

  return SF;
}

static inline llvm::Value* GetOrInitStackFrame(Function& F, bool fast = true) {
  llvm::Value* V = LookupStackFrame(F);
  if( V ) return V;
  
  llvm::AllocaInst* alloc = CreateStackFrame(F);
  llvm::Value* args[1] = { alloc };
  llvm::Instruction* inst;
  if( fast ) {
    inst = CallInst::Create(CILKRTS_FUNC(enter_frame_fast_1, *F.getParent()), args, "" );
  } else {
    inst = CallInst::Create(CILKRTS_FUNC(enter_frame_1, *F.getParent()), args, "" );
  }
  inst->insertAfter(alloc);
  
  std::vector<ReturnInst*> rets;
    
	for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
		TerminatorInst* term = i->getTerminator();
		if( term == nullptr ) continue;
		if( ReturnInst* inst = llvm::dyn_cast<ReturnInst>(term) ) {
			rets.emplace_back( inst );
		} else continue;
	}
	
	assert( rets.size() > 0 );
	
	Instruction* retInst = nullptr;
	if( rets.size() > 1 ) {
	  //TODO check this
	  BasicBlock* retB = BasicBlock::Create( rets[0]->getContext(), "retMerge", rets[0]->getParent()->getParent() );
	  PHINode* PN = nullptr;
	  if( Value* V = rets[0]->getReturnValue() )
	    PN = PHINode::Create( V->getType(), rets.size(), "finalRet", retB );
	  for( auto& a : rets ) {
	    if( PN )
  	    PN->addIncoming( a->getReturnValue(), a->getParent() );
	    ReplaceInstWithInst( a, BranchInst::Create( retB ) );
	  }
	  retInst = ReturnInst::Create( rets[0]->getContext(), PN, retB );
	} else {
	  retInst = rets[0];
	}
	
	assert( retInst );
  CallInst::Create( GetCilkParentEpilogue( *F.getParent() ), args, "", retInst );
  return alloc;
}

/*
Type* getFrameType(LLVMContext& Context){
	return StackFrameBuilder::get(Context);
}

Value* getFrameCTX(Instruction& insertBefore){
	//todo will require gep / load
	llvm::Value* frame = getFrameFromFunction(*insertBefore.getParent()->getParent());
	return CastInst::CreatePointerCast( frame, Type::getInt8PtrTy(insertBefore.getContext()), "", &insertBefore );
}


Value* getFrameFlagsIsZero(Instruction& insertBefore){
	//todo will require gep / load
	LLVMContext& Context = insertBefore.getContext();
	Value* frame = getFrameFromFunction(*insertBefore.getParent()->getParent());
	Value *Idxs[] = {
		ConstantInt::get(Type::getInt32Ty(Context), 0),
		ConstantInt::get(Type::getInt32Ty(Context), 1)
	};
	Value* gep = GetElementPtrInst::Create( nullptr, frame, Idxs, "", &insertBefore );
	Value* load = new LoadInst( gep, "", &insertBefore );

	return new ICmpInst(&insertBefore, ICmpInst::ICMP_EQ, load, ConstantInt::get( load->getType(), 0 ) );
}

Type* getFramePointerType(LLVMContext& Context){
	return getFrameType(Context)->getPointerTo();
}
*/

//todo beef up
//use EmitCilkSetJmp

/*
Value* saveStack( Instruction& insertBefore ) {
	Function* setjumpF = Intrinsic::getDeclaration(insertBefore.getModule(), Intrinsic::eh_sjlj_setjmp);
	Value* args[1] = { getFrameCTX( insertBefore ) };
	return CallInst::Create(setjumpF, ArrayRef<Value*>(args), "", &insertBefore );
}
*/

//#####################################################################################################3333


static inline void createSync(SyncInst& inst){
	Function& F = *(inst.getParent()->getParent());
	Module* M = F.getParent();

	llvm::Value* args[1] = { LookupStackFrame( F ) };
	assert( args[0] && "sync used in function without frame!" );
	CallInst::Create( GetCilkSyncFn( *M ), args, "", /*insert before*/&inst );

	BranchInst* toReplace = BranchInst::Create( inst.getSuccessor(0) );

	ReplaceInstWithInst(&inst, toReplace);
}

static inline void replaceInList( llvm::Value* v, llvm::Value* replaceWith, SmallPtrSet<BasicBlock*,32>& blocks ) {
    AShrOperator::use_iterator UI = v->use_begin(), E = v->use_end();
    for (; UI != E;) {
      Use &U = *UI;
      ++UI;
      auto *Usr = dyn_cast<Instruction>(U.getUser());
      if (Usr && ( blocks.count(Usr->getParent()) ) ) {
        U.set(replaceWith);
      }
    }

}

//Returns true if success
static inline Function* extractDetachBodyToFunction(DetachInst& detach, llvm::CallInst** call = 0, llvm::Value* closure = 0, llvm::Value** closed = 0 ) {
  llvm::BasicBlock* detB = detach.getParent();
	Function& F = *(detB->getParent());
	Module* M = F.getParent();
	LLVMContext& Context = F.getContext();
	const DataLayout& DL = M->getDataLayout();

	BasicBlock* Spawned  = detach.getSuccessor(0);
	BasicBlock* Continue = detach.getSuccessor(1);

	SmallPtrSet<BasicBlock *, 32> functionPieces;
	SmallVector<BasicBlock *, 32> todo;
	todo.push_back(Spawned);

  SmallVector<BasicBlock*, 32 > reattachB;
	while( todo.size() > 0 ){
		BasicBlock* BB = todo.pop_back_val();
		// functionPieces.insert(BB);
		if (!functionPieces.insert(BB).second)
		  continue;

		TerminatorInst* term = BB->getTerminator();      
		if( term == nullptr ) return nullptr;
		if( ReattachInst* inst = llvm::dyn_cast<ReattachInst>(term) ) {
			//only analyze reattaches going to the same continuation
			if( inst->getSuccessor(0) != Continue ) continue;
			BranchInst* toReplace = BranchInst::Create( Continue );
			ReplaceInstWithInst(inst, toReplace);
      reattachB.push_back(BB);
			continue;
		} else if( DetachInst* inst = llvm::dyn_cast<DetachInst>(term) ) {
			assert( inst != &detach && "Found recursive detach!" );
			todo.emplace_back( inst->getSuccessor(0) );
			todo.emplace_back( inst->getSuccessor(1) );
			continue;
		} else if( SyncInst* inst = llvm::dyn_cast<SyncInst>(term) ) {
			//only sync inner elements, consider as branch
			todo.emplace_back( inst->getSuccessor(0) );
			continue;
		} else if( BranchInst* inst = llvm::dyn_cast<BranchInst>(term) ) {
			//only sync inner elements, consider as branch
			for( unsigned idx = 0, max = inst->getNumSuccessors(); idx < max; idx++ )
				todo.emplace_back( inst->getSuccessor(idx) );
			continue;
		} else {
			assert( 0 && "Detached block did not absolutely terminate in reattach");
			return nullptr;
		}
	}

	std::vector<BasicBlock*> blocks( functionPieces.begin(), functionPieces.end() );
	for( auto& a : blocks ){
	  if( a == Spawned ) {
			//assert only came from the detach
			for (pred_iterator PI = pred_begin(a), E = pred_end(a); PI != E; ++PI) {
				BasicBlock *Pred = *PI;
				if ( Pred == a ) continue;
				assert(Pred == detach.getParent() &&
				       "Block inside of detached context branched into from outside branch context from detach");
				// if( Pred != detach.getParent() ) {
				//   DEBUG(dbgs() << "Bad pred " << *Pred);
				// 	assert( 0 && "Block inside of detached context branched into from outside branch context from detach");
				// }
			}
		} else {
			for (pred_iterator PI = pred_begin(a), E = pred_end(a); PI != E; ++PI) {
				BasicBlock *Pred = *PI;
        //printf("block:%s pred %s count:%u\n", a->getName().str().c_str(), Pred->getName().str().c_str(), functionPieces.count(Pred) );
				assert(functionPieces.count(Pred) &&
				       "Block inside of detached context branched into from outside branch context");
				// if( functionPieces.find(Pred) == functionPieces.end() ) {
				// 	assert( 0 && "Block inside of detached context branched into from outside branch context");
				// }
			}
		}
	}

  PHINode *rstart = 0, *rend = 0;
  Instruction* inst = 0;
  PHINode *fake;
  Instruction* add = 0;
  if( closure ) {
      
    IRBuilder<> inspawn(Spawned->getFirstNonPHI());

    SetVector<Value*> Inputs, Outputs;
	  CodeExtractor extractor( ArrayRef<BasicBlock*>( blocks ), /*dominator tree -- todo? */ nullptr );
    extractor.findInputsOutputs(Inputs, Outputs);
    Inputs.insert(Outputs.begin(), Outputs.end());
    SmallVector<Type*,0> types;
    for(auto& a: Inputs) {
      if( a == closure ) continue;
        types.push_back(a->getType());
    };
    StructType* st = StructType::get(F.getContext(), types);
    IRBuilder<> builder(&detach);
    rend   = PHINode::Create( closure->getType(), 2, "", &detB->front() );
    rstart = PHINode::Create( closure->getType(), 2, "", &detB->front() );
    llvm::AllocaInst* alloc = builder.CreateAlloca(st);
    unsigned i=0;
    for(auto& a: Inputs) {
      if( a == closure ) continue;
      auto V = builder.CreateConstGEP2_32(st, alloc, 0U, i);
      builder.CreateStore(a, V);
      auto ld = inspawn.CreateLoad(inspawn.CreateConstGEP2_32(st, alloc, 0U, i));
      replaceInList( a, ld, functionPieces );
      i++;
    }
    assert( closed );
    *closed = alloc;
    // force it to be used, in right order
    add  = dyn_cast<Instruction>(inspawn.CreateAdd(rstart, rstart));
    assert( add );
    inst = dyn_cast<Instruction>(inspawn.CreateAdd(rend,   rend));
    assert( inst);
    BasicBlock* lp = Spawned->splitBasicBlock(inst);

    PHINode* idx = PHINode::Create( closure->getType(), 2, "", &Spawned->front() );
    idx->addIncoming( rstart, detB );

    fake = PHINode::Create( alloc->getType(), 2, "", &Spawned->front() );
    fake->addIncoming( alloc, detB );

    functionPieces.insert(lp);
    blocks.push_back(lp);

    BasicBlock* next = BasicBlock::Create( lp->getContext(), "next", &F );
    blocks.push_back(next);
    functionPieces.insert(next);

    for( auto a : reattachB ) {
      if( a == Spawned ) {
        a = lp;
      }
      ((BranchInst*) a->getTerminator() )->setSuccessor(0, next );
    }
    IRBuilder<> nextB(next);
    auto p1 = nextB.CreateAdd(idx, ConstantInt::get(idx->getType(), 1, false) );
    nextB.CreateCondBr( nextB.CreateICmpEQ( p1, rend ), Continue, Spawned );
    idx->addIncoming( p1, next );
    fake->addIncoming( alloc, next );

    replaceInList( closure, idx, functionPieces );
  }

	CodeExtractor extractor( ArrayRef<BasicBlock*>( blocks ), /*dominator tree -- todo? */ nullptr );
	assert( extractor.isEligible() && "Code not able to be extracted!" );


  if( closure ) {
    SetVector<Value*> Inputs, Outputs;
    extractor.findInputsOutputs(Inputs, Outputs);
    assert( Outputs.size() == 0 );
    assert( Inputs.size() == 3 );
    assert( Inputs[1] == rstart );
    assert( Inputs[2] == rend );
  }

	Function* extracted = extractor.extractCodeRegion();
	assert( extracted && "could not extract code" );


    TerminatorInst* bi = llvm::dyn_cast<TerminatorInst>(detB->getTerminator() );
    assert( bi );
    Spawned = (detach.getSuccessor(0) == Continue)?detach.getSuccessor(1):detach.getSuccessor(0);
    CallInst* cal = llvm::dyn_cast<CallInst>(Spawned->getFirstNonPHI());
    assert(cal);
    if( call ) *call = cal;


  if( closure ) {
    cal->eraseFromParent();
    rstart->eraseFromParent();
    rend->eraseFromParent();
    inst->eraseFromParent();
    add->eraseFromParent();
    fake->eraseFromParent();

  }

  	return extracted;
}

static inline bool makeFunctionDetachable( Function& extracted ) {
	Module* M = extracted.getParent();
	LLVMContext& Context = extracted.getContext();
	const DataLayout& DL = M->getDataLayout();
	/*
	   __cilkrts_stack_frame sf;
	   __cilkrts_enter_frame_fast(&sf);
	   __cilkrts_detach();
	   *x = f(y);
	*/

	llvm::Value* sf = CreateStackFrame( extracted );
	assert(sf);
	Value* args[1] = { sf };

	//TODO check difference between frame fast and frame fast 1
	Instruction* call = CallInst::Create(CILKRTS_FUNC(enter_frame_fast_1, *M), args, "", extracted.getEntryBlock().getTerminator() );
	// IRBuilder<> B(call);
	// // sf->worker = 0;
	// StoreField(B,
	// 	   Constant::getNullValue(TypeBuilder<__cilkrts_worker*, false>::get(M->getContext())),
	// 	   sf, StackFrameBuilder::worker);
	Instruction* call2 = CallInst::Create(CILKRTS_FUNC(detach, *M), args, "", extracted.getEntryBlock().getTerminator() );

	ReturnInst* ret = nullptr;
	for (Function::iterator i = extracted.begin(), e = extracted.end(); i != e; ++i) {
		BasicBlock* bb = &*i;
		TerminatorInst* term = bb->getTerminator();
		if( !term ) continue;
		if( ReturnInst* inst = llvm::dyn_cast<ReturnInst>( term ) ) {
			assert( ret == nullptr && "Multiple return" );
			ret = inst;
		}
	}
	assert( ret && "No return from extract function" );
	//TODO alow to work for functions with multiple returns

	/* 
	   __cilkrts_pop_frame(&sf);
	   if (sf->flags)
	   __cilkrts_leave_frame(&sf);
	*/
  //TODO WHY I
  auto PE = GetCilkParentEpilogue(*M);

  CallInst::Create(PE, args,"",ret);
  return true;
}

static inline bool createDetach(DetachInst& detach) {
  BasicBlock* detB = detach.getParent();
	Function& F = *(detB->getParent());
        
	BasicBlock* Spawned  = detach.getSuccessor(0);
	BasicBlock* Continue = detach.getSuccessor(1);

	Module* M = F.getParent();
	LLVMContext& Context = F.getContext();
	const DataLayout& DL = M->getDataLayout();

	//replace with branch to succesor
	//entry / cilk.spawn.savestate
    	Value *SF = GetOrInitStackFrame( F, /*isFast*/ false );
    	assert(SF && "null stack frame unexpected");

	Function* extracted = extractDetachBodyToFunction( detach );

  TerminatorInst* bi = llvm::dyn_cast<TerminatorInst>(detB->getTerminator() );
  assert( bi );
  Spawned = (detach.getSuccessor(0) == Continue)?detach.getSuccessor(1):detach.getSuccessor(0);

	assert( extracted && "could not extract detach body to function" );


	IRBuilder<> B(bi);
	  
    	// Need to save state before spawning
    	Value *C = EmitCilkSetJmp(B, SF, *M);
    	C = B.CreateICmpEQ(C, ConstantInt::get(C->getType(), 0));
    	B.CreateCondBr(C, Spawned, Continue);
    
    	detach.eraseFromParent();
 
        makeFunctionDetachable( *extracted );


	return true;
}

}  // end of cilk namespace
}  // end of llvm namespace

#endif
