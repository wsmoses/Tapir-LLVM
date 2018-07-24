/// Given a LLVM Type, compute its size in bytes,
static int computeSizeInBytes(const Type *T) {
  int bytes = T->getPrimitiveSizeInBits() / 8;
  if (bytes == 0)
    bytes = T->getScalarSizeInBits() / 8;
  return bytes;
}

/// Generate code for a GPU specific isl AST.
///
/// The GPUNodeBuilder augments the general existing IslNodeBuilder, which
/// generates code for general-purpose AST nodes, with special functionality
/// for generating GPU specific user nodes.
///
/// @see GPUNodeBuilder::createUser
class GPUNodeBuilder : public IslNodeBuilder {
public:
  GPUNodeBuilder(PollyIRBuilder &Builder, ScopAnnotator &Annotator,
                 const DataLayout &DL, LoopInfo &LI, ScalarEvolution &SE,
                 DominatorTree &DT, Scop &S, BasicBlock *StartBlock,
                 gpu_prog *Prog, GPURuntime Runtime, GPUArch Arch)
      : IslNodeBuilder(Builder, Annotator, DL, LI, SE, DT, S, StartBlock),
        Prog(Prog), Runtime(Runtime), Arch(Arch) {
    getExprBuilder().setIDToSAI(&IDToSAI);
  }

  /// Create after-run-time-check initialization code.
  void initializeAfterRTH();

  /// Finalize the generated scop.
  virtual void finalize();

  /// Track if the full build process was successful.
  ///
  /// This value is set to false, if throughout the build process an error
  /// occurred which prevents us from generating valid GPU code.
  bool BuildSuccessful = true;

  /// The maximal number of loops surrounding a sequential kernel.
  unsigned DeepestSequential = 0;

  /// The maximal number of loops surrounding a parallel kernel.
  unsigned DeepestParallel = 0;

  /// Return the name to set for the ptx_kernel.
  std::string getKernelFuncName(int Kernel_id);

private:
  /// A vector of array base pointers for which a new ScopArrayInfo was created.
  ///
  /// This vector is used to delete the ScopArrayInfo when it is not needed any
  /// more.
  std::vector<Value *> LocalArrays;

  /// A map from ScopArrays to their corresponding device allocations.
  std::map<ScopArrayInfo *, Value *> DeviceAllocations;

  /// The current GPU context.
  Value *GPUContext;

  /// The set of isl_ids allocated in the kernel
  std::vector<isl_id *> KernelIds;

  /// A module containing GPU code.
  ///
  /// This pointer is only set in case we are currently generating GPU code.
  std::unique_ptr<Module> GPUModule;

  /// The GPU program we generate code for.
  gpu_prog *Prog;

  /// The GPU Runtime implementation to use (OpenCL or CUDA).
  GPURuntime Runtime;

  /// The GPU Architecture to target.
  GPUArch Arch;

  /// Class to free isl_ids.
  class IslIdDeleter {
  public:
    void operator()(__isl_take isl_id *Id) { isl_id_free(Id); };
  };

  /// A set containing all isl_ids allocated in a GPU kernel.
  ///
  /// By releasing this set all isl_ids will be freed.
  std::set<std::unique_ptr<isl_id, IslIdDeleter>> KernelIDs;

  IslExprBuilder::IDToScopArrayInfoTy IDToSAI;

  /// Create code for user-defined AST nodes.
  ///
  /// These AST nodes can be of type:
  ///
  ///   - ScopStmt:      A computational statement (TODO)
  ///   - Kernel:        A GPU kernel call (TODO)
  ///   - Data-Transfer: A GPU <-> CPU data-transfer
  ///   - In-kernel synchronization
  ///   - In-kernel memory copy statement
  ///
  /// @param UserStmt The ast node to generate code for.
  virtual void createUser(__isl_take isl_ast_node *UserStmt);

  enum DataDirection { HOST_TO_DEVICE, DEVICE_TO_HOST };

  /// Create code for a data transfer statement
  ///
  /// @param TransferStmt The data transfer statement.
  /// @param Direction The direction in which to transfer data.
  void createDataTransfer(__isl_take isl_ast_node *TransferStmt,
                          enum DataDirection Direction);

  /// Find llvm::Values referenced in GPU kernel.
  ///
  /// @param Kernel The kernel to scan for llvm::Values
  ///
  /// @returns A pair, whose first element contains the set of values
  ///          referenced by the kernel, and whose second element contains the
  ///          set of functions referenced by the kernel. All functions in the
  ///          second set satisfy isValidFunctionInKernel.
  std::pair<SetVector<Value *>, SetVector<Function *>>
  getReferencesInKernel(ppcg_kernel *Kernel);

  /// Compute the sizes of the execution grid for a given kernel.
  ///
  /// @param Kernel The kernel to compute grid sizes for.
  ///
  /// @returns A tuple with grid sizes for X and Y dimension
  std::tuple<Value *, Value *> getGridSizes(ppcg_kernel *Kernel);

  /// Creates a array that can be sent to the kernel on the device using a
  /// host pointer. This is required for managed memory, when we directly send
  /// host pointers to the device.
  /// \note
  /// This is to be used only with managed memory
  Value *getOrCreateManagedDeviceArray(gpu_array_info *Array,
                                       ScopArrayInfo *ArrayInfo);

  /// Compute the sizes of the thread blocks for a given kernel.
  ///
  /// @param Kernel The kernel to compute thread block sizes for.
  ///
  /// @returns A tuple with thread block sizes for X, Y, and Z dimensions.
  std::tuple<Value *, Value *, Value *> getBlockSizes(ppcg_kernel *Kernel);

  /// Store a specific kernel launch parameter in the array of kernel launch
  /// parameters.
  ///
  /// @param Parameters The list of parameters in which to store.
  /// @param Param      The kernel launch parameter to store.
  /// @param Index      The index in the parameter list, at which to store the
  ///                   parameter.
  void insertStoreParameter(Instruction *Parameters, Instruction *Param,
                            int Index);

  /// Create kernel launch parameters.
  ///
  /// @param Kernel        The kernel to create parameters for.
  /// @param F             The kernel function that has been created.
  /// @param SubtreeValues The set of llvm::Values referenced by this kernel.
  ///
  /// @returns A stack allocated array with pointers to the parameter
  ///          values that are passed to the kernel.
  Value *createLaunchParameters(ppcg_kernel *Kernel, Function *F,
                                SetVector<Value *> SubtreeValues);

  /// Create declarations for kernel variable.
  ///
  /// This includes shared memory declarations.
  ///
  /// @param Kernel        The kernel definition to create variables for.
  /// @param FN            The function into which to generate the variables.
  void createKernelVariables(ppcg_kernel *Kernel, Function *FN);

  /// Add CUDA annotations to module.
  ///
  /// Add a set of CUDA annotations that declares the maximal block dimensions
  /// that will be used to execute the CUDA kernel. This allows the NVIDIA
  /// PTX compiler to bound the number of allocated registers to ensure the
  /// resulting kernel is known to run with up to as many block dimensions
  /// as specified here.
  ///
  /// @param M         The module to add the annotations to.
  /// @param BlockDimX The size of block dimension X.
  /// @param BlockDimY The size of block dimension Y.
  /// @param BlockDimZ The size of block dimension Z.
  void addCUDAAnnotations(Module *M, Value *BlockDimX, Value *BlockDimY,
                          Value *BlockDimZ);

  /// Create GPU kernel.
  ///
  /// Code generate the kernel described by @p KernelStmt.
  ///
  /// @param KernelStmt The ast node to generate kernel code for.
  void createKernel(__isl_take isl_ast_node *KernelStmt);

  /// Generate code that computes the size of an array.
  ///
  /// @param Array The array for which to compute a size.
  Value *getArraySize(gpu_array_info *Array);

  /// Generate code to compute the minimal offset at which an array is accessed.
  ///
  /// The offset of an array is the minimal array location accessed in a scop.
  ///
  /// Example:
  ///
  ///   for (long i = 0; i < 100; i++)
  ///     A[i + 42] += ...
  ///
  ///   getArrayOffset(A) results in 42.
  ///
  /// @param Array The array for which to compute the offset.
  /// @returns An llvm::Value that contains the offset of the array.
  Value *getArrayOffset(gpu_array_info *Array);

  /// Prepare the kernel arguments for kernel code generation
  ///
  /// @param Kernel The kernel to generate code for.
  /// @param FN     The function created for the kernel.
  void prepareKernelArguments(ppcg_kernel *Kernel, Function *FN);

  /// Create kernel function.
  ///
  /// Create a kernel function located in a newly created module that can serve
  /// as target for device code generation. Set the Builder to point to the
  /// start block of this newly created function.
  ///
  /// @param Kernel The kernel to generate code for.
  /// @param SubtreeValues The set of llvm::Values referenced by this kernel.
  /// @param SubtreeFunctions The set of llvm::Functions referenced by this
  ///                         kernel.
  void createKernelFunction(ppcg_kernel *Kernel,
                            SetVector<Value *> &SubtreeValues,
                            SetVector<Function *> &SubtreeFunctions);

  /// Create the declaration of a kernel function.
  ///
  /// The kernel function takes as arguments:
  ///
  ///   - One i8 pointer for each external array reference used in the kernel.
  ///   - Host iterators
  ///   - Parameters
  ///   - Other LLVM Value references (TODO)
  ///
  /// @param Kernel The kernel to generate the function declaration for.
  /// @param SubtreeValues The set of llvm::Values referenced by this kernel.
  ///
  /// @returns The newly declared function.
  Function *createKernelFunctionDecl(ppcg_kernel *Kernel,
                                     SetVector<Value *> &SubtreeValues);

  /// Insert intrinsic functions to obtain thread and block ids.
  ///
  /// @param The kernel to generate the intrinsic functions for.
  void insertKernelIntrinsics(ppcg_kernel *Kernel);

  /// Setup the creation of functions referenced by the GPU kernel.
  ///
  /// 1. Create new function declarations in GPUModule which are the same as
  /// SubtreeFunctions.
  ///
  /// 2. Populate IslNodeBuilder::ValueMap with mappings from
  /// old functions (that come from the original module) to new functions
  /// (that are created within GPUModule). That way, we generate references
  /// to the correct function (in GPUModule) in BlockGenerator.
  ///
  /// @see IslNodeBuilder::ValueMap
  /// @see BlockGenerator::GlobalMap
  /// @see BlockGenerator::getNewValue
  /// @see GPUNodeBuilder::getReferencesInKernel.
  ///
  /// @param SubtreeFunctions The set of llvm::Functions referenced by
  ///                         this kernel.
  void setupKernelSubtreeFunctions(SetVector<Function *> SubtreeFunctions);

  /// Create a global-to-shared or shared-to-global copy statement.
  ///
  /// @param CopyStmt The copy statement to generate code for
  void createKernelCopy(ppcg_kernel_stmt *CopyStmt);

  /// Create code for a ScopStmt called in @p Expr.
  ///
  /// @param Expr The expression containing the call.
  /// @param KernelStmt The kernel statement referenced in the call.
  void createScopStmt(isl_ast_expr *Expr, ppcg_kernel_stmt *KernelStmt);

  /// Create an in-kernel synchronization call.
  void createKernelSync();

  /// Create a PTX assembly string for the current GPU kernel.
  ///
  /// @returns A string containing the corresponding PTX assembly code.
  std::string createKernelASM();

  /// Remove references from the dominator tree to the kernel function @p F.
  ///
  /// @param F The function to remove references to.
  void clearDominators(Function *F);

  /// Remove references from scalar evolution to the kernel function @p F.
  ///
  /// @param F The function to remove references to.
  void clearScalarEvolution(Function *F);

  /// Remove references from loop info to the kernel function @p F.
  ///
  /// @param F The function to remove references to.
  void clearLoops(Function *F);

  /// Finalize the generation of the kernel function.
  ///
  /// Free the LLVM-IR module corresponding to the kernel and -- if requested --
  /// dump its IR to stderr.
  ///
  /// @returns The Assembly string of the kernel.
  std::string finalizeKernelFunction();

  /// Finalize the generation of the kernel arguments.
  ///
  /// This function ensures that not-read-only scalars used in a kernel are
  /// stored back to the global memory location they are backed with before
  /// the kernel terminates.
  ///
  /// @params Kernel The kernel to finalize kernel arguments for.
  void finalizeKernelArguments(ppcg_kernel *Kernel);

  /// Create code that allocates memory to store arrays on device.
  void allocateDeviceArrays();

  /// Free all allocated device arrays.
  void freeDeviceArrays();

  /// Create a call to initialize the GPU context.
  ///
  /// @returns A pointer to the newly initialized context.
  Value *createCallInitContext();

  /// Create a call to get the device pointer for a kernel allocation.
  ///
  /// @param Allocation The Polly GPU allocation
  ///
  /// @returns The device parameter corresponding to this allocation.
  Value *createCallGetDevicePtr(Value *Allocation);

  /// Create a call to free the GPU context.
  ///
  /// @param Context A pointer to an initialized GPU context.
  void createCallFreeContext(Value *Context);

  /// Create a call to allocate memory on the device.
  ///
  /// @param Size The size of memory to allocate
  ///
  /// @returns A pointer that identifies this allocation.
  Value *createCallAllocateMemoryForDevice(Value *Size);

  /// Create a call to free a device array.
  ///
  /// @param Array The device array to free.
  void createCallFreeDeviceMemory(Value *Array);

  /// Create a call to copy data from host to device.
  ///
  /// @param HostPtr A pointer to the host data that should be copied.
  /// @param DevicePtr A device pointer specifying the location to copy to.
  void createCallCopyFromHostToDevice(Value *HostPtr, Value *DevicePtr,
                                      Value *Size);

  /// Create a call to copy data from device to host.
  ///
  /// @param DevicePtr A pointer to the device data that should be copied.
  /// @param HostPtr A host pointer specifying the location to copy to.
  void createCallCopyFromDeviceToHost(Value *DevicePtr, Value *HostPtr,
                                      Value *Size);

  /// Create a call to synchronize Host & Device.
  /// \note
  /// This is to be used only with managed memory.
  void createCallSynchronizeDevice();

  /// Create a call to get a kernel from an assembly string.
  ///
  /// @param Buffer The string describing the kernel.
  /// @param Entry  The name of the kernel function to call.
  ///
  /// @returns A pointer to a kernel object
  Value *createCallGetKernel(Value *Buffer, Value *Entry);

  /// Create a call to free a GPU kernel.
  ///
  /// @param GPUKernel THe kernel to free.
  void createCallFreeKernel(Value *GPUKernel);

  /// Create a call to launch a GPU kernel.
  ///
  /// @param GPUKernel  The kernel to launch.
  /// @param GridDimX   The size of the first grid dimension.
  /// @param GridDimY   The size of the second grid dimension.
  /// @param GridBlockX The size of the first block dimension.
  /// @param GridBlockY The size of the second block dimension.
  /// @param GridBlockZ The size of the third block dimension.
  /// @param Parameters A pointer to an array that contains itself pointers to
  ///                   the parameter values passed for each kernel argument.
  void createCallLaunchKernel(Value *GPUKernel, Value *GridDimX,
                              Value *GridDimY, Value *BlockDimX,
                              Value *BlockDimY, Value *BlockDimZ,
                              Value *Parameters);
};

std::string GPUNodeBuilder::getKernelFuncName(int Kernel_id) {
  return "FUNC_" + S.getFunction().getName().str() + "_SCOP_" +
         std::to_string(S.getID()) + "_KERNEL_" + std::to_string(Kernel_id);
}

void GPUNodeBuilder::initializeAfterRTH() {
  BasicBlock *NewBB = SplitBlock(Builder.GetInsertBlock(),
                                 &*Builder.GetInsertPoint(), &DT, &LI);
  NewBB->setName("polly.acc.initialize");
  Builder.SetInsertPoint(&NewBB->front());

  GPUContext = createCallInitContext();

  if (!ManagedMemory)
    allocateDeviceArrays();
}

void GPUNodeBuilder::finalize() {
  if (!ManagedMemory)
    freeDeviceArrays();

  createCallFreeContext(GPUContext);
  IslNodeBuilder::finalize();
}

void GPUNodeBuilder::allocateDeviceArrays() {
  assert(!ManagedMemory && "Managed memory will directly send host pointers "
                           "to the kernel. There is no need for device arrays");
  isl_ast_build *Build = isl_ast_build_from_context(S.getContext());

  for (int i = 0; i < Prog->n_array; ++i) {
    gpu_array_info *Array = &Prog->array[i];
    auto *ScopArray = (ScopArrayInfo *)Array->user;
    std::string DevArrayName("p_dev_array_");
    DevArrayName.append(Array->name);

    Value *ArraySize = getArraySize(Array);
    Value *Offset = getArrayOffset(Array);
    if (Offset)
      ArraySize = Builder.CreateSub(
          ArraySize,
          Builder.CreateMul(Offset,
                            Builder.getInt64(ScopArray->getElemSizeInBytes())));
    Value *DevArray = createCallAllocateMemoryForDevice(ArraySize);
    DevArray->setName(DevArrayName);
    DeviceAllocations[ScopArray] = DevArray;
  }

  isl_ast_build_free(Build);
}

void GPUNodeBuilder::addCUDAAnnotations(Module *M, Value *BlockDimX,
                                        Value *BlockDimY, Value *BlockDimZ) {
  auto AnnotationNode = M->getOrInsertNamedMetadata("nvvm.annotations");

  for (auto &F : *M) {
    if (F.getCallingConv() != CallingConv::PTX_Kernel)
      continue;

    Value *V[] = {BlockDimX, BlockDimY, BlockDimZ};

    Metadata *Elements[] = {
        ValueAsMetadata::get(&F),   MDString::get(M->getContext(), "maxntidx"),
        ValueAsMetadata::get(V[0]), MDString::get(M->getContext(), "maxntidy"),
        ValueAsMetadata::get(V[1]), MDString::get(M->getContext(), "maxntidz"),
        ValueAsMetadata::get(V[2]),
    };
    MDNode *Node = MDNode::get(M->getContext(), Elements);
    AnnotationNode->addOperand(Node);
  }
}

void GPUNodeBuilder::freeDeviceArrays() {
  assert(!ManagedMemory && "Managed memory does not use device arrays");
  for (auto &Array : DeviceAllocations)
    createCallFreeDeviceMemory(Array.second);
}

Value *GPUNodeBuilder::createCallGetKernel(Value *Buffer, Value *Entry) {
  const char *Name = "polly_getKernel";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getInt8PtrTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return Builder.CreateCall(F, {Buffer, Entry});
}

Value *GPUNodeBuilder::createCallGetDevicePtr(Value *Allocation) {
  const char *Name = "polly_getDevicePtr";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getInt8PtrTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return Builder.CreateCall(F, {Allocation});
}

void GPUNodeBuilder::createCallLaunchKernel(Value *GPUKernel, Value *GridDimX,
                                            Value *GridDimY, Value *BlockDimX,
                                            Value *BlockDimY, Value *BlockDimZ,
                                            Value *Parameters) {
  const char *Name = "polly_launchKernel";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt32Ty());
    Args.push_back(Builder.getInt32Ty());
    Args.push_back(Builder.getInt32Ty());
    Args.push_back(Builder.getInt32Ty());
    Args.push_back(Builder.getInt32Ty());
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {GPUKernel, GridDimX, GridDimY, BlockDimX, BlockDimY,
                         BlockDimZ, Parameters});
}

void GPUNodeBuilder::createCallFreeKernel(Value *GPUKernel) {
  const char *Name = "polly_freeKernel";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {GPUKernel});
}

void GPUNodeBuilder::createCallFreeDeviceMemory(Value *Array) {
  assert(!ManagedMemory && "Managed memory does not allocate or free memory "
                           "for device");
  const char *Name = "polly_freeDeviceMemory";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {Array});
}

Value *GPUNodeBuilder::createCallAllocateMemoryForDevice(Value *Size) {
  assert(!ManagedMemory && "Managed memory does not allocate or free memory "
                           "for device");
  const char *Name = "polly_allocateMemoryForDevice";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt64Ty());
    FunctionType *Ty = FunctionType::get(Builder.getInt8PtrTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return Builder.CreateCall(F, {Size});
}

void GPUNodeBuilder::createCallCopyFromHostToDevice(Value *HostData,
                                                    Value *DeviceData,
                                                    Value *Size) {
  assert(!ManagedMemory && "Managed memory does not transfer memory between "
                           "device and host");
  const char *Name = "polly_copyFromHostToDevice";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt64Ty());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {HostData, DeviceData, Size});
}

void GPUNodeBuilder::createCallCopyFromDeviceToHost(Value *DeviceData,
                                                    Value *HostData,
                                                    Value *Size) {
  assert(!ManagedMemory && "Managed memory does not transfer memory between "
                           "device and host");
  const char *Name = "polly_copyFromDeviceToHost";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt8PtrTy());
    Args.push_back(Builder.getInt64Ty());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {DeviceData, HostData, Size});
}

void GPUNodeBuilder::createCallSynchronizeDevice() {
  assert(ManagedMemory && "explicit synchronization is only necessary for "
                          "managed memory");
  const char *Name = "polly_synchronizeDevice";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F);
}

Value *GPUNodeBuilder::createCallInitContext() {
  const char *Name;

  switch (Runtime) {
  case GPURuntime::CUDA:
    Name = "polly_initContextCUDA";
    break;
  case GPURuntime::OpenCL:
    Name = "polly_initContextCL";
    break;
  }

  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    FunctionType *Ty = FunctionType::get(Builder.getInt8PtrTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return Builder.CreateCall(F, {});
}

void GPUNodeBuilder::createCallFreeContext(Value *Context) {
  const char *Name = "polly_freeContext";
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    std::vector<Type *> Args;
    Args.push_back(Builder.getInt8PtrTy());
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Args, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {Context});
}

/// Check if one string is a prefix of another.
///
/// @param String The string in which to look for the prefix.
/// @param Prefix The prefix to look for.
static bool isPrefix(std::string String, std::string Prefix) {
  return String.find(Prefix) == 0;
}

Value *GPUNodeBuilder::getArraySize(gpu_array_info *Array) {
  isl_ast_build *Build = isl_ast_build_from_context(S.getContext());
  Value *ArraySize = ConstantInt::get(Builder.getInt64Ty(), Array->size);

  if (!gpu_array_is_scalar(Array)) {
    auto OffsetDimZero = isl_pw_aff_copy(Array->bound[0]);
    isl_ast_expr *Res = isl_ast_build_expr_from_pw_aff(Build, OffsetDimZero);

    for (unsigned int i = 1; i < Array->n_index; i++) {
      isl_pw_aff *Bound_I = isl_pw_aff_copy(Array->bound[i]);
      isl_ast_expr *Expr = isl_ast_build_expr_from_pw_aff(Build, Bound_I);
      Res = isl_ast_expr_mul(Res, Expr);
    }

    Value *NumElements = ExprBuilder.create(Res);
    if (NumElements->getType() != ArraySize->getType())
      NumElements = Builder.CreateSExt(NumElements, ArraySize->getType());
    ArraySize = Builder.CreateMul(ArraySize, NumElements);
  }
  isl_ast_build_free(Build);
  return ArraySize;
}

Value *GPUNodeBuilder::getArrayOffset(gpu_array_info *Array) {
  if (gpu_array_is_scalar(Array))
    return nullptr;

  isl_ast_build *Build = isl_ast_build_from_context(S.getContext());

  isl_set *Min = isl_set_lexmin(isl_set_copy(Array->extent));

  isl_set *ZeroSet = isl_set_universe(isl_set_get_space(Min));

  for (long i = 0; i < isl_set_dim(Min, isl_dim_set); i++)
    ZeroSet = isl_set_fix_si(ZeroSet, isl_dim_set, i, 0);

  if (isl_set_is_subset(Min, ZeroSet)) {
    isl_set_free(Min);
    isl_set_free(ZeroSet);
    isl_ast_build_free(Build);
    return nullptr;
  }
  isl_set_free(ZeroSet);

  isl_ast_expr *Result =
      isl_ast_expr_from_val(isl_val_int_from_si(isl_set_get_ctx(Min), 0));

  for (long i = 0; i < isl_set_dim(Min, isl_dim_set); i++) {
    if (i > 0) {
      isl_pw_aff *Bound_I = isl_pw_aff_copy(Array->bound[i - 1]);
      isl_ast_expr *BExpr = isl_ast_build_expr_from_pw_aff(Build, Bound_I);
      Result = isl_ast_expr_mul(Result, BExpr);
    }
    isl_pw_aff *DimMin = isl_set_dim_min(isl_set_copy(Min), i);
    isl_ast_expr *MExpr = isl_ast_build_expr_from_pw_aff(Build, DimMin);
    Result = isl_ast_expr_add(Result, MExpr);
  }

  Value *ResultValue = ExprBuilder.create(Result);
  isl_set_free(Min);
  isl_ast_build_free(Build);

  return ResultValue;
}

Value *GPUNodeBuilder::getOrCreateManagedDeviceArray(gpu_array_info *Array,
                                                     ScopArrayInfo *ArrayInfo) {

  assert(ManagedMemory && "Only used when you wish to get a host "
                          "pointer for sending data to the kernel, "
                          "with managed memory");
  std::map<ScopArrayInfo *, Value *>::iterator it;
  if ((it = DeviceAllocations.find(ArrayInfo)) != DeviceAllocations.end()) {
    return it->second;
  } else {
    Value *HostPtr;

    if (gpu_array_is_scalar(Array))
      HostPtr = BlockGen.getOrCreateAlloca(ArrayInfo);
    else
      HostPtr = ArrayInfo->getBasePtr();

    Value *Offset = getArrayOffset(Array);
    if (Offset) {
      HostPtr = Builder.CreatePointerCast(
          HostPtr, ArrayInfo->getElementType()->getPointerTo());
      HostPtr = Builder.CreateGEP(HostPtr, Offset);
    }

    HostPtr = Builder.CreatePointerCast(HostPtr, Builder.getInt8PtrTy());
    DeviceAllocations[ArrayInfo] = HostPtr;
    return HostPtr;
  }
}

void GPUNodeBuilder::createDataTransfer(__isl_take isl_ast_node *TransferStmt,
                                        enum DataDirection Direction) {
  assert(!ManagedMemory && "Managed memory needs no data transfers");
  isl_ast_expr *Expr = isl_ast_node_user_get_expr(TransferStmt);
  isl_ast_expr *Arg = isl_ast_expr_get_op_arg(Expr, 0);
  isl_id *Id = isl_ast_expr_get_id(Arg);
  auto Array = (gpu_array_info *)isl_id_get_user(Id);
  auto ScopArray = (ScopArrayInfo *)(Array->user);

  Value *Size = getArraySize(Array);
  Value *Offset = getArrayOffset(Array);
  Value *DevPtr = DeviceAllocations[ScopArray];

  Value *HostPtr;

  if (gpu_array_is_scalar(Array))
    HostPtr = BlockGen.getOrCreateAlloca(ScopArray);
  else
    HostPtr = ScopArray->getBasePtr();

  if (Offset) {
    HostPtr = Builder.CreatePointerCast(
        HostPtr, ScopArray->getElementType()->getPointerTo());
    HostPtr = Builder.CreateGEP(HostPtr, Offset);
  }

  HostPtr = Builder.CreatePointerCast(HostPtr, Builder.getInt8PtrTy());

  if (Offset) {
    Size = Builder.CreateSub(
        Size, Builder.CreateMul(
                  Offset, Builder.getInt64(ScopArray->getElemSizeInBytes())));
  }

  if (Direction == HOST_TO_DEVICE)
    createCallCopyFromHostToDevice(HostPtr, DevPtr, Size);
  else
    createCallCopyFromDeviceToHost(DevPtr, HostPtr, Size);

  isl_id_free(Id);
  isl_ast_expr_free(Arg);
  isl_ast_expr_free(Expr);
  isl_ast_node_free(TransferStmt);
}

void GPUNodeBuilder::createUser(__isl_take isl_ast_node *UserStmt) {
  isl_ast_expr *Expr = isl_ast_node_user_get_expr(UserStmt);
  isl_ast_expr *StmtExpr = isl_ast_expr_get_op_arg(Expr, 0);
  isl_id *Id = isl_ast_expr_get_id(StmtExpr);
  isl_id_free(Id);
  isl_ast_expr_free(StmtExpr);

  const char *Str = isl_id_get_name(Id);
  if (!strcmp(Str, "kernel")) {
    createKernel(UserStmt);
    isl_ast_expr_free(Expr);
    return;
  }

  if (isPrefix(Str, "to_device")) {
    if (!ManagedMemory)
      createDataTransfer(UserStmt, HOST_TO_DEVICE);
    else
      isl_ast_node_free(UserStmt);

    isl_ast_expr_free(Expr);
    return;
  }

  if (isPrefix(Str, "from_device")) {
    if (!ManagedMemory) {
      createDataTransfer(UserStmt, DEVICE_TO_HOST);
    } else {
      createCallSynchronizeDevice();
      isl_ast_node_free(UserStmt);
    }
    isl_ast_expr_free(Expr);
    return;
  }

  isl_id *Anno = isl_ast_node_get_annotation(UserStmt);
  struct ppcg_kernel_stmt *KernelStmt =
      (struct ppcg_kernel_stmt *)isl_id_get_user(Anno);
  isl_id_free(Anno);

  switch (KernelStmt->type) {
  case ppcg_kernel_domain:
    createScopStmt(Expr, KernelStmt);
    isl_ast_node_free(UserStmt);
    return;
  case ppcg_kernel_copy:
    createKernelCopy(KernelStmt);
    isl_ast_expr_free(Expr);
    isl_ast_node_free(UserStmt);
    return;
  case ppcg_kernel_sync:
    createKernelSync();
    isl_ast_expr_free(Expr);
    isl_ast_node_free(UserStmt);
    return;
  }

  isl_ast_expr_free(Expr);
  isl_ast_node_free(UserStmt);
  return;
}
void GPUNodeBuilder::createKernelCopy(ppcg_kernel_stmt *KernelStmt) {
  isl_ast_expr *LocalIndex = isl_ast_expr_copy(KernelStmt->u.c.local_index);
  LocalIndex = isl_ast_expr_address_of(LocalIndex);
  Value *LocalAddr = ExprBuilder.create(LocalIndex);
  isl_ast_expr *Index = isl_ast_expr_copy(KernelStmt->u.c.index);
  Index = isl_ast_expr_address_of(Index);
  Value *GlobalAddr = ExprBuilder.create(Index);

  if (KernelStmt->u.c.read) {
    LoadInst *Load = Builder.CreateLoad(GlobalAddr, "shared.read");
    Builder.CreateStore(Load, LocalAddr);
  } else {
    LoadInst *Load = Builder.CreateLoad(LocalAddr, "shared.write");
    Builder.CreateStore(Load, GlobalAddr);
  }
}

void GPUNodeBuilder::createScopStmt(isl_ast_expr *Expr,
                                    ppcg_kernel_stmt *KernelStmt) {
  auto Stmt = (ScopStmt *)KernelStmt->u.d.stmt->stmt;
  isl_id_to_ast_expr *Indexes = KernelStmt->u.d.ref2expr;

  LoopToScevMapT LTS;
  LTS.insert(OutsideLoopIterations.begin(), OutsideLoopIterations.end());

  createSubstitutions(Expr, Stmt, LTS);

  if (Stmt->isBlockStmt())
    BlockGen.copyStmt(*Stmt, LTS, Indexes);
  else
    RegionGen.copyStmt(*Stmt, LTS, Indexes);
}

void GPUNodeBuilder::createKernelSync() {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();

  Function *Sync;

  switch (Arch) {
  case GPUArch::NVPTX64:
    Sync = Intrinsic::getDeclaration(M, Intrinsic::nvvm_barrier0);
    break;
  }

  Builder.CreateCall(Sync, {});
}

/// Collect llvm::Values referenced from @p Node
///
/// This function only applies to isl_ast_nodes that are user_nodes referring
/// to a ScopStmt. All other node types are ignore.
///
/// @param Node The node to collect references for.
/// @param User A user pointer used as storage for the data that is collected.
///
/// @returns isl_bool_true if data could be collected successfully.
isl_bool collectReferencesInGPUStmt(__isl_keep isl_ast_node *Node, void *User) {
  if (isl_ast_node_get_type(Node) != isl_ast_node_user)
    return isl_bool_true;

  isl_ast_expr *Expr = isl_ast_node_user_get_expr(Node);
  isl_ast_expr *StmtExpr = isl_ast_expr_get_op_arg(Expr, 0);
  isl_id *Id = isl_ast_expr_get_id(StmtExpr);
  const char *Str = isl_id_get_name(Id);
  isl_id_free(Id);
  isl_ast_expr_free(StmtExpr);
  isl_ast_expr_free(Expr);

  if (!isPrefix(Str, "Stmt"))
    return isl_bool_true;

  Id = isl_ast_node_get_annotation(Node);
  auto *KernelStmt = (ppcg_kernel_stmt *)isl_id_get_user(Id);
  auto Stmt = (ScopStmt *)KernelStmt->u.d.stmt->stmt;
  isl_id_free(Id);

  addReferencesFromStmt(Stmt, User, false /* CreateScalarRefs */);

  return isl_bool_true;
}

/// Do not take `Function` as a subtree value.
///
/// We try to take the reference of all subtree values and pass them along
/// to the kernel from the host. Taking an address of any function and
/// trying to pass along is nonsensical. Only allow `Value`s that are not
/// `Function`s.
static bool isValidSubtreeValue(llvm::Value *V) { return !isa<Function>(V); }

/// Return `Function`s from `RawSubtreeValues`.
static SetVector<Function *>
getFunctionsFromRawSubtreeValues(SetVector<Value *> RawSubtreeValues) {
  SetVector<Function *> SubtreeFunctions;
  for (Value *It : RawSubtreeValues) {
    Function *F = dyn_cast<Function>(It);
    if (F) {
      assert(isValidFunctionInKernel(F) && "Code should have bailed out by "
                                           "this point if an invalid function "
                                           "were present in a kernel.");
      SubtreeFunctions.insert(F);
    }
  }
  return SubtreeFunctions;
}

std::pair<SetVector<Value *>, SetVector<Function *>>
GPUNodeBuilder::getReferencesInKernel(ppcg_kernel *Kernel) {
  SetVector<Value *> SubtreeValues;
  SetVector<const SCEV *> SCEVs;
  SetVector<const Loop *> Loops;
  SubtreeReferences References = {
      LI, SE, S, ValueMap, SubtreeValues, SCEVs, getBlockGenerator()};

  for (const auto &I : IDToValue)
    SubtreeValues.insert(I.second);

  isl_ast_node_foreach_descendant_top_down(
      Kernel->tree, collectReferencesInGPUStmt, &References);

  for (const SCEV *Expr : SCEVs)
    findValues(Expr, SE, SubtreeValues);

  for (auto &SAI : S.arrays())
    SubtreeValues.remove(SAI->getBasePtr());

  isl_space *Space = S.getParamSpace();
  for (long i = 0; i < isl_space_dim(Space, isl_dim_param); i++) {
    isl_id *Id = isl_space_get_dim_id(Space, isl_dim_param, i);
    assert(IDToValue.count(Id));
    Value *Val = IDToValue[Id];
    SubtreeValues.remove(Val);
    isl_id_free(Id);
  }
  isl_space_free(Space);

  for (long i = 0; i < isl_space_dim(Kernel->space, isl_dim_set); i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_set, i);
    assert(IDToValue.count(Id));
    Value *Val = IDToValue[Id];
    SubtreeValues.remove(Val);
    isl_id_free(Id);
  }

  // Note: { ValidSubtreeValues, ValidSubtreeFunctions } partitions
  // SubtreeValues. This is important, because we should not lose any
  // SubtreeValues in the process of constructing the
  // "ValidSubtree{Values, Functions} sets. Nor should the set
  // ValidSubtree{Values, Functions} have any common element.
  auto ValidSubtreeValuesIt =
      make_filter_range(SubtreeValues, isValidSubtreeValue);
  SetVector<Value *> ValidSubtreeValues(ValidSubtreeValuesIt.begin(),
                                        ValidSubtreeValuesIt.end());
  SetVector<Function *> ValidSubtreeFunctions(
      getFunctionsFromRawSubtreeValues(SubtreeValues));

  // @see IslNodeBuilder::getReferencesInSubtree
  SetVector<Value *> ReplacedValues;
  for (Value *V : ValidSubtreeValues) {
    auto It = ValueMap.find(V);
    if (It == ValueMap.end())
      ReplacedValues.insert(V);
    else
      ReplacedValues.insert(It->second);
  }
  return std::make_pair(ReplacedValues, ValidSubtreeFunctions);
}

void GPUNodeBuilder::clearDominators(Function *F) {
  DomTreeNode *N = DT.getNode(&F->getEntryBlock());
  std::vector<BasicBlock *> Nodes;
  for (po_iterator<DomTreeNode *> I = po_begin(N), E = po_end(N); I != E; ++I)
    Nodes.push_back(I->getBlock());

  for (BasicBlock *BB : Nodes)
    DT.eraseNode(BB);
}

void GPUNodeBuilder::clearScalarEvolution(Function *F) {
  for (BasicBlock &BB : *F) {
    Loop *L = LI.getLoopFor(&BB);
    if (L)
      SE.forgetLoop(L);
  }
}

void GPUNodeBuilder::clearLoops(Function *F) {
  for (BasicBlock &BB : *F) {
    Loop *L = LI.getLoopFor(&BB);
    if (L)
      SE.forgetLoop(L);
    LI.removeBlock(&BB);
  }
}

std::tuple<Value *, Value *> GPUNodeBuilder::getGridSizes(ppcg_kernel *Kernel) {
  std::vector<Value *> Sizes;
  isl_ast_build *Context = isl_ast_build_from_context(S.getContext());

  for (long i = 0; i < Kernel->n_grid; i++) {
    isl_pw_aff *Size = isl_multi_pw_aff_get_pw_aff(Kernel->grid_size, i);
    isl_ast_expr *GridSize = isl_ast_build_expr_from_pw_aff(Context, Size);
    Value *Res = ExprBuilder.create(GridSize);
    Res = Builder.CreateTrunc(Res, Builder.getInt32Ty());
    Sizes.push_back(Res);
  }
  isl_ast_build_free(Context);

  for (long i = Kernel->n_grid; i < 3; i++)
    Sizes.push_back(ConstantInt::get(Builder.getInt32Ty(), 1));

  return std::make_tuple(Sizes[0], Sizes[1]);
}

std::tuple<Value *, Value *, Value *>
GPUNodeBuilder::getBlockSizes(ppcg_kernel *Kernel) {
  std::vector<Value *> Sizes;

  for (long i = 0; i < Kernel->n_block; i++) {
    Value *Res = ConstantInt::get(Builder.getInt32Ty(), Kernel->block_dim[i]);
    Sizes.push_back(Res);
  }

  for (long i = Kernel->n_block; i < 3; i++)
    Sizes.push_back(ConstantInt::get(Builder.getInt32Ty(), 1));

  return std::make_tuple(Sizes[0], Sizes[1], Sizes[2]);
}

void GPUNodeBuilder::insertStoreParameter(Instruction *Parameters,
                                          Instruction *Param, int Index) {
  Value *Slot = Builder.CreateGEP(
      Parameters, {Builder.getInt64(0), Builder.getInt64(Index)});
  Value *ParamTyped = Builder.CreatePointerCast(Param, Builder.getInt8PtrTy());
  Builder.CreateStore(ParamTyped, Slot);
}

Value *
GPUNodeBuilder::createLaunchParameters(ppcg_kernel *Kernel, Function *F,
                                       SetVector<Value *> SubtreeValues) {
  const int NumArgs = F->arg_size();
  std::vector<int> ArgSizes(NumArgs);

  Type *ArrayTy = ArrayType::get(Builder.getInt8PtrTy(), 2 * NumArgs);

  BasicBlock *EntryBlock =
      &Builder.GetInsertBlock()->getParent()->getEntryBlock();
  auto AddressSpace = F->getParent()->getDataLayout().getAllocaAddrSpace();
  std::string Launch = "polly_launch_" + std::to_string(Kernel->id);
  Instruction *Parameters = new AllocaInst(
      ArrayTy, AddressSpace, Launch + "_params", EntryBlock->getTerminator());

  int Index = 0;
  for (long i = 0; i < Prog->n_array; i++) {
    if (!ppcg_kernel_requires_array_argument(Kernel, i))
      continue;

    isl_id *Id = isl_space_get_tuple_id(Prog->array[i].space, isl_dim_set);
    const ScopArrayInfo *SAI = ScopArrayInfo::getFromId(Id);

    ArgSizes[Index] = SAI->getElemSizeInBytes();

    Value *DevArray = nullptr;
    if (ManagedMemory) {
      DevArray = getOrCreateManagedDeviceArray(
          &Prog->array[i], const_cast<ScopArrayInfo *>(SAI));
    } else {
      DevArray = DeviceAllocations[const_cast<ScopArrayInfo *>(SAI)];
      DevArray = createCallGetDevicePtr(DevArray);
    }
    assert(DevArray != nullptr && "Array to be offloaded to device not "
                                  "initialized");
    Value *Offset = getArrayOffset(&Prog->array[i]);

    if (Offset) {
      DevArray = Builder.CreatePointerCast(
          DevArray, SAI->getElementType()->getPointerTo());
      DevArray = Builder.CreateGEP(DevArray, Builder.CreateNeg(Offset));
      DevArray = Builder.CreatePointerCast(DevArray, Builder.getInt8PtrTy());
    }
    Value *Slot = Builder.CreateGEP(
        Parameters, {Builder.getInt64(0), Builder.getInt64(Index)});

    if (gpu_array_is_read_only_scalar(&Prog->array[i])) {
      Value *ValPtr = nullptr;
      if (ManagedMemory)
        ValPtr = DevArray;
      else
        ValPtr = BlockGen.getOrCreateAlloca(SAI);

      assert(ValPtr != nullptr && "ValPtr that should point to a valid object"
                                  " to be stored into Parameters");
      Value *ValPtrCast =
          Builder.CreatePointerCast(ValPtr, Builder.getInt8PtrTy());
      Builder.CreateStore(ValPtrCast, Slot);
    } else {
      Instruction *Param =
          new AllocaInst(Builder.getInt8PtrTy(), AddressSpace,
                         Launch + "_param_" + std::to_string(Index),
                         EntryBlock->getTerminator());
      Builder.CreateStore(DevArray, Param);
      Value *ParamTyped =
          Builder.CreatePointerCast(Param, Builder.getInt8PtrTy());
      Builder.CreateStore(ParamTyped, Slot);
    }
    Index++;
  }

  int NumHostIters = isl_space_dim(Kernel->space, isl_dim_set);

  for (long i = 0; i < NumHostIters; i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_set, i);
    Value *Val = IDToValue[Id];
    isl_id_free(Id);

    ArgSizes[Index] = computeSizeInBytes(Val->getType());

    Instruction *Param =
        new AllocaInst(Val->getType(), AddressSpace,
                       Launch + "_param_" + std::to_string(Index),
                       EntryBlock->getTerminator());
    Builder.CreateStore(Val, Param);
    insertStoreParameter(Parameters, Param, Index);
    Index++;
  }

  int NumVars = isl_space_dim(Kernel->space, isl_dim_param);

  for (long i = 0; i < NumVars; i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_param, i);
    Value *Val = IDToValue[Id];
    if (ValueMap.count(Val))
      Val = ValueMap[Val];
    isl_id_free(Id);

    ArgSizes[Index] = computeSizeInBytes(Val->getType());

    Instruction *Param =
        new AllocaInst(Val->getType(), AddressSpace,
                       Launch + "_param_" + std::to_string(Index),
                       EntryBlock->getTerminator());
    Builder.CreateStore(Val, Param);
    insertStoreParameter(Parameters, Param, Index);
    Index++;
  }

  for (auto Val : SubtreeValues) {
    ArgSizes[Index] = computeSizeInBytes(Val->getType());

    Instruction *Param =
        new AllocaInst(Val->getType(), AddressSpace,
                       Launch + "_param_" + std::to_string(Index),
                       EntryBlock->getTerminator());
    Builder.CreateStore(Val, Param);
    insertStoreParameter(Parameters, Param, Index);
    Index++;
  }

  for (int i = 0; i < NumArgs; i++) {
    Value *Val = ConstantInt::get(Builder.getInt32Ty(), ArgSizes[i]);
    Instruction *Param =
        new AllocaInst(Builder.getInt32Ty(), AddressSpace,
                       Launch + "_param_size_" + std::to_string(i),
                       EntryBlock->getTerminator());
    Builder.CreateStore(Val, Param);
    insertStoreParameter(Parameters, Param, Index);
    Index++;
  }

  auto Location = EntryBlock->getTerminator();
  return new BitCastInst(Parameters, Builder.getInt8PtrTy(),
                         Launch + "_params_i8ptr", Location);
}

void GPUNodeBuilder::setupKernelSubtreeFunctions(
    SetVector<Function *> SubtreeFunctions) {
  for (auto Fn : SubtreeFunctions) {
    const std::string ClonedFnName = Fn->getName();
    Function *Clone = GPUModule->getFunction(ClonedFnName);
    if (!Clone)
      Clone =
          Function::Create(Fn->getFunctionType(), GlobalValue::ExternalLinkage,
                           ClonedFnName, GPUModule.get());
    assert(Clone && "Expected cloned function to be initialized.");
    assert(ValueMap.find(Fn) == ValueMap.end() &&
           "Fn already present in ValueMap");
    ValueMap[Fn] = Clone;
  }
}
void GPUNodeBuilder::createKernel(__isl_take isl_ast_node *KernelStmt) {
  isl_id *Id = isl_ast_node_get_annotation(KernelStmt);
  ppcg_kernel *Kernel = (ppcg_kernel *)isl_id_get_user(Id);
  isl_id_free(Id);
  isl_ast_node_free(KernelStmt);

  if (Kernel->n_grid > 1)
    DeepestParallel =
        std::max(DeepestParallel, isl_space_dim(Kernel->space, isl_dim_set));
  else
    DeepestSequential =
        std::max(DeepestSequential, isl_space_dim(Kernel->space, isl_dim_set));

  Value *BlockDimX, *BlockDimY, *BlockDimZ;
  std::tie(BlockDimX, BlockDimY, BlockDimZ) = getBlockSizes(Kernel);

  SetVector<Value *> SubtreeValues;
  SetVector<Function *> SubtreeFunctions;
  std::tie(SubtreeValues, SubtreeFunctions) = getReferencesInKernel(Kernel);

  assert(Kernel->tree && "Device AST of kernel node is empty");

  Instruction &HostInsertPoint = *Builder.GetInsertPoint();
  IslExprBuilder::IDToValueTy HostIDs = IDToValue;
  ValueMapT HostValueMap = ValueMap;
  BlockGenerator::AllocaMapTy HostScalarMap = ScalarMap;
  ScalarMap.clear();

  SetVector<const Loop *> Loops;

  // Create for all loops we depend on values that contain the current loop
  // iteration. These values are necessary to generate code for SCEVs that
  // depend on such loops. As a result we need to pass them to the subfunction.
  for (const Loop *L : Loops) {
    const SCEV *OuterLIV = SE.getAddRecExpr(SE.getUnknown(Builder.getInt64(0)),
                                            SE.getUnknown(Builder.getInt64(1)),
                                            L, SCEV::FlagAnyWrap);
    Value *V = generateSCEV(OuterLIV);
    OutsideLoopIterations[L] = SE.getUnknown(V);
    SubtreeValues.insert(V);
  }

  createKernelFunction(Kernel, SubtreeValues, SubtreeFunctions);
  setupKernelSubtreeFunctions(SubtreeFunctions);

  create(isl_ast_node_copy(Kernel->tree));

  finalizeKernelArguments(Kernel);
  Function *F = Builder.GetInsertBlock()->getParent();
  addCUDAAnnotations(F->getParent(), BlockDimX, BlockDimY, BlockDimZ);
  clearDominators(F);
  clearScalarEvolution(F);
  clearLoops(F);

  IDToValue = HostIDs;

  ValueMap = std::move(HostValueMap);
  ScalarMap = std::move(HostScalarMap);
  EscapeMap.clear();
  IDToSAI.clear();
  Annotator.resetAlternativeAliasBases();
  for (auto &BasePtr : LocalArrays)
    S.invalidateScopArrayInfo(BasePtr, MemoryKind::Array);
  LocalArrays.clear();

  std::string ASMString = finalizeKernelFunction();
  Builder.SetInsertPoint(&HostInsertPoint);
  Value *Parameters = createLaunchParameters(Kernel, F, SubtreeValues);

  std::string Name = getKernelFuncName(Kernel->id);
  Value *KernelString = Builder.CreateGlobalStringPtr(ASMString, Name);
  Value *NameString = Builder.CreateGlobalStringPtr(Name, Name + "_name");
  Value *GPUKernel = createCallGetKernel(KernelString, NameString);

  Value *GridDimX, *GridDimY;
  std::tie(GridDimX, GridDimY) = getGridSizes(Kernel);

  createCallLaunchKernel(GPUKernel, GridDimX, GridDimY, BlockDimX, BlockDimY,
                         BlockDimZ, Parameters);
  createCallFreeKernel(GPUKernel);

  for (auto Id : KernelIds)
    isl_id_free(Id);

  KernelIds.clear();
}

/// Compute the DataLayout string for the NVPTX backend.
///
/// @param is64Bit Are we looking for a 64 bit architecture?
static std::string computeNVPTXDataLayout(bool is64Bit) {
  std::string Ret = "";

  if (!is64Bit) {
    Ret += "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:"
           "64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:"
           "64-v128:128:128-n16:32:64";
  } else {
    Ret += "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:"
           "64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:"
           "64-v128:128:128-n16:32:64";
  }

  return Ret;
}

Function *
GPUNodeBuilder::createKernelFunctionDecl(ppcg_kernel *Kernel,
                                         SetVector<Value *> &SubtreeValues) {
  std::vector<Type *> Args;
  std::string Identifier = getKernelFuncName(Kernel->id);

  for (long i = 0; i < Prog->n_array; i++) {
    if (!ppcg_kernel_requires_array_argument(Kernel, i))
      continue;

    if (gpu_array_is_read_only_scalar(&Prog->array[i])) {
      isl_id *Id = isl_space_get_tuple_id(Prog->array[i].space, isl_dim_set);
      const ScopArrayInfo *SAI = ScopArrayInfo::getFromId(Id);
      Args.push_back(SAI->getElementType());
    } else {
      static const int UseGlobalMemory = 1;
      Args.push_back(Builder.getInt8PtrTy(UseGlobalMemory));
    }
  }

  int NumHostIters = isl_space_dim(Kernel->space, isl_dim_set);

  for (long i = 0; i < NumHostIters; i++)
    Args.push_back(Builder.getInt64Ty());

  int NumVars = isl_space_dim(Kernel->space, isl_dim_param);

  for (long i = 0; i < NumVars; i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_param, i);
    Value *Val = IDToValue[Id];
    isl_id_free(Id);
    Args.push_back(Val->getType());
  }

  for (auto *V : SubtreeValues)
    Args.push_back(V->getType());

  auto *FT = FunctionType::get(Builder.getVoidTy(), Args, false);
  auto *FN = Function::Create(FT, Function::ExternalLinkage, Identifier,
                              GPUModule.get());

  switch (Arch) {
  case GPUArch::NVPTX64:
    FN->setCallingConv(CallingConv::PTX_Kernel);
    break;
  }

  auto Arg = FN->arg_begin();
  for (long i = 0; i < Kernel->n_array; i++) {
    if (!ppcg_kernel_requires_array_argument(Kernel, i))
      continue;

    Arg->setName(Kernel->array[i].array->name);

    isl_id *Id = isl_space_get_tuple_id(Prog->array[i].space, isl_dim_set);
    const ScopArrayInfo *SAI = ScopArrayInfo::getFromId(isl_id_copy(Id));
    Type *EleTy = SAI->getElementType();
    Value *Val = &*Arg;
    SmallVector<const SCEV *, 4> Sizes;
    isl_ast_build *Build =
        isl_ast_build_from_context(isl_set_copy(Prog->context));
    Sizes.push_back(nullptr);
    for (long j = 1; j < Kernel->array[i].array->n_index; j++) {
      isl_ast_expr *DimSize = isl_ast_build_expr_from_pw_aff(
          Build, isl_pw_aff_copy(Kernel->array[i].array->bound[j]));
      auto V = ExprBuilder.create(DimSize);
      Sizes.push_back(SE.getSCEV(V));
    }
    const ScopArrayInfo *SAIRep =
        S.getOrCreateScopArrayInfo(Val, EleTy, Sizes, MemoryKind::Array);
    LocalArrays.push_back(Val);

    isl_ast_build_free(Build);
    KernelIds.push_back(Id);
    IDToSAI[Id] = SAIRep;
    Arg++;
  }

  for (long i = 0; i < NumHostIters; i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_set, i);
    Arg->setName(isl_id_get_name(Id));
    IDToValue[Id] = &*Arg;
    KernelIDs.insert(std::unique_ptr<isl_id, IslIdDeleter>(Id));
    Arg++;
  }

  for (long i = 0; i < NumVars; i++) {
    isl_id *Id = isl_space_get_dim_id(Kernel->space, isl_dim_param, i);
    Arg->setName(isl_id_get_name(Id));
    Value *Val = IDToValue[Id];
    ValueMap[Val] = &*Arg;
    IDToValue[Id] = &*Arg;
    KernelIDs.insert(std::unique_ptr<isl_id, IslIdDeleter>(Id));
    Arg++;
  }

  for (auto *V : SubtreeValues) {
    Arg->setName(V->getName());
    ValueMap[V] = &*Arg;
    Arg++;
  }

  return FN;
}

void GPUNodeBuilder::insertKernelIntrinsics(ppcg_kernel *Kernel) {
  Intrinsic::ID IntrinsicsBID[2];
  Intrinsic::ID IntrinsicsTID[3];

  switch (Arch) {
  case GPUArch::NVPTX64:
    IntrinsicsBID[0] = Intrinsic::nvvm_read_ptx_sreg_ctaid_x;
    IntrinsicsBID[1] = Intrinsic::nvvm_read_ptx_sreg_ctaid_y;

    IntrinsicsTID[0] = Intrinsic::nvvm_read_ptx_sreg_tid_x;
    IntrinsicsTID[1] = Intrinsic::nvvm_read_ptx_sreg_tid_y;
    IntrinsicsTID[2] = Intrinsic::nvvm_read_ptx_sreg_tid_z;
    break;
  }

  auto addId = [this](__isl_take isl_id *Id, Intrinsic::ID Intr) mutable {
    std::string Name = isl_id_get_name(Id);
    Module *M = Builder.GetInsertBlock()->getParent()->getParent();
    Function *IntrinsicFn = Intrinsic::getDeclaration(M, Intr);
    Value *Val = Builder.CreateCall(IntrinsicFn, {});
    Val = Builder.CreateIntCast(Val, Builder.getInt64Ty(), false, Name);
    IDToValue[Id] = Val;
    KernelIDs.insert(std::unique_ptr<isl_id, IslIdDeleter>(Id));
  };

  for (int i = 0; i < Kernel->n_grid; ++i) {
    isl_id *Id = isl_id_list_get_id(Kernel->block_ids, i);
    addId(Id, IntrinsicsBID[i]);
  }

  for (int i = 0; i < Kernel->n_block; ++i) {
    isl_id *Id = isl_id_list_get_id(Kernel->thread_ids, i);
    addId(Id, IntrinsicsTID[i]);
  }
}

void GPUNodeBuilder::prepareKernelArguments(ppcg_kernel *Kernel, Function *FN) {
  auto Arg = FN->arg_begin();
  for (long i = 0; i < Kernel->n_array; i++) {
    if (!ppcg_kernel_requires_array_argument(Kernel, i))
      continue;

    isl_id *Id = isl_space_get_tuple_id(Prog->array[i].space, isl_dim_set);
    const ScopArrayInfo *SAI = ScopArrayInfo::getFromId(isl_id_copy(Id));
    isl_id_free(Id);

    if (SAI->getNumberOfDimensions() > 0) {
      Arg++;
      continue;
    }

    Value *Val = &*Arg;

    if (!gpu_array_is_read_only_scalar(&Prog->array[i])) {
      Type *TypePtr = SAI->getElementType()->getPointerTo();
      Value *TypedArgPtr = Builder.CreatePointerCast(Val, TypePtr);
      Val = Builder.CreateLoad(TypedArgPtr);
    }

    Value *Alloca = BlockGen.getOrCreateAlloca(SAI);
    Builder.CreateStore(Val, Alloca);

    Arg++;
  }
}

void GPUNodeBuilder::finalizeKernelArguments(ppcg_kernel *Kernel) {
  auto *FN = Builder.GetInsertBlock()->getParent();
  auto Arg = FN->arg_begin();

  bool StoredScalar = false;
  for (long i = 0; i < Kernel->n_array; i++) {
    if (!ppcg_kernel_requires_array_argument(Kernel, i))
      continue;

    isl_id *Id = isl_space_get_tuple_id(Prog->array[i].space, isl_dim_set);
    const ScopArrayInfo *SAI = ScopArrayInfo::getFromId(isl_id_copy(Id));
    isl_id_free(Id);

    if (SAI->getNumberOfDimensions() > 0) {
      Arg++;
      continue;
    }

    if (gpu_array_is_read_only_scalar(&Prog->array[i])) {
      Arg++;
      continue;
    }

    Value *Alloca = BlockGen.getOrCreateAlloca(SAI);
    Value *ArgPtr = &*Arg;
    Type *TypePtr = SAI->getElementType()->getPointerTo();
    Value *TypedArgPtr = Builder.CreatePointerCast(ArgPtr, TypePtr);
    Value *Val = Builder.CreateLoad(Alloca);
    Builder.CreateStore(Val, TypedArgPtr);
    StoredScalar = true;

    Arg++;
  }

  if (StoredScalar)
    /// In case more than one thread contains scalar stores, the generated
    /// code might be incorrect, if we only store at the end of the kernel.
    /// To support this case we need to store these scalars back at each
    /// memory store or at least before each kernel barrier.
    if (Kernel->n_block != 0 || Kernel->n_grid != 0)
      BuildSuccessful = 0;
}

void GPUNodeBuilder::createKernelVariables(ppcg_kernel *Kernel, Function *FN) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();

  for (int i = 0; i < Kernel->n_var; ++i) {
    struct ppcg_kernel_var &Var = Kernel->var[i];
    isl_id *Id = isl_space_get_tuple_id(Var.array->space, isl_dim_set);
    Type *EleTy = ScopArrayInfo::getFromId(Id)->getElementType();

    Type *ArrayTy = EleTy;
    SmallVector<const SCEV *, 4> Sizes;

    Sizes.push_back(nullptr);
    for (unsigned int j = 1; j < Var.array->n_index; ++j) {
      isl_val *Val = isl_vec_get_element_val(Var.size, j);
      long Bound = isl_val_get_num_si(Val);
      isl_val_free(Val);
      Sizes.push_back(S.getSE()->getConstant(Builder.getInt64Ty(), Bound));
    }

    for (int j = Var.array->n_index - 1; j >= 0; --j) {
      isl_val *Val = isl_vec_get_element_val(Var.size, j);
      long Bound = isl_val_get_num_si(Val);
      isl_val_free(Val);
      ArrayTy = ArrayType::get(ArrayTy, Bound);
    }

    const ScopArrayInfo *SAI;
    Value *Allocation;
    if (Var.type == ppcg_access_shared) {
      auto GlobalVar = new GlobalVariable(
          *M, ArrayTy, false, GlobalValue::InternalLinkage, 0, Var.name,
          nullptr, GlobalValue::ThreadLocalMode::NotThreadLocal, 3);
      GlobalVar->setAlignment(EleTy->getPrimitiveSizeInBits() / 8);
      GlobalVar->setInitializer(Constant::getNullValue(ArrayTy));

      Allocation = GlobalVar;
    } else if (Var.type == ppcg_access_private) {
      Allocation = Builder.CreateAlloca(ArrayTy, 0, "private_array");
    } else {
      llvm_unreachable("unknown variable type");
    }
    SAI =
        S.getOrCreateScopArrayInfo(Allocation, EleTy, Sizes, MemoryKind::Array);
    Id = isl_id_alloc(S.getIslCtx(), Var.name, nullptr);
    IDToValue[Id] = Allocation;
    LocalArrays.push_back(Allocation);
    KernelIds.push_back(Id);
    IDToSAI[Id] = SAI;
  }
}

void GPUNodeBuilder::createKernelFunction(
    ppcg_kernel *Kernel, SetVector<Value *> &SubtreeValues,
    SetVector<Function *> &SubtreeFunctions) {
  std::string Identifier = getKernelFuncName(Kernel->id);
  GPUModule.reset(new Module(Identifier, Builder.getContext()));

  switch (Arch) {
  case GPUArch::NVPTX64:
    if (Runtime == GPURuntime::CUDA)
      GPUModule->setTargetTriple(Triple::normalize("nvptx64-nvidia-cuda"));
    else if (Runtime == GPURuntime::OpenCL)
      GPUModule->setTargetTriple(Triple::normalize("nvptx64-nvidia-nvcl"));
    GPUModule->setDataLayout(computeNVPTXDataLayout(true /* is64Bit */));
    break;
  }

  Function *FN = createKernelFunctionDecl(Kernel, SubtreeValues);

  BasicBlock *PrevBlock = Builder.GetInsertBlock();
  auto EntryBlock = BasicBlock::Create(Builder.getContext(), "entry", FN);

  DT.addNewBlock(EntryBlock, PrevBlock);

  Builder.SetInsertPoint(EntryBlock);
  Builder.CreateRetVoid();
  Builder.SetInsertPoint(EntryBlock, EntryBlock->begin());

  ScopDetection::markFunctionAsInvalid(FN);

  prepareKernelArguments(Kernel, FN);
  createKernelVariables(Kernel, FN);
  insertKernelIntrinsics(Kernel);
}

std::string GPUNodeBuilder::createKernelASM() {
  llvm::Triple GPUTriple;

  switch (Arch) {
  case GPUArch::NVPTX64:
    switch (Runtime) {
    case GPURuntime::CUDA:
      GPUTriple = llvm::Triple(Triple::normalize("nvptx64-nvidia-cuda"));
      break;
    case GPURuntime::OpenCL:
      GPUTriple = llvm::Triple(Triple::normalize("nvptx64-nvidia-nvcl"));
      break;
    }
    break;
  }

  std::string ErrMsg;
  auto GPUTarget = TargetRegistry::lookupTarget(GPUTriple.getTriple(), ErrMsg);

  if (!GPUTarget) {
    errs() << ErrMsg << "\n";
    return "";
  }

  TargetOptions Options;
  Options.UnsafeFPMath = FastMath;

  std::string subtarget;

  switch (Arch) {
  case GPUArch::NVPTX64:
    subtarget = CudaVersion;
    break;
  }

  std::unique_ptr<TargetMachine> TargetM(GPUTarget->createTargetMachine(
      GPUTriple.getTriple(), subtarget, "", Options, Optional<Reloc::Model>()));

  SmallString<0> ASMString;
  raw_svector_ostream ASMStream(ASMString);
  llvm::legacy::PassManager PM;

  PM.add(createTargetTransformInfoWrapperPass(TargetM->getTargetIRAnalysis()));

  if (TargetM->addPassesToEmitFile(
          PM, ASMStream, TargetMachine::CGFT_AssemblyFile, true /* verify */)) {
    errs() << "The target does not support generation of this file type!\n";
    return "";
  }

  PM.run(*GPUModule);

  return ASMStream.str();
}

std::string GPUNodeBuilder::finalizeKernelFunction() {

  if (verifyModule(*GPUModule)) {
    DEBUG(dbgs() << "verifyModule failed on module:\n";
          GPUModule->print(dbgs(), nullptr); dbgs() << "\n";);

    if (FailOnVerifyModuleFailure)
      llvm_unreachable("VerifyModule failed.");

    BuildSuccessful = false;
    return "";
  }

  if (DumpKernelIR)
    outs() << *GPUModule << "\n";

  // Optimize module.
  llvm::legacy::PassManager OptPasses;
  PassManagerBuilder PassBuilder;
  PassBuilder.OptLevel = 3;
  PassBuilder.SizeLevel = 0;
  PassBuilder.populateModulePassManager(OptPasses);
  OptPasses.run(*GPUModule);

  std::string Assembly = createKernelASM();

  if (DumpKernelASM)
    outs() << Assembly << "\n";

  GPUModule.release();
  KernelIDs.clear();

  return Assembly;
}