add_llvm_loadable_module( LLVMMarshallArgs
  MarshallArgs.cpp
)

add_dependencies(LLVMMarshallArgs intrinsics_gen)
