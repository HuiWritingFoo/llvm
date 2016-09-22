//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file inserts an addreespacecast intruction to convert arguments 
// from NVPTX's shared or constants address space to generic address space
// before they can be used in some HCC intrinsics starting 
// with "atomic_" and "opencl_".
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"

using namespace llvm;

// Due to nVidia GPU limitation, we can not directly pass any arguments from
// NVPTX's shared address space to some HCC instrinsics starting with 
// "atomic_" and "opencl_". This pass mainly converts the arguments to
// NVPTX's generic address space. And those HCC instrinsics have the reverse
// convertion in their implementation.

namespace {

enum AddressSpace {
  ADDRESS_SPACE_GENERIC = 0,
  ADDRESS_SPACE_GLOBAL = 1,
  ADDRESS_SPACE_SHARED = 3,
  ADDRESS_SPACE_CONST = 4,
  ADDRESS_SPACE_LOCAL = 5,

  // NVVM Internal
  ADDRESS_SPACE_PARAM = 101
};

using IRBuilderTy = llvm::IRBuilder<>;

class MarshallArgs : public FunctionPass {

  // Find the target pointer and insert addrspacecast
  class TargetFinder : public InstVisitor<TargetFinder> {
  private:
    Function* F;
    Module* M;
    SmallVector<Instruction*, 32> WorkList;

  public:
    TargetFinder(Function* Func) : F(Func) { M = F->getParent(); }

    void run() {
      for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
        visit(*I);

      while(!WorkList.empty()) {
        Instruction* I = WorkList.pop_back_val();
        CallSite CS(I);
        Function *F = CS.getCalledFunction();
        if (!F) continue;

        IRBuilderTy IRB(I);
        std::vector<Type *> newArgs;

        for (unsigned i = 0, e = CS.getNumArgOperands(); i != e; ++i) {
          Type* argType = CS.getArgument(i)->getType();
          if (PointerType* PTy = dyn_cast<PointerType>(argType)) {
            // Insert Addrspacecast before the callsite
            if (ADDRESS_SPACE_SHARED == PTy->getAddressSpace()
               // constant, local memory used in PTX atom is not legal
               //|| (ADDRESS_SPACE_CONST == PTy->getAddressSpace() && F->getName().find("atomic_") != StringRef::npos) 
               ) {
              PointerType* DestTy = PTy->getPointerElementType()->getPointerTo(0);
              Value* Cast = IRB.CreatePointerCast(CS.getArgOperand(i), DestTy);
              // Replace with new use
              if (CS.isCall()) {
                CallInst* CI = dyn_cast<CallInst>(I);
                CI->setArgOperand(i, Cast);
              } else {
                InvokeInst* II = dyn_cast<InvokeInst>(I);
                II->setArgOperand(i, Cast);
              }
              argType = Cast->getType();
            }
          }
          newArgs.push_back (argType);
        }

        // Ensure the called function is the same type as the call
        FunctionType* FTy = F->getFunctionType();
        // TODO: Assuming return type is the same
        Type* returnType = FTy->getReturnType();
        FunctionType* newFTy = FunctionType::get(returnType, newArgs, F->isVarArg());
        CS.mutateFunctionType(newFTy);

        // Not sure why LLVM would add suffix to newFunction's name. Just erase old F first.
        StringRef Name = F->getName();
        llvm::GlobalValue::LinkageTypes LinkTy = F->getLinkage();
        Module* M = F->getParent();
        AttributeSet Att = F->getAttributes();

        F->eraseFromParent();

        Function * newFunction = Function::Create (newFTy, LinkTy, Name, M);
        newFunction->setAttributes(Att);
        if (CS.isCall()) {
          CallInst* CI = dyn_cast<CallInst>(I);
          CI->setCalledFunction(newFunction);
        } else {
          InvokeInst* II = dyn_cast<InvokeInst>(I);
          II->setCalledFunction(newFunction);
        }

        // Remove the old function declaration and declare new function siganture
        // Fortunately, those return type is not a pointer type for now
        M->getOrInsertFunction(newFunction->getName(), newFTy);
      }
    }

    void visitCallSite(CallSite &CS) {
      if (Function *F = CS.getCalledFunction()) {
        if ( (F->getName().find("opencl_") != StringRef::npos || 
             F->getName().find("atomic_") != StringRef::npos) ) { // Marshall shared memory at this time
          for (unsigned i = 0, e = CS.getNumArgOperands(); i != e; ++i) {
            Value* V = CS.getArgument(i);
            Type* Ty = V->getType();
            if (PointerType* PTy = dyn_cast<PointerType>(Ty)) {
              if (ADDRESS_SPACE_SHARED == PTy->getAddressSpace()) {
                WorkList.push_back(CS.getInstruction());
                return;
              } else if (Instruction* I = CS.getInstruction()) {
                // For PTX atom, accesses to const and local memory are illegal.
                if (F->getName().find("atomic_") != StringRef::npos) {
                  if (ADDRESS_SPACE_SHARED != PTy->getAddressSpace() && ADDRESS_SPACE_GLOBAL != PTy->getAddressSpace()) {
                    WorkList.push_back(I);
                  }
                }

              }
            }
          }
        }
      } 
      return;
    }

    void visitCallInst(CallInst &I) {
      CallSite CS(&I);
      visitCallSite(CS);
    }

    void visitInvokeInst(InvokeInst &I) {
      CallSite CS(&I);
      visitCallSite(CS);
    }
  };

public:
  static char ID;
  MarshallArgs() : FunctionPass(ID) { }
  ~MarshallArgs() { }

  void getAnalysisUsage(AnalysisUsage& AU) const {
    AU.addRequired<CallGraphWrapperPass>();
  }

  bool runOnFunction(Function& F) {
    if (!F.getCallingConv() == llvm::CallingConv::PTX_Kernel)
      return false;

    TargetFinder finder(&F);
    finder.run();

    return true;
  }
};

} // namespace

char MarshallArgs::ID = 0;
static RegisterPass<MarshallArgs>
Y("marshall-args", "Marshall args by converting them from NVPTX's shared to generic address space");
