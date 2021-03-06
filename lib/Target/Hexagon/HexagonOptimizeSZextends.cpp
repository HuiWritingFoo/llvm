//===- HexagonOptimizeSZextends.cpp - Remove unnecessary argument extends -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pass that removes sign extends for function parameters. These parameters
// are already sign extended by the caller per Hexagon's ABI
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/StackProtector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"

#include "Hexagon.h"

using namespace llvm;

namespace llvm {
  FunctionPass *createHexagonOptimizeSZextends();
  void initializeHexagonOptimizeSZextendsPass(PassRegistry&);
}

namespace {
  struct HexagonOptimizeSZextends : public FunctionPass {
  public:
    static char ID;
    HexagonOptimizeSZextends() : FunctionPass(ID) {
      initializeHexagonOptimizeSZextendsPass(*PassRegistry::getPassRegistry());
    }
    bool runOnFunction(Function &F) override;

    const char *getPassName() const override {
      return "Remove sign extends";
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addPreserved<StackProtector>();
      FunctionPass::getAnalysisUsage(AU);
    }

    bool intrinsicAlreadySextended(Intrinsic::ID IntID);
  };
}

char HexagonOptimizeSZextends::ID = 0;

INITIALIZE_PASS(HexagonOptimizeSZextends, "reargs",
                "Remove Sign and Zero Extends for Args", false, false)

bool HexagonOptimizeSZextends::intrinsicAlreadySextended(Intrinsic::ID IntID) {
  switch(IntID) {
    case llvm::Intrinsic::hexagon_A2_addh_l16_sat_ll:
      return true;
    default:
      break;
  }
  return false;
}

bool HexagonOptimizeSZextends::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  unsigned Idx = 1;
  // Try to optimize sign extends in formal parameters. It's relying on
  // callee already sign extending the values. I'm not sure if our ABI
  // requires callee to sign extend though.
  for (auto &Arg : F.args()) {
    if (F.getAttributes().hasAttribute(Idx, Attribute::SExt)) {
      if (!isa<PointerType>(Arg.getType())) {
        for (auto UI = Arg.use_begin(); UI != Arg.use_end();) {
          if (isa<SExtInst>(*UI)) {
            Instruction* Use = cast<Instruction>(*UI);
            SExtInst* SI = new SExtInst(&Arg, Use->getType());
            assert (EVT::getEVT(SI->getType()) ==
                    (EVT::getEVT(Use->getType())));
            ++UI;
            Use->replaceAllUsesWith(SI);
            Instruction* First = &F.getEntryBlock().front();
            SI->insertBefore(First);
            Use->eraseFromParent();
          } else {
            ++UI;
          }
        }
      }
    }
    ++Idx;
  }

  // Try to remove redundant sext operations on Hexagon. The hardware
  // already sign extends many 16 bit intrinsic operations to 32 bits.
  // For example:
  // %34 = tail call i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32 %x, i32 %y)
  // %sext233 = shl i32 %34, 16
  // %conv52 = ashr exact i32 %sext233, 16
  for (auto &B : F) {
    for (auto &I : B) {
      // Look for arithmetic shift right by 16.
      BinaryOperator *Ashr = dyn_cast<BinaryOperator>(&I);
      if (!(Ashr && Ashr->getOpcode() == Instruction::AShr))
        continue;
      Value *AshrOp1 = Ashr->getOperand(1);
      ConstantInt *C = dyn_cast<ConstantInt>(AshrOp1);
      // Right shifted by 16.
      if (!(C && C->getSExtValue() == 16))
        continue;

      // The first operand of Ashr comes from logical shift left.
      Instruction *Shl = dyn_cast<Instruction>(Ashr->getOperand(0));
      if (!(Shl && Shl->getOpcode() == Instruction::Shl))
        continue;
      Value *Intr = Shl->getOperand(0);
      Value *ShlOp1 = Shl->getOperand(1);
      C = dyn_cast<ConstantInt>(ShlOp1);
      // Left shifted by 16.
      if (!(C && C->getSExtValue() == 16))
        continue;

      // The first operand of Shl comes from an intrinsic.
      if (IntrinsicInst *I = dyn_cast<IntrinsicInst>(Intr)) {
        if (!intrinsicAlreadySextended(I->getIntrinsicID()))
          continue;
        // All is well. Replace all uses of AShr with I.
        for (auto UI = Ashr->user_begin(), UE = Ashr->user_end();
             UI != UE; ++UI) {
          const Use &TheUse = UI.getUse();
          if (Instruction *J = dyn_cast<Instruction>(TheUse.getUser())) {
            J->replaceUsesOfWith(Ashr, I);
          }
        }
      }
    }
  }

  return true;
}


FunctionPass *llvm::createHexagonOptimizeSZextends() {
  return new HexagonOptimizeSZextends();
}
