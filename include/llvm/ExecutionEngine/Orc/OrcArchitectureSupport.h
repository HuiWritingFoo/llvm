//===-- OrcArchitectureSupport.h - Architecture support code  ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Architecture specific code for Orc, e.g. callback assembly.
//
// Architecture classes should be part of the JIT *target* process, not the host
// process (except where you're doing hosted JITing and the two are one and the
// same).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_ORCARCHITECTURESUPPORT_H
#define LLVM_EXECUTIONENGINE_ORC_ORCARCHITECTURESUPPORT_H

#include "IndirectionUtils.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Process.h"

namespace llvm {
namespace orc {

class OrcX86_64 {
public:
  static const unsigned PointerSize = 8;
  static const unsigned TrampolineSize = 8;
  static const unsigned ResolverCodeSize = 0x78;

  typedef TargetAddress (*JITReentryFn)(void *CallbackMgr,
                                        void *TrampolineId);

  /// @brief Write the resolver code into the given memory. The user is be
  ///        responsible for allocating the memory and setting permissions.
  static void writeResolverCode(uint8_t *ResolveMem, JITReentryFn Reentry,
                                void *CallbackMgr);

  /// @brief Write the requsted number of trampolines into the given memory,
  ///        which must be big enough to hold 1 pointer, plus NumTrampolines
  ///        trampolines.
  static void writeTrampolines(uint8_t *TrampolineMem, void *ResolverAddr,
                               unsigned NumTrampolines);

  /// @brief Provide information about stub blocks generated by the
  ///        makeIndirectStubsBlock function.
  class IndirectStubsInfo {
    friend class OrcX86_64;
  public:
    const static unsigned StubSize = 8;

    IndirectStubsInfo() : NumStubs(0) {}
    IndirectStubsInfo(IndirectStubsInfo &&Other)
        : NumStubs(Other.NumStubs), StubsMem(std::move(Other.StubsMem)) {
      Other.NumStubs = 0;
    }
    IndirectStubsInfo& operator=(IndirectStubsInfo &&Other) {
      NumStubs = Other.NumStubs;
      Other.NumStubs = 0;
      StubsMem = std::move(Other.StubsMem);
      return *this;
    }

    /// @brief Number of stubs in this block.
    unsigned getNumStubs() const { return NumStubs; }

    /// @brief Get a pointer to the stub at the given index, which must be in
    ///        the range 0 .. getNumStubs() - 1.
    void* getStub(unsigned Idx) const {
      return static_cast<uint64_t*>(StubsMem.base()) + Idx;
    }

    /// @brief Get a pointer to the implementation-pointer at the given index,
    ///        which must be in the range 0 .. getNumStubs() - 1.
    void** getPtr(unsigned Idx) const {
      char *PtrsBase =
        static_cast<char*>(StubsMem.base()) + NumStubs * StubSize;
      return reinterpret_cast<void**>(PtrsBase) + Idx;
    }
  private:
    unsigned NumStubs;
    sys::OwningMemoryBlock StubsMem;
  };

  /// @brief Emit at least MinStubs worth of indirect call stubs, rounded out to
  ///        the nearest page size.
  ///
  ///   E.g. Asking for 4 stubs on x86-64, where stubs are 8-bytes, with 4k
  /// pages will return a block of 512 stubs (4096 / 8 = 512). Asking for 513
  /// will return a block of 1024 (2-pages worth).
  static std::error_code emitIndirectStubsBlock(IndirectStubsInfo &StubsInfo,
                                                unsigned MinStubs,
                                                void *InitialPtrVal);
};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_ORCARCHITECTURESUPPORT_H