; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown -mattr=+ssse3 | FileCheck %s --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-unknown -mattr=+avx | FileCheck %s --check-prefix=AVX

define <8 x i16> @phaddw1(<8 x i16> %x, <8 x i16> %y) {
; SSSE3-LABEL: phaddw1:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phaddw %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phaddw1:
; AVX:       # BB#0:
; AVX-NEXT:    vphaddw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <8 x i16> %x, <8 x i16> %y, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %b = shufflevector <8 x i16> %x, <8 x i16> %y, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %r = add <8 x i16> %a, %b
  ret <8 x i16> %r
}

define <8 x i16> @phaddw2(<8 x i16> %x, <8 x i16> %y) {
; SSSE3-LABEL: phaddw2:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phaddw %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phaddw2:
; AVX:       # BB#0:
; AVX-NEXT:    vphaddw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <8 x i16> %x, <8 x i16> %y, <8 x i32> <i32 1, i32 2, i32 5, i32 6, i32 9, i32 10, i32 13, i32 14>
  %b = shufflevector <8 x i16> %y, <8 x i16> %x, <8 x i32> <i32 8, i32 11, i32 12, i32 15, i32 0, i32 3, i32 4, i32 7>
  %r = add <8 x i16> %a, %b
  ret <8 x i16> %r
}

define <4 x i32> @phaddd1(<4 x i32> %x, <4 x i32> %y) {
; SSSE3-LABEL: phaddd1:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phaddd %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phaddd1:
; AVX:       # BB#0:
; AVX-NEXT:    vphaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <4 x i32> %x, <4 x i32> %y, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %b = shufflevector <4 x i32> %x, <4 x i32> %y, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %r = add <4 x i32> %a, %b
  ret <4 x i32> %r
}

define <4 x i32> @phaddd2(<4 x i32> %x, <4 x i32> %y) {
; SSSE3-LABEL: phaddd2:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phaddd %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phaddd2:
; AVX:       # BB#0:
; AVX-NEXT:    vphaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <4 x i32> %x, <4 x i32> %y, <4 x i32> <i32 1, i32 2, i32 5, i32 6>
  %b = shufflevector <4 x i32> %y, <4 x i32> %x, <4 x i32> <i32 4, i32 7, i32 0, i32 3>
  %r = add <4 x i32> %a, %b
  ret <4 x i32> %r
}

define <4 x i32> @phaddd3(<4 x i32> %x) {
; SSSE3-LABEL: phaddd3:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phaddd %xmm0, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phaddd3:
; AVX:       # BB#0:
; AVX-NEXT:    vphaddd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 undef, i32 2, i32 4, i32 6>
  %b = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 undef, i32 3, i32 5, i32 7>
  %r = add <4 x i32> %a, %b
  ret <4 x i32> %r
}

define <4 x i32> @phaddd4(<4 x i32> %x) {
; SSSE3-LABEL: phaddd4:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phaddd %xmm0, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phaddd4:
; AVX:       # BB#0:
; AVX-NEXT:    vphaddd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 0, i32 2, i32 undef, i32 undef>
  %b = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %r = add <4 x i32> %a, %b
  ret <4 x i32> %r
}

define <4 x i32> @phaddd5(<4 x i32> %x) {
; SSSE3-LABEL: phaddd5:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phaddd %xmm0, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phaddd5:
; AVX:       # BB#0:
; AVX-NEXT:    vphaddd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 0, i32 3, i32 undef, i32 undef>
  %b = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 1, i32 2, i32 undef, i32 undef>
  %r = add <4 x i32> %a, %b
  ret <4 x i32> %r
}

define <4 x i32> @phaddd6(<4 x i32> %x) {
; SSSE3-LABEL: phaddd6:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phaddd %xmm0, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phaddd6:
; AVX:       # BB#0:
; AVX-NEXT:    vphaddd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %b = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %r = add <4 x i32> %a, %b
  ret <4 x i32> %r
}

define <4 x i32> @phaddd7(<4 x i32> %x) {
; SSSE3-LABEL: phaddd7:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phaddd %xmm0, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phaddd7:
; AVX:       # BB#0:
; AVX-NEXT:    vphaddd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 undef, i32 3, i32 undef, i32 undef>
  %b = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 undef, i32 2, i32 undef, i32 undef>
  %r = add <4 x i32> %a, %b
  ret <4 x i32> %r
}

define <8 x i16> @phsubw1(<8 x i16> %x, <8 x i16> %y) {
; SSSE3-LABEL: phsubw1:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phsubw %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phsubw1:
; AVX:       # BB#0:
; AVX-NEXT:    vphsubw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <8 x i16> %x, <8 x i16> %y, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %b = shufflevector <8 x i16> %x, <8 x i16> %y, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %r = sub <8 x i16> %a, %b
  ret <8 x i16> %r
}

define <4 x i32> @phsubd1(<4 x i32> %x, <4 x i32> %y) {
; SSSE3-LABEL: phsubd1:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phsubd %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phsubd1:
; AVX:       # BB#0:
; AVX-NEXT:    vphsubd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <4 x i32> %x, <4 x i32> %y, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %b = shufflevector <4 x i32> %x, <4 x i32> %y, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %r = sub <4 x i32> %a, %b
  ret <4 x i32> %r
}

define <4 x i32> @phsubd2(<4 x i32> %x) {
; SSSE3-LABEL: phsubd2:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phsubd %xmm0, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phsubd2:
; AVX:       # BB#0:
; AVX-NEXT:    vphsubd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 undef, i32 2, i32 4, i32 6>
  %b = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 undef, i32 3, i32 5, i32 7>
  %r = sub <4 x i32> %a, %b
  ret <4 x i32> %r
}

define <4 x i32> @phsubd3(<4 x i32> %x) {
; SSSE3-LABEL: phsubd3:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phsubd %xmm0, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phsubd3:
; AVX:       # BB#0:
; AVX-NEXT:    vphsubd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 0, i32 2, i32 undef, i32 undef>
  %b = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %r = sub <4 x i32> %a, %b
  ret <4 x i32> %r
}

define <4 x i32> @phsubd4(<4 x i32> %x) {
; SSSE3-LABEL: phsubd4:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    phsubd %xmm0, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: phsubd4:
; AVX:       # BB#0:
; AVX-NEXT:    vphsubd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    retq
  %a = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %b = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %r = sub <4 x i32> %a, %b
  ret <4 x i32> %r
}
