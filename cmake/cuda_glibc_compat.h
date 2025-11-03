/* cmake/cuda_glibc_compat.h
   Ensure glibc math headers don’t expose IEC _Float* types to NVCC. */
#ifndef __NO_LONG_DOUBLE_MATH
#  define __NO_LONG_DOUBLE_MATH 1
#endif
#ifndef __HAVE_DISTINCT_FLOAT128
#  define __HAVE_DISTINCT_FLOAT128 0
#endif
#ifndef __HAVE_DISTINCT_FLOAT64X
#  define __HAVE_DISTINCT_FLOAT64X 0
#endif
/* Optional: avoid GNU inline math pulling in Float128 variants */
#ifdef _GNU_SOURCE
#  undef _GNU_SOURCE
#endif
