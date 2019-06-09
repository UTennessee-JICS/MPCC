AC_INIT([MPCC], 0.0.1) dnl package name, version

dnl Select an optional include path, from a configure option
dnl or from an environment variable.
AC_ARG_WITH([mkl-home],
            AC_HELP_STRING([--with-mkl-home=MKL_HOME],
                           [the location of MKL]),
            [MKL_HOME=$withval])
if test [ -n "$MKL_HOME" ] ; then
   MPCC_CPPFLAGS="-DMKL_ILP64 -std=c++11 -I. -I$MKL_HOME/include/"
   LIBS="-Wl,--start-group $MKL_HOME/lib/intel64/libmkl_intel_ilp64.a $MKL_HOME/lib/intel64/libmkl_intel_thread.a $MKL_HOME/lib/intel64/libmkl_core.a -Wl,--end-group -lm -ldl ${LIBS}"
else
   echo "[WARN] Could not determine MKL_HOME"
   MPCC_CPPFLAGS="-DNOMKL -std=c++11 -I."
fi

dnl Now find the compiler and compiler flags to use
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "[WARN] Could not determine R_HOME"
  exit 1
fi
CC=`"${R_HOME}/bin/R" CMD config CC`
CPP=`"${R_HOME}/bin/R" CMD config CPP`
CFLAGS=`"${R_HOME}/bin/R" CMD config CFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`
AC_PROG_CC
AC_PROG_CPP

CPPFLAGS="${CPPFLAGS} ${MPCC_CPPFLAGS}"

dnl substitute MPCC_CPPFLAGS and LIBS
AC_SUBST(MPCC_CPPFLAGS)
AC_SUBST(LIBS)
dnl and do substitution in the src/Makevars.in and src/config.h
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

