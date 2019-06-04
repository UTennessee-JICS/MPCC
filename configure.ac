AC_INIT([MPCC], 0.0.1) dnl package name, version

dnl Select an optional include path, from a configure option
dnl or from an environment variable.
AC_ARG_WITH([mkl-home],
            AC_HELP_STRING([--with-mkl-home=MKL_HOME],
                           [the location of MKL]),
            [mkl_home=$withval])
MPCC_CPPFLAGS="-DMKL_ILP64 -std=c++11 -I."
if test [ -n "$mkl_home" ] ; then
   MPCC_CPPFLAGS="-DMKL_ILP64 -std=c++11 -I. -I$mkl_home/include/"
   LIBS="-L$mkl_home -Wl,--start-group $mkl_home/lib/intel64/libmkl_intel_ilp64.a $mkl_home/lib/intel64/libmkl_intel_thread.a $mkl_home/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl ${LIBS}"
fi

dnl Now find the compiler and compiler flags to use
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
CC=`"${R_HOME}/bin/R" CMD config CC`
CPP=`"${R_HOME}/bin/R" CMD config CPP`
CFLAGS=`"${R_HOME}/bin/R" CMD config CFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`
AC_PROG_CC
AC_PROG_CPP

CPPFLAGS="${CPPFLAGS} ${MPCC_CPPFLAGS}"

dnl Check the headers can be found
AC_CHECK_HEADERS(mkl.h)
if test "${ac_cv_header_mkl_h}" = no; then
   AC_MSG_ERROR("mkl.h not found")
fi

dnl substitute MPCC_CPPFLAGS and LIBS
AC_SUBST(MPCC_CPPFLAGS)
AC_SUBST(LIBS)
dnl and do substitution in the src/Makevars.in and src/config.h
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

