AC_INIT([MPCC], 0.0.1) dnl package name, version

dnl Select an optional include path, from a configure option
dnl or from an environment variable.
AC_ARG_WITH([mpcc-include],
            AC_HELP_STRING([--with-mpcc-include=INCLUDE_PATH],
                           [the location of MKL header files]),
            [mpcc_include_path=$withval])
MPCC_CPPFLAGS="-std=c++11 -I."
if test [ -n "$mpcc_include_path" ] ; then
   MPCC_CPPFLAGS="-std=c++11 -I. -I${mpcc_include_path}"
else
  if test [ -n "${ODBC_INCLUDE}" ] ; then
     MPCC_CPPFLAGS="-std=c++11 -I. -I${MPCC_INCLUDE}"
  fi
fi

dnl ditto for a library path
AC_ARG_WITH([mpcc-lib],
            AC_HELP_STRING([--with-mpcc-lib=LIB_PATH],
                           [the location of MKL libraries]),
            [mpcc_lib_path=$withval])
if test [ -n "$mpcc_lib_path" ] ; then
   LIBS="-L$mpcc_lib_path -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl ${LIBS}"
else
  if test [ -n "${MPCC_LIBS}" ] ; then
     LIBS="-L${MPCC_LIBS} -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl ${LIBS}"
  fi
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

dnl substitute RODBC_CPPFLAGS and LIBS
AC_SUBST(MPCC_CPPFLAGS)
AC_SUBST(LIBS)
dnl and do substitution in the src/Makevars.in and src/config.h
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

