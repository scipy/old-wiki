cd ${BASE}
cat > pythonrc.py <<EOF
# Python startup script: sets up things like the history
# and tab completion features for the interpreter.
import os
import rlcompleter, readline
readline.parse_and_bind('tab: complete')
readline.parse_and_bind('"\M-OA": history-search-backward')
readline.parse_and_bind('"\M-[A": history-search-backward')
readline.parse_and_bind('"\M-\C-OA": history-search-backward')
readline.parse_and_bind('"\M-\C-[A": history-search-backward')
readline.parse_and_bind('"\M-OB": history-search-forward')
readline.parse_and_bind('"\M-[B": history-search-forward')
readline.parse_and_bind('"\M-\C-OB": history-search-forward')
readline.parse_and_bind('"\M-\C-[B": history-search-forward')

# Save and load history from a local file .pyhist
if False:
    import atexit
    histfile = ".pyhist"
    try:
        readline.read_history_file(histfile)
    except IOError:
        pass
    atexit.register(readline.write_history_file, histfile)
    del histfile

# Load user's startup file ~/.pythonrc.py if it exists
user_startup_filename = os.path.expanduser("~/.pythonrc.py")

if os.access(user_startup_filename,os.R_OK):
   del os
   execfile(user_startup_filename)
else:
   del os

del user_startup_filename
EOF

cat > setenv_gcc <<EOF
# Setup path and environment for using intel compilers but us combo python build.
BASE=${BASE}

export CC=gcc

shopt -s extglob

PATH=":\${PATH%:}:"             # Add leading and trailing colons
function add () { PATH=":\$1\${PATH//:\$1:/:}"; }
add \${BASE}/apps/tcl8.4.14_gcc/bin/
add \${BASE}/apps/Python-2.5_gcc/bin/
add \${BASE}/apps/fftw-3.1.2_gcc/bin/
add \${BASE}/src/ATLAS-3.6.0_gcc/bin/Linux_P4SSE2/
add \${BASE}/apps/freetype-2.3.2_gcc/bin/
add \${BASE}/apps/libpng-1.2.16_gcc/bin/
PATH="\${PATH//+(:)/:}"         # Remove duplicate colons
PATH="\${PATH%:}"               # Remove trailing colon
PATH="\${PATH#:}"               # Remove leading colon
export PATH

CPATH=":\${CPATH%:}:"           # Add leading and trailing colons
function add () { CPATH=":\$1\${CPATH//:\$1:/:}"; }
add \${BASE}/apps/tcl8.4.14_gcc/include/
add \${BASE}/apps/Python-2.5_gcc/include/
add \${BASE}/apps/fftw-3.1.2_gcc/include/
add \${BASE}/src/ATLAS-3.6.0_gcc/include/Linux_P4SSE2/
add \${BASE}/apps/freetype-2.3.2_gcc/include/
add \${BASE}/apps/libpng-1.2.16_gcc/include/
CPATH="\${CPATH//+(:)/:}"       # Remove duplicate colons
CPATH="\${CPATH%:}"             # Remove trailing colon
CPATH="\${CPATH#:}"             # Remove leading colon
export CPATH

LD_LIBRARY_PATH=":\${LD_LIBRARY_PATH%:}:" # Add leading and trailing colons
function add () { LD_LIBRARY_PATH=":\$1\${LD_LIBRARY_PATH//:\$1:/:}"; }
add \${BASE}/apps/tcl8.4.14_gcc/lib/
add \${BASE}/apps/Python-2.5_gcc/lib/
add \${BASE}/apps/fftw-3.1.2_gcc/lib/
add \${BASE}/src/ATLAS-3.6.0_gcc/lib/Linux_P4SSE2/
add \${BASE}/apps/freetype-2.3.2_gcc/lib/
add \${BASE}/apps/libpng-1.2.16_gcc/lib/
LD_LIBRARY_PATH="\${LD_LIBRARY_PATH//+(:)/:}" # Remove duplicate colons
LD_LIBRARY_PATH="\${LD_LIBRARY_PATH%:}"       # Remove trailing colon
LD_LIBRARY_PATH="\${LD_LIBRARY_PATH#:}"       # Remove leading colon
export LD_LIBRARY_PATH

MANPATH=":\${MANPATH%:}:"       # Add leading and trailing colons
function add () { MANPATH=":\$1\${MANPATH//:\$1:/:}"; }
add \${BASE}/apps/tcl8.4.14_gcc/man/
add \${BASE}/apps/Python-2.5_gcc/man/
add \${BASE}/apps/fftw-3.1.2_gcc/share/man/
add \${BASE}/apps/libpng-1.2.16_gcc/share/man/
MANPATH="\${MANPATH//+(:)/:}"   # Remove duplicate colons
MANPATH="\${MANPATH%:}"         # Remove trailing colon
MANPATH="\${MANPATH#:}"         # Remove leading colon
export MANPATH

export PYTHONSTARTUP=\${BASE}/pythonrc.py

EOF

# Source file to set environment up
. ${BASE}/setenv_gcc

# Download files and check MD5 sums
cd ${BASE}/zips
wget -nv http://prdownloads.sourceforge.net/tcl/tcl8.4.14-src.tar.gz
wget -nv http://prdownloads.sourceforge.net/tcl/tk8.4.14-src.tar.gz
wget -nv http://www.python.org/ftp/python/2.5/Python-2.5.tar.bz2
wget -nv http://www.netlib.org/lapack/lapack-3.1.1.tgz
wget -nv http://downloads.sourceforge.net/math-atlas/atlas3.6.0.tar.bz2
wget -nv http://www.fftw.org/fftw-3.1.2.tar.gz
wget -nv http://download.savannah.gnu.org/releases/freetype/freetype-2.3.2.tar.bz2
wget -nv http://downloads.sourceforge.net/libpng/libpng-1.2.16.tar.bz2
wget -nv http://www.zlib.net/zlib-1.2.3.tar.gz

# These are from svn sources: if SVN does not work,
# you will have to get these manually.  On my machine, I had a problem
# here and had to download these to my home directory first
cd ~
svn co http://svn.scipy.org/svn/numpy/trunk numpy
svn co http://svn.scipy.org/svn/scipy/trunk scipy
svn co http://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/matplotlib\
 matplotlib
mv ~/numpy ${BASE}/zips/
mv ~/scipy ${BASE}/zips/
mv ~/matplotlib ${BASE}/zips/

# Package the svn sources
cd ${BASE}/zips
tar -jcf numpy_svn.tbz numpy
tar -jcf scipy_svn.tbz scipy
tar -jcf matplotlib_svn.tbz matplotlib
rm -rf numpy scipy matplotlib

cat > md5sums.md5 <<EOF
51c6bf74d3ffdb0bd866ecdac6ff6460  tcl8.4.14-src.tar.gz
d12f591f5689f95c82bfb9c1015407bb  tk8.4.14-src.tar.gz
ddb7401e711354ca83b7842b733825a3  Python-2.5.tar.bz2
00b21551a899bcfbaa7b8443e1faeef9  lapack-3.1.1.tgz
df2ee2eb65d1c08ee93d04370172c262  atlas3.6.0.tar.bz2
08f2e21c9fd02f4be2bd53a62592afa4  fftw-3.1.2.tar.gz
119e1fe126fcfa5a70bc56db55f573d5  freetype-2.3.2.tar.bz2
7a1ca4f49bcffdec60d50f48460642bd  libpng-1.2.16.tar.bz2
debc62758716a169df9f62e6ab2bc634  zlib-1.2.3.tar.gz
EOF

md5sum -cw md5sums.md5
