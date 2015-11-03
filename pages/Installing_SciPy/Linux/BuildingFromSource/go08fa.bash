export BASE=${BASE:-$1}

mkdir -p ${BASE}/logs/

function exec_and_log () {
    bash -v $1 2>&1 | tee ${BASE}/logs/$1.log
}

setup_files=`ls ??_*.bash | sort -n`

for file in $setup_files
do
  exec_and_log $file
done 

