#!/usr/bin/env bash
set -x
CGROUPDIR="/sys/fs/cgroup/bench"
CMD=$1
SERVER_PRIVATE_IP=$2
CIRCUIT_FILE=$3
ROUNDS=$4
shift 4

CIRCUIT=$(basename "$CIRCUIT_FILE")

#source ~/.bash_profile
source ~/.bashrc
export PATH=$PATH:/home/ubuntu/glibc/bin

cd ~/ABY/build

runtime() {
    ./bin/cheapsmc -r 0 -a "$SERVER_PRIVATE_IP" -i "$ROUNDS" -c "$CIRCUIT_FILE" | tee -a log
}

memory() {
    for i in $(seq 1 ${ROUNDS})
    do
        memusage ./bin/cheapsmc -r 0 -a "$SERVER_PRIVATE_IP" -i 1 -c "$CIRCUIT_FILE" 2>&1 | grep -oP "heap peak: \K\d+" | tee -a log
    done
}

restrict_memory() {
    MEM_LIMIT=$1
    echo $MEM_LIMIT
    echo $$ > "$CGROUPDIR/cgroup.procs"
    echo "$MEM_LIMIT" > "$CGROUPDIR/memory.high"
    runtime
    echo max > "$CGROUPDIR/memory.high"
}

rm -f log
$CMD $@

#for i in $(seq 1 ${ROUNDS})
#do
#valgrind --tool=massif --time-unit=B --massif-out-file=massif-$CIRCUIT ./bin/cheapsmc -r 0 -a $SERVER_PRIVATE_IP -i 1 -c $CIRCUIT_FILE
#ms_print massif-$CIRCUIT | grep -E "^\s+[0-9]+\s+[0-9,]+\s+" | tr -s ' ' | cut -d ' ' -f "5" | sort -nr | head -n1 >> log
#done
#for i in $(seq 1 ${ROUNDS})
#do
    #valgrind --tool=massif --massif-out-file=massif-$CIRCUIT ./bin/cheapsmc -r 0 -a $SERVER_PRIVATE_IP -i 1 -c $CIRCUIT_FILE
    #ms_print massif-$CIRCUIT | grep -E "^\s+[0-9]+\s+[0-9,]+\s+" | tr -s ' ' | cut -d ' ' -f "5" | sort -nr | head -n1 >> log
#done
