#!/usr/bin/env bash
if [ $# -lt 6 ]
then
    echo "Usage: aby.sh CMD SERVER_IP SERVER_PRIVATE_IP CLIENT_IP ROUNDS CIRCUIT_FILES"
    echo "[CMD]: runtime memory restrict_memory"
    exit 1
fi

SSH_KEY="~/.ssh/aws-us-west-1.pem"
CIRCUITS="../circuits"
LOGS="../experiments/logs"
USER="ubuntu"
CMD=$1
SERVER_IP=$2
SERVER_PRIVATE_IP=$3
CLIENT_IP=$4
ROUNDS=$5
shift 5

MEM_LIMIT=""
if [[ "$CMD" == "restrict_memory" ]]
then
    MEM_LIMIT=$1
    shift 1
fi

set -e
set -x

scp -i "${SSH_KEY}" aby_server.sh ${USER}@${SERVER_IP}:~/ABY/build
scp -i "${SSH_KEY}" aby_client.sh ${USER}@${CLIENT_IP}:~/ABY/build

rsync -avzP -e "ssh -i $SSH_KEY" "$CIRCUITS" ${USER}@${SERVER_IP}:~/ABY/build
rsync -avzP -e "ssh -i $SSH_KEY" "$CIRCUITS" ${USER}@${CLIENT_IP}:~/ABY/build

mkdir -p ${LOGS}

for CIRCUIT_FILE in $@
do
    CIRCUIT_FILE=${CIRCUIT_FILE#../}
    ssh -i "${SSH_KEY}" "${USER}@${SERVER_IP}" "~/ABY/build/aby_server.sh $CMD $SERVER_PRIVATE_IP $CIRCUIT_FILE $ROUNDS $MEM_LIMIT" | awk '{ print "[server]", $0 }' &
    ssh -i "${SSH_KEY}" "${USER}@${CLIENT_IP}" "~/ABY/build/aby_client.sh $SERVER_PRIVATE_IP $CIRCUIT_FILE $ROUNDS" | awk '{ print "[client]", $0 }' &

    for job in `jobs -p`
    do
        wait ${job}
    done
    mkdir -p $LOGS/$CMD/`dirname ${CIRCUIT_FILE}`
    #scp -i ~/.ssh/vivian_aws.pem ${USER}@${SERVER_IP}:~/ABY/build/log logs/memory/${CIRCUIT_FILE}
    scp -i "${SSH_KEY}" ${USER}@${SERVER_IP}:~/ABY/build/log $LOGS/$CMD/${CIRCUIT_FILE}
done
