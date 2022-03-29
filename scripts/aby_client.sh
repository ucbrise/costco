#!/usr/bin/env bash
#set -x

SERVER_PRIVATE_IP=$1
CIRCUIT_FILE=$2
ROUNDS=$3

#source ~/.bash_profile
source ~/.bashrc
cd ~/ABY/build

./bin/cheapsmc -r 1 -a $SERVER_PRIVATE_IP -i ${ROUNDS} -c $CIRCUIT_FILE
#2&> /dev/null
#for i in $(seq 1 ${ROUNDS})
#do
#./bin/cheapsmc -r 1 -a $SERVER_PRIVATE_IP -i 1 -c $CIRCUIT_FILE 2&> /dev/null
#done
