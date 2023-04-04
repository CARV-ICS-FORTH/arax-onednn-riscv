#!/usr/bin/env bash

HOST_PORT=8888

if [ "$1" == "client" ]; then
        HOST_PORT=9999
fi

git clone https://github.com/CARV-ICS-FORTH/arax.git
git clone https://github.com/CARV-ICS-FORTH/ivshmem-uio.git

scp -P ${HOST_PORT} conf.json ubuntu@127.0.0.1:~/
scp -q -P ${HOST_PORT} -r ./arax ubuntu@127.0.0.1:~/
scp -q -P ${HOST_PORT} -r ./OneDnn ubuntu@127.0.0.1:~/
scp -q -P ${HOST_PORT} -r ./ivshmem-uio ubuntu@127.0.0.1:~/
scp -q -P ${HOST_PORT} -r ./prep.sh ubuntu@127.0.0.1:~/
scp -q -P ${HOST_PORT} .arax ubuntu@127.0.0.1:~/
