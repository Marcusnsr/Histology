#!/bin/bash
key=~/.ssh/id_ed25519
user=dlf903@di.ku.dk
erdadir=./
mnt=/home/dlf903/erda_mount

if [ -f "$key" ]; then
    mkdir -p ${mnt}
    
    # Check for stale files/mounts and clean automatically (NO PROMPT)
    if [ "$(ls -A ${mnt})" ]; then
        echo "Mount point '${mnt}' is not empty. Cleaning..."
        # Force unmount just in case it's a stale mount
        fusermount -uz ${mnt} 2>/dev/null
        # Clean folder
        rm -rf ${mnt}/*
    fi
    
    sshfs ${user}@io.erda.dk:${erdadir} ${mnt} \
        -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
        -o IdentityFile=${key} \
        -o StrictHostKeyChecking=no
    
    # Verify success
    if mountpoint -q ${mnt}; then
        echo "Success: ERDA mounted at ${mnt}"
        exit 0
    else
        echo "Error: Mount failed."
        exit 1
    fi
else
    echo "Error: Key not found."
    exit 1
fi