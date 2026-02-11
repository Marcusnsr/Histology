#!/bin/bash
mnt=/home/dlf903/erda_mount

if mountpoint -q ${mnt}; then
    fusermount -uz ${mnt}
    echo "Success: Unmounted ${mnt}"
else
    echo "Info: '${mnt}' was not mounted."
fi

if [ -d ${mnt} ]; then
    rmdir ${mnt} 2>/dev/null || echo "Kept mount dir (not empty or busy)"
fi