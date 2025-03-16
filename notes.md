## Linux

- tested on Linux (Fedora 41) - works well, images working in CLI

## macOS 

- macFUSE OK but appears to force multithreading and rapidly hits rate limit
- fuse-t OK but needed to link libfuse-t.dylib as libfuse.dylib
- since fuse-t is NFS it seems to lose a bunch of the file attributes?
- no filesystem icon, sad times (basically none of the FS metadata seems to work on mac)
