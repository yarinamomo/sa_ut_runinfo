# Run Pythia

Download doop image:
```shell
docker pull gfour/doop
```

Run doop over all `.py` files in folder `files`:
```shell
docker run -ti -v $(pwd):/tmp -w /tmp gfour/doop /bin/bash -c "/tmp/run_pythia.sh"
```

If you want to analyse your own files, just copy your files on the folder `files`.
