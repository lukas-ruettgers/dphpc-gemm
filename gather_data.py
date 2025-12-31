import subprocess

bins = ['./build/non_blocked_gemm_vec', './build/blocked_gemm_vec']

# for size in [128, 256, 512, 1024, 2048, 4096, 8192]:
for size in [8, 16, 32]:
    f = open(f'graphing/graphs/data/tb_size/m{size}_n{size}_k{size}.txt', 'w')

    for bin in bins:
        cmd = ['srun', '-A', 'dphpc', bin, '4096', '4096', '4096', str(size), str(size), str(size)]
        print(f'Running: {" ".join(cmd)}')
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        stdout = result.stdout
        stderr = result.stderr

        data = stdout.split('-----BEGIN\n')[1].split('-----END\n')[0].strip()
        f.write(data)
        f.write('\n==========\n')

    f.close()