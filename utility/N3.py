import sys

def main():
    data = sys.stdin.read().strip().split()
    it = iter(data)
    n = int(next(it))

    A = [[int(next(it)) for _ in range(n)] for _ in range(n)]
    B = [[int(next(it)) for _ in range(n)] for _ in range(n)]

    C = [[0] * n for _ in range(n)]

    for i in range(n):
        for k in range(n):
            aik = A[i][k]
            for j in range(n):
                C[i][j] += aik * B[k][j]

    out_lines = []
    for i in range(n):
        out_lines.append(" ".join(str(x) for x in C[i]))
    sys.stdout.write("\n".join(out_lines))

if __name__ == "__main__":
    main()
   
 