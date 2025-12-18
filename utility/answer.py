def read_matrix(n):
    return [list(map(int, input().split())) for _ in range(n)]

def matmul_classic(A, B):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        rowA = A[i]
        rowC = C[i]
        for k in range(n):
            aik = rowA[k]
            rowB = B[k]
            for j in range(n):
                rowC[j] += aik * rowB[j]
    return C

def add_matrix(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def sub_matrix(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def strassen_core(A, B, cutoff):
    n = len(A)
    if n <= cutoff:
        return matmul_classic(A, B)

    if n % 2 == 1:
        return matmul_classic(A, B)

    m = n // 2

    a11 = [row[:m] for row in A[:m]]
    a12 = [row[m:] for row in A[:m]]
    a21 = [row[:m] for row in A[m:]]
    a22 = [row[m:] for row in A[m:]]

    b11 = [row[:m] for row in B[:m]]
    b12 = [row[m:] for row in B[:m]]
    b21 = [row[:m] for row in B[m:]]
    b22 = [row[m:] for row in B[m:]]

    m1 = strassen_core(add_matrix(a11, a22), add_matrix(b11, b22), cutoff)
    m2 = strassen_core(add_matrix(a21, a22), b11, cutoff)
    m3 = strassen_core(a11, sub_matrix(b12, b22), cutoff)
    m4 = strassen_core(a22, sub_matrix(b21, b11), cutoff)
    m5 = strassen_core(add_matrix(a11, a12), b22, cutoff)
    m6 = strassen_core(sub_matrix(a21, a11), add_matrix(b11, b12), cutoff)
    m7 = strassen_core(sub_matrix(a12, a22), add_matrix(b21, b22), cutoff)
    
    c11 = sub_matrix(add_matrix(add_matrix(m1, m4), m7), m5)
    c12 = add_matrix(m3, m5)
    c21 = add_matrix(m2, m4)
    c22 = add_matrix(sub_matrix(add_matrix(m1, m3), m2), m6)

    C = [c11[i] + c12[i] for i in range(m)] + \
        [c21[i] + c22[i] for i in range(m)]
    return C

def matmul_strassen(A, B, cutoff=64):
    n = len(A)
    if n == 0:
        return []

    if n != len(B) or n != len(A[0]) or n != len(B[0]):
        raise ValueError("Матрицы должны быть квадратными и одинакового размера")

    return strassen_core(A, B, cutoff)

def main():
    n = int(input())
    A = read_matrix(n)
    B = read_matrix(n)

    C = matmul_strassen(A, B, cutoff=64)

    for row in C:
        print(*row)

if __name__ == "__main__":

    main()
