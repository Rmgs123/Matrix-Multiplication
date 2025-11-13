# Вариант Штрассена–Винограда

Вариант Штрассена–Винограда сохраняет семь блочных умножений на уровень
рекурсии, но перестраивает схему сложений, уменьшая их количество
по сравнению с оригинальным алгоритмом Штрассена. Асимптотика та же:
$O(n^{\log_2 7})$ для квадратных матриц.

Здесь используется схема с промежуточными $S_1,\dots,S_{10}$ и
семью произведениями $P_1,\dots,P_7$. Как и в реализации Штрассена:

- вход — квадратные матрицы одного размера;
- при необходимости они дополняются нулями до степени двойки;
- рекурсивная база — случай $n = 1$;
- результат обрезается до исходного размера.

```python
def mat_add(A, B):
    n = len(A)
    m = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]


def mat_sub(A, B):
    n = len(A)
    m = len(A[0])
    return [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]


def pad_square(A, size):
    n = len(A)
    m = len(A[0])
    R = [row + [0] * (size - m) for row in A]
    for _ in range(size - n):
        R.append([0] * size)
    return R


def split_quadrants(A):
    n = len(A)
    k = n // 2
    a11 = [row[:k] for row in A[:k]]
    a12 = [row[k:] for row in A[:k]]
    a21 = [row[:k] for row in A[k:]]
    a22 = [row[k:] for row in A[k:]]
    return a11, a12, a21, a22


def join_quadrants(c11, c12, c21, c22):
    k = len(c11)
    R = [c11[i] + c12[i] for i in range(k)]
    R.extend(c21[i] + c22[i] for i in range(k))
    return R


def next_power_of_two(n):
    m = 1
    while m < n:
        m <<= 1
    return m


def winograd_recursive(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    a11, a12, a21, a22 = split_quadrants(A)
    b11, b12, b21, b22 = split_quadrants(B)
    S1 = mat_sub(b12, b22)
    S2 = mat_add(a11, a12)
    S3 = mat_add(a21, a22)
    S4 = mat_sub(b21, b11)
    S5 = mat_add(a11, a22)
    S6 = mat_add(b11, b22)
    S7 = mat_sub(a12, a22)
    S8 = mat_add(b21, b22)
    S9 = mat_sub(a11, a21)
    S10 = mat_add(b11, b12)
    P1 = winograd_recursive(a11, S1)
    P2 = winograd_recursive(S2, b22)
    P3 = winograd_recursive(S3, b11)
    P4 = winograd_recursive(a22, S4)
    P5 = winograd_recursive(S5, S6)
    P6 = winograd_recursive(S7, S8)
    P7 = winograd_recursive(S9, S10)
    C11 = mat_add(mat_sub(mat_add(P5, P4), P2), P6)
    C12 = mat_add(P1, P2)
    C21 = mat_add(P3, P4)
    C22 = mat_sub(mat_add(P5, P1), mat_add(P3, P7))
    return join_quadrants(C11, C12, C21, C22)


def matmul_strassen_winograd(A, B):
    n = len(A)
    m = next_power_of_two(n)
    if m == n:
        return winograd_recursive(A, B)
    Ap = pad_square(A, m)
    Bp = pad_square(B, m)
    Cp = winograd_recursive(Ap, Bp)
    return [row[:n] for row in Cp[:n]]
