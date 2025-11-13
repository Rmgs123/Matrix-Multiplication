# Классический алгоритм умножения матриц (тройной цикл)

Классический алгоритм напрямую реализует определение матричного произведения.
Он работает для прямоугольных матриц размеров $n \times m$ и $m \times p$,
даёт $n \cdot m \cdot p$ умножений и $n \cdot p \cdot (m-1)$ сложений,
и имеет асимптотику $O(nmp)$ (в квадратном случае — $O(n^3)$).

Ниже — базовая реализация без оптимизаций уровня кэша и без внешних библиотек.

```python
def matmul_classic(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        rowA = A[i]
        rowC = C[i]
        for k in range(m):
            aik = rowA[k]
            rowB = B[k]
            for j in range(p):
                rowC[j] += aik * rowB[j]
    return C
