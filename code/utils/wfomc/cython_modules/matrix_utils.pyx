# matrix_utils.pyx

cimport numpy as cnp

def swap_rows_cols(cnp.ndarray[cnp.int8_t, ndim=2] matrix, int i, int j):
    """
    交换矩阵的第 i 和第 j 行，同时交换第 i 和第 j 列。
    """
    cdef int n = matrix.shape[0]
    cdef int k, temp

    # 交换第 i 和第 j 行
    for k in range(n):
        temp = matrix[i, k]
        matrix[i, k] = matrix[j, k]
        matrix[j, k] = temp

    # 交换第 i 和第 j 列
    for k in range(n):
        temp = matrix[k, i]
        matrix[k, i] = matrix[k, j]
        matrix[k, j] = temp


def compare_tuples_cython(cnp.ndarray[cnp.int8_t, ndim=1] tuple1, 
                          cnp.ndarray[cnp.int8_t, ndim=1] tuple2):
    """
    比较两个整数数组，判断每一对元素是否满足 (a >= b and b > 0) or (a == b and b == 0)。
    """
    cdef Py_ssize_t i, n = tuple1.shape[0]
    
    for i in range(n):
        if not ((tuple1[i] >= tuple2[i] and tuple2[i] > 0) or 
                (tuple1[i] == tuple2[i] and tuple2[i] == 0)):
            return False
    return True
