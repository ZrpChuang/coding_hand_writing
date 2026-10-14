/*
 * 常用位运算
 *
 * n >> k：把第 k 位移动到最低位。
 * & 1：只保留最低位，因此得到 n 的第 k 位。
 * n & -n：利用补码性质，只保留 n 最低位的 1。
 */
int get_bit(int n, int k)
{
    // 等价于 (n >> k) & 1，& 的优先级低于 >>。
    return n >> k & 1;
}

// 例如 12 = 1100(2)，lowbit(12) = 0100(2) = 4。
int lowbit(int n)
{
    return n & -n;
}

// 每次删除最低位的 1，循环次数正好等于 1 的数量。
int count_ones(int n)
{
    int result = 0;
    while (n)
    {
        n -= lowbit(n);
        result ++;
    }
    return result;
}
