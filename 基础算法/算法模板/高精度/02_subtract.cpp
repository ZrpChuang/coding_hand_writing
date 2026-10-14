#include <vector>

/*
 * 高精度减法：C = A - B
 *
 * 前提：A >= B；数字按低位到高位存储。
 * borrow 表示上一位是否向当前位借了 1。
 * (value + 10) % 10 同时处理正常减法和借位后的负数。
 * 复杂度：时间 O(|A|)，空间 O(|A|)。
 */
std::vector<int> subtract(
    const std::vector<int> &A,
    const std::vector<int> &B)
{
    std::vector<int> C;
    int borrow = 0;

    for (std::size_t i = 0; i < A.size(); i ++)
    {
        int value = A[i] - borrow;
        if (i < B.size()) value -= B[i];
        C.push_back((value + 10) % 10);
        // value < 0 表示当前位需要向更高位借 1。
        borrow = value < 0;
    }

    // 删除高位多余的 0，但数字 0 本身至少保留一位。
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
