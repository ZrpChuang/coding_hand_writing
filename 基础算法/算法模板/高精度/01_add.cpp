#include <vector>

/*
 * 高精度加法：C = A + B
 *
 * A、B、C 都按低位到高位存储，例如 123 表示为 {3, 2, 1}。
 * 从最低位逐位相加，carry 同时保存当前位总和与向高位的进位。
 * 复杂度：时间 O(max(|A|, |B|))，空间 O(max(|A|, |B|))。
 */
std::vector<int> add(const std::vector<int> &A, const std::vector<int> &B)
{
    std::vector<int> C;
    int carry = 0;

    for (std::size_t i = 0; i < A.size() || i < B.size() || carry; i ++)
    {
        if (i < A.size()) carry += A[i];
        if (i < B.size()) carry += B[i];
        // 个位写入答案，其余部分作为下一位的进位。
        C.push_back(carry % 10);
        carry /= 10;
    }

    return C;
}
