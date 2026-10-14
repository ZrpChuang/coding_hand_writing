#include <vector>

/*
 * 高精度乘低精度：C = A * b
 *
 * A、C 按低位到高位存储，b 是普通非负整数。
 * 每轮将 A[i] * b 累加到 carry，写入个位后把其余部分进位。
 * 使用 long long，避免 A[i] * b 和进位相加时溢出 int。
 * 复杂度：时间 O(|A|)，空间 O(|A|)。
 */
std::vector<int> multiply(const std::vector<int> &A, int b)
{
    std::vector<int> C;
    long long carry = 0;

    for (std::size_t i = 0; i < A.size() || carry; i ++)
    {
        if (i < A.size()) carry += static_cast<long long>(A[i]) * b;
        // 当前结果位是 carry 的个位。
        C.push_back(carry % 10);
        carry /= 10;
    }

    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
