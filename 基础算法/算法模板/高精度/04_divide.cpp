#include <algorithm>
#include <vector>

/*
 * 高精度除以低精度：A / b = C ... remainder
 *
 * A、C 按低位到高位存储，要求 b > 0。
 * 除法必须从最高位开始模拟竖式，因此先逆序产生商，再翻转回来。
 * remainder 既保存每一步的余数，也通过引用参数返回最终余数。
 * 复杂度：时间 O(|A|)，空间 O(|A|)。
 */
std::vector<int> divide(
    const std::vector<int> &A,
    int b,
    int &remainder)
{
    std::vector<int> C;
    remainder = 0;

    for (int i = static_cast<int>(A.size()) - 1; i >= 0; i --)
    {
        // 把下一位落下来，再计算当前商位和新余数。
        remainder = remainder * 10 + A[i];
        C.push_back(remainder / b);
        remainder %= b;
    }

    // 当前商按高位到低位产生，翻转后恢复统一的低位在前表示。
    std::reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
