#include <vector>

/*
 * 一维前缀和
 *
 * 约定：a 使用 1-based 下标，a[0] 不存有效数据。
 * 定义：s[i] = a[1] + ... + a[i]。
 * 查询：[l, r] 的和 = s[r] - s[l - 1]。
 * 复杂度：预处理 O(n)，每次查询 O(1)，空间 O(n)。
 */
std::vector<long long> build_prefix_sum(const std::vector<int> &a)
{
    std::vector<long long> s(a.size());
    for (std::size_t i = 1; i < a.size(); i ++)
        s[i] = s[i - 1] + a[i];
    return s;
}

// 从 s[r] 中减掉区间 [1, l - 1]，只留下 [l, r]。
long long range_sum(const std::vector<long long> &s, int l, int r)
{
    return s[r] - s[l - 1];
}
