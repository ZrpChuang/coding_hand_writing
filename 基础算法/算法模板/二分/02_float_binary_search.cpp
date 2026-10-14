#include <functional>

/*
 * 浮点数二分
 *
 * 前提：check 在区间 [l, r] 上具有单调性，且答案被该区间覆盖。
 * 与整数二分不同，浮点数没有 mid +/- 1，而是直接收缩到 mid。
 * 当区间长度不超过 eps 时停止，返回 l 或 r 均可视为近似答案。
 * 复杂度：迭代次数约为 log2((r-l)/eps)。
 */
double bsearch_float(
    double l,
    double r,
    const std::function<bool(double)> &check,
    double eps = 1e-6)
{
    while (r - l > eps)
    {
        double mid = (l + r) / 2;
        // 本模板假设答案位于 check 为真的左边界方向。
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}
