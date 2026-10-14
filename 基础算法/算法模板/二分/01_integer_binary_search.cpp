#include <functional>

/*
 * 整数二分
 *
 * 前提：check 在 [l, r] 上具有单调性。
 * bsearch_first：找到第一个满足 check 的位置。
 * bsearch_last：找到最后一个满足 check 的位置。
 * 复杂度：时间 O(log n)，空间 O(1)。
 */
int bsearch_first(int l, int r, const std::function<bool(int)> &check)
{
    while (l < r)
    {
        // check(mid) 成立时答案可能就是 mid，因此保留 [l, mid]。
        int mid = (l + r) >> 1;
        if (check(mid)) r = mid;
        else l = mid + 1;
    }
    return l;
}

int bsearch_last(int l, int r, const std::function<bool(int)> &check)
{
    while (l < r)
    {
        // 向上取整，避免只剩两个元素时 mid == l 导致死循环。
        int mid = (l + r + 1) >> 1;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}
