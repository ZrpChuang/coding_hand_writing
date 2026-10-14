#include <vector>

/*
 * 归并排序
 *
 * 作用：将闭区间 q[l..r] 按升序排列。
 * 思路：先递归排好左右两半，再用双指针合并两个有序区间。
 * 特点：当相等时优先取左边元素，因此排序是稳定的。
 * 复杂度：时间 O(n log n)，辅助空间 O(n)。
 */
void merge_sort(int q[], int l, int r)
{
    if (l >= r) return;

    // 分治：把当前区间拆成 [l, mid] 和 [mid + 1, r]。
    int mid = (l + r) >> 1;
    merge_sort(q, l, mid);
    merge_sort(q, mid + 1, r);

    // i、j 分别扫描两个有序区间，k 指向临时数组的写入位置。
    std::vector<int> tmp(r - l + 1);
    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) tmp[k ++] = q[i ++];
        else tmp[k ++] = q[j ++];

    // 某一侧先耗尽后，把另一侧剩余元素全部接到末尾。
    while (i <= mid) tmp[k ++] = q[i ++];
    while (j <= r) tmp[k ++] = q[j ++];

    // 将合并结果写回原数组对应区间。
    for (i = l, j = 0; i <= r; i ++, j ++)
        q[i] = tmp[j];
}
