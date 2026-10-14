#include <algorithm>

/*
 * 快速排序（Hoare 分区）
 *
 * 作用：将闭区间 q[l..r] 按升序排列。
 * 思路：选择中间元素作为基准 x，双指针从两端向中间移动；
 *       左侧找到 >= x 的数，右侧找到 <= x 的数，然后交换。
 * 分区结束后：[l, j] 中的数 <= x，[j + 1, r] 中的数 >= x。
 * 复杂度：平均 O(n log n)，最坏 O(n^2)；递归栈平均 O(log n)。
 */
void quick_sort(int q[], int l, int r)
{
    // 区间为空或只有一个元素时已经有序。
    if (l >= r) return;

    // 指针从区间外开始，配合 do-while 保证每轮至少移动一次。
    int i = l - 1, j = r + 1;
    int x = q[(l + r) >> 1];
    while (i < j)
    {
        // 找到左右两侧放错位置的元素。
        do i ++; while (q[i] < x);
        do j --; while (q[j] > x);
        if (i < j) std::swap(q[i], q[j]);
    }

    // Hoare 分区以 j 为边界递归，两个区间不能重叠。
    quick_sort(q, l, j);
    quick_sort(q, j + 1, r);
}
