#include <iostream>
#include <iomanip>
#include <vector>
#include "Sort.cpp"

void testWithDifferentSizes() {
    std::cout << "\n不同规模数据的排序时间比较（单位：秒）：" << std::endl;
    std::cout << std::setw(10) << "数组大小" 
              << std::setw(15) << "插入排序"
              << std::setw(15) << "归并排序"
              << std::setw(15) << "快速排序"
              << std::setw(15) << "堆排序"
              << std::setw(15) << "基数排序"
              << std::setw(15) << "桶排序" << std::endl;

    std::vector<int> sizes = {50000, 100000, 200000, 300000, 500000};
    
    for (int size : sizes) {
        std::vector<int> arr = Sort::generateRandomArray(size);
        std::cout << std::setw(10) << size;
        

        // 插入排序
        std::vector<int> arr1 = arr;
        std::cout << std::setw(15) << std::fixed << std::setprecision(6)
                  << Sort::measureSortingTime(Sort::insertionSort, arr1);
        if (Sort::isOrdered(arr1)){
            std::cout << "正确";
        }
        
        // 归并排序
        std::vector<int> arr2 = arr;
        std::cout << std::setw(15) << Sort::measureSortingTime(
            [](std::vector<int>& a) { Sort::mergeSort(a, 0, a.size()-1); }, arr2);
        if (Sort::isOrdered(arr2)){
            std::cout << "正确";
        }
        // 快速排序
        std::vector<int> arr3 = arr;
        std::cout << std::setw(15) << Sort::measureSortingTime(
            [](std::vector<int>& a) { Sort::quickSort(a, 0, a.size()-1); }, arr3);
        if (Sort::isOrdered(arr3)){
            std::cout << "正确";
        }
        // 堆排序
        std::vector<int> arr4 = arr;
        std::cout << std::setw(15) << Sort::measureSortingTime(Sort::heapSort, arr4);
        if (Sort::isOrdered(arr4)){
            std::cout << "正确";
        }
        // 基数排序
        std::vector<int> arr5 = arr;
        std::cout << std::setw(15) << Sort::measureSortingTime(Sort::radixSort, arr5);
        if (Sort::isOrdered(arr5)){
            std::cout << "正确";
        }
        // 桶排序
        std::vector<int> arr6 = arr;
        std::cout << std::setw(15) << Sort::measureSortingTime(Sort::bucketSort, arr6);
        if (Sort::isOrdered(arr6)){
            std::cout << "正确";
        }
        std::cout << std::endl;
    }
}

void testFixedSize() {
    const int SIZE = 100000;
    const int REPEAT = 5;
    //创建一个数组
    std::vector<int> arr = Sort::generateRandomArray(SIZE);
    
    std::cout << "\n固定规模(" << SIZE << "个元素)重复" << REPEAT << "次的排序时间比较（单位：秒）：" << std::endl;
    std::cout << std::setw(10) << "测试次数" 
              << std::setw(15) << "插入排序"
              << std::setw(15) << "归并排序"
              << std::setw(15) << "快速排序"
              << std::setw(15) << "堆排序"
              << std::setw(15) << "基数排序"
              << std::setw(15) << "桶排序" << std::endl;

    for (int i = 1; i <= REPEAT; i++) {
        std::cout << std::setw(3) << i;
        
        if (i == 4) {
            //顺序，最佳情况
            arr = Sort::generateOrderArray(arr);
            // 插入排序
            std::vector<int> arr1 = arr;
            std::cout << std::setw(15) << std::fixed << std::setprecision(6)
                    << Sort::measureSortingTime(Sort::insertionSort, arr1);

            // 归并排序
            std::vector<int> arr2 = arr;
            std::cout << std::setw(15) << Sort::measureSortingTime(
                [](std::vector<int>& a) { Sort::mergeSort(a, 0, a.size()-1); }, arr2);

            // 快速排序
            std::vector<int> arr3 = arr;
            std::cout << std::setw(15) << Sort::measureSortingTime(
                [](std::vector<int>& a) { Sort::quickSort(a, 0, a.size()-1); }, arr3);

            // 堆排序
            std::vector<int> arr4 = arr;
            std::cout << std::setw(15) << Sort::measureSortingTime(Sort::heapSort, arr4);

            // 基数排序
            std::vector<int> arr5 = arr;
            std::cout << std::setw(15) << Sort::measureSortingTime(Sort::radixSort, arr5);

            // 桶排序
            std::vector<int> arr6 = arr;
            std::cout << std::setw(15) << Sort::measureSortingTime(Sort::bucketSort, arr6);

            std::cout << std::endl;
            continue;
        }
        if (i == 5){
            //倒序，最坏情况
            arr = Sort::generateReverseArray(arr);
            // 插入排序
            std::vector<int> arr1 = arr;
            std::cout << std::setw(15) << std::fixed << std::setprecision(6)
                    << Sort::measureSortingTime(Sort::insertionSort, arr1);

            // 归并排序
            std::vector<int> arr2 = arr;
            std::cout << std::setw(15) << Sort::measureSortingTime(
                [](std::vector<int>& a) { Sort::mergeSort(a, 0, a.size()-1); }, arr2);

            // 快速排序
            std::vector<int> arr3 = arr;
            std::cout << std::setw(15) << Sort::measureSortingTime(
                [](std::vector<int>& a) { Sort::quickSort(a, 0, a.size()-1); }, arr3);

            // 堆排序
            std::vector<int> arr4 = arr;
            std::cout << std::setw(15) << Sort::measureSortingTime(Sort::heapSort, arr4);

            // 基数排序
            std::vector<int> arr5 = arr;
            std::cout << std::setw(15) << Sort::measureSortingTime(Sort::radixSort, arr5);

            // 桶排序
            std::vector<int> arr6 = arr;
            std::cout << std::setw(15) << Sort::measureSortingTime(Sort::bucketSort, arr6);

            std::cout << std::endl;
            continue;
        }
        arr = Sort::disruptRandomArray(arr);
        // 插入排序
        std::vector<int> arr1 = arr;
        std::cout << std::setw(15) << std::fixed << std::setprecision(6)
                  << Sort::measureSortingTime(Sort::insertionSort, arr1);

        // 归并排序
        std::vector<int> arr2 = arr;
        std::cout << std::setw(15) << Sort::measureSortingTime(
            [](std::vector<int>& a) { Sort::mergeSort(a, 0, a.size()-1); }, arr2);

        // 快速排序
        std::vector<int> arr3 = arr;
        std::cout << std::setw(15) << Sort::measureSortingTime(
            [](std::vector<int>& a) { Sort::quickSort(a, 0, a.size()-1); }, arr3);

        // 堆排序
        std::vector<int> arr4 = arr;
        std::cout << std::setw(15) << Sort::measureSortingTime(Sort::heapSort, arr4);

        // 基数排序
        std::vector<int> arr5 = arr;
        std::cout << std::setw(15) << Sort::measureSortingTime(Sort::radixSort, arr5);

        // 桶排序
        std::vector<int> arr6 = arr;
        std::cout << std::setw(15) << Sort::measureSortingTime(Sort::bucketSort, arr6);

        std::cout << std::endl;
    }
}

int main() {
    std::cout << "排序算法时间复杂度比较实验\n" << std::endl;
    
    testWithDifferentSizes();
    testFixedSize();
    
    return 0;
}