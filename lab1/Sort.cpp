#include <vector>
#include <algorithm>
#include <ctime>
#include <random>
#include <chrono>
#include <iostream>

class Sort {
private:
    // 用于快速排序的分区函数
    static int partition(std::vector<int>& arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);
        return i + 1;
    }

    // 用于归并排序的合并函数
    static void merge(std::vector<int>& arr, int left, int mid, int right) {
        std::vector<int> temp(right - left + 1);
        int i = left, j = mid + 1, k = 0;
        
        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
            }
        }
        
        while (i <= mid) temp[k++] = arr[i++];
        while (j <= right) temp[k++] = arr[j++];
        
        for (i = 0; i < k; i++) {
            arr[left + i] = temp[i];
        }
    }

    // 堆排序的堆化函数
    static void heapify(std::vector<int>& arr, int n, int i) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && arr[left] > arr[largest])
            largest = left;

        if (right < n && arr[right] > arr[largest])
            largest = right;

        if (largest != i) {
            std::swap(arr[i], arr[largest]);
            heapify(arr, n, largest);
        }
    }

public:
    // 插入排序
    static void insertionSort(std::vector<int>& arr) {
        int n = arr.size();
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }

    // 归并排序
    static void mergeSort(std::vector<int>& arr, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            merge(arr, left, mid, right);
        }
    }

    // 快速排序
    static void quickSort(std::vector<int>& arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    // 堆排序
    static void heapSort(std::vector<int>& arr) {
        int n = arr.size();

        for (int i = n / 2 - 1; i >= 0; i--)
            heapify(arr, n, i);

        for (int i = n - 1; i > 0; i--) {
            std::swap(arr[0], arr[i]);
            heapify(arr, i, 0);
        }
    }

    // 基数排序
    static void radixSort(std::vector<int>& arr) {
        int max = *std::max_element(arr.begin(), arr.end());
        
        for (int exp = 1; max/exp > 0; exp *= 10) {
            std::vector<int> output(arr.size());
            std::vector<int> count(10, 0);
            
            for (int i = 0; i < arr.size(); i++)
                count[(arr[i]/exp)%10]++;
                
            for (int i = 1; i < 10; i++)
                count[i] += count[i-1];
                
            for (int i = arr.size()-1; i >= 0; i--) {
                output[count[(arr[i]/exp)%10]-1] = arr[i];
                count[(arr[i]/exp)%10]--;
            }
            
            arr = output;
        }
    }

    // 桶排序
    static void bucketSort(std::vector<int>& arr) {
        int max = *std::max_element(arr.begin(), arr.end());
        int min = *std::min_element(arr.begin(), arr.end());
        
        int bucketCount = arr.size();
        std::vector<std::vector<int>> buckets(bucketCount);
        
        for (int i = 0; i < arr.size(); i++) {
            int bucketIndex = (arr[i] - min) * (bucketCount - 1) / (max - min);
            buckets[bucketIndex].push_back(arr[i]);
        }
        
        for (auto& bucket : buckets) {
            std::sort(bucket.begin(), bucket.end());
        }
        
        int index = 0;
        for (const auto& bucket : buckets) {
            for (int num : bucket) {
                arr[index++] = num;
            }
        }
    }

    static std::vector<int> generateOrderArray(std::vector<int>& arr) {
        int max = *std::max_element(arr.begin(), arr.end());
        
        for (int exp = 1; max/exp > 0; exp *= 10) {
            std::vector<int> output(arr.size());
            std::vector<int> count(10, 0);
            
            for (int i = 0; i < arr.size(); i++)
                count[(arr[i]/exp)%10]++;
                
            for (int i = 1; i < 10; i++)
                count[i] += count[i-1];
                
            for (int i = arr.size()-1; i >= 0; i--) {
                output[count[(arr[i]/exp)%10]-1] = arr[i];
                count[(arr[i]/exp)%10]--;
            }
            
            arr = output;
        }
        return arr;
    }
    static std::vector<int> generateReverseArray(std::vector<int>& arr) {
        int max = *std::max_element(arr.begin(), arr.end());
        
        for (int exp = 1; max/exp > 0; exp *= 10) {
            std::vector<int> output(arr.size());
            std::vector<int> count(10, 0);
            
            for (int i = 0; i < arr.size(); i++)
                count[(arr[i]/exp)%10]++;
                
            for (int i = 1; i < 10; i++)
                count[i] += count[i-1];
                
            for (int i = arr.size()-1; i >= 0; i--) {
                output[count[(arr[i]/exp)%10]-1] = arr[i];
                count[(arr[i]/exp)%10]--;
            }
            
            arr = output;
        }
        return {arr.rbegin(),arr.rend()};
    }

    // 生成随机数组
    static std::vector<int> disruptRandomArray(std::vector<int>& arr) {
        std::random_shuffle(arr.begin(),arr.end());
        return arr;
    }
    // 打乱数组
    static std::vector<int> generateRandomArray(int size) {
        std::vector<int> arr(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 1000);
        
        for (int i = 0; i < size; i++) {
            arr[i] = dis(gen);
        }
        return arr;
    }

    //判断是否为顺序
    static bool isOrdered(const std::vector<int>& arr) {
        for(int i = 0; i < arr.size()-1; i++) {
            if(arr[i] > arr[i+1]) {
                return false;
            }
        }
        return true;
    }

    // 测量排序算法的执行时间
    static double measureSortingTime(void (*sortFunc)(std::vector<int>&), std::vector<int>& arr) {
        auto start = std::chrono::high_resolution_clock::now();
        sortFunc(arr);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        return diff.count();
    }
};