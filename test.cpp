// #include <iostream>
// #include <cstring>
// using namespace std;
// int main() 
// {
// 	int* nums1;
// 	nums1 = (int *)malloc(4 * sizeof(int));
// 	nums1[0] = 3; nums1[1] = 6; nums1[2] = 4; nums1[3] = 89;

// 	for(int i = 0; i < 4; i++)
// 		std::cout << nums1[i] << " ";
// 	std::cout << std::endl;

// 	int* nums2;
// 	nums2 = (int *)malloc(5 * sizeof(int));
// 	nums2[0] = 21; nums2[1] = 21; nums2[2] = 21; nums2[3] = 84; nums2[4] = 23;
// 	for(int i = 0; i < 5; i++)
// 		std::cout << nums2[i] << " ";
// 	std::cout << std::endl;

// 	int* nums3;
// 	nums3 = (int *)malloc(sizeof(nums1) + sizeof(nums2));

// 	std::cout << nums3 << std::endl;
// 	memcpy (nums3, nums1, 4 * sizeof(int));
// 	std::cout << nums3 << std::endl;
// 	memcpy (nums3 + 4, nums2, 5 * sizeof(int));

// 	for(int i = 0; i < 9; i++)
// 		std::cout << nums3[i] << " ";
// 	std::cout << std::endl;
// 	return 0;
// }