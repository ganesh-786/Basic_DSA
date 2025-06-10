nums=[1,2,3,4,5]
k= 4
def rotate(nums, k):
    k = k % len(nums)
    nums[:] = nums[-k:] + nums[:-k]


rotate(nums,k)
print(nums)