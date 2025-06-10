nums=[0,0,1,2,3,4]
def moveZeros(nums):
    left = 0
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1

moveZeros(nums)
print(nums)