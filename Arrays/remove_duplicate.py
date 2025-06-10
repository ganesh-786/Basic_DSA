nums=[1,2,3,3,4,5]
def removeDuplicates(nums):
    if not nums:
        return 0
    write_index = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            nums[write_index] = nums[i]
            write_index += 1
    return write_index

length = removeDuplicates(nums)
print(nums[:length])
