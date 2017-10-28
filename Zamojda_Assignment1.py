#David Zamojda Assignment 1
import numpy as np

#flatten
def flatten(set):
    new_list = []
    for item in set:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis
	
#Power Set
def powerset(set):
  new_list=[[]]
  for elem in set:
    for sub_set in new_list:
      new_list=new_list+[list(sub_set)+[elem]]
  return new_list

#Permutations
def all_perms(set):
	if len(set)>0:
 		lis1 = []
 		lis2 = [] 
 		for item in set:
 			if item not in lis1:
 				temp = set[:] 
 				temp.remove(item)
 				for p in all_perms(temp):
 					lis2.append([item]+p)
 			lis1.append(item)
 		return lis2
 	else :
 		return [[]]

# number spiral
def spiral(n, end_corner) :
	matrix = np.zeros((n,n))
	num = (n**2) - 1 
	if end_corner == 1:
		x = 0
		y = 0
		while num >= 1:
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				x += 1
		#turn left while left one == 0
			x-=1
			y += 1
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				y += 1
			y-=1
			x-= 1
			while (x != -1) and y != -1 and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				x -= 1
			x += 1
			y -= 1
			if num == 0: 
				return
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				y -= 1
			y += 1			
			x += 1
		return matrix
		
		#do stuff
	elif end_corner == 2:
		x = 0
		y = n-1
		while num >= 1:	
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				y -= 1			
			y += 1			
			x += 1
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				x += 1
			#turn left while left one == 0
			x-=1
			y += 1
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				y += 1
			y-=1
			x-= 1
			while (x != -1) and y != -1 and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				x -= 1
			x += 1
			y -= 1
		return matrix
		# do stuff
	elif end_corner == 3:
		x = n-1
		y = n-1
		while num >= 1:	
			while (x != -1) and y != -1 and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				x -= 1
			x += 1
			y -= 1
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				y -= 1			
			y += 1			
			x += 1
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				x += 1
			#turn left while left one == 0
			x-=1
			y += 1
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				y += 1
			y-=1
			x-= 1
			
		return matrix
		# do stuff
	elif end_corner == 4:
		x = n-1
		y = 0
		while num >= 1:	
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				y += 1
			y-=1
			x-= 1
			while (x != -1) and y != -1 and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				x -= 1
			x += 1
			y -= 1
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				y -= 1			
			y += 1			
			x += 1
			while (x != n) and y != n and matrix[x][y] == 0:
				matrix[x][y] = num
				num -= 1
				x += 1
			#turn left while left one == 0
			x-=1
			y += 1
			
			
		return matrix
		# do stuff
	else :
		return 'Wrong input'
	

