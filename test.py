import numpy as np

randscore = np.load("CartPoleRandScores100.npy")
learnscore = np.load("CartPoleLearnScores100.npy")

print(randscore)
print(learnscore)

print((learnscore[0] - randscore[0]) / ((learnscore[0] + randscore[0])/2))
learnscore[0,0,0] = 0
learnscore[0,1,1] = 0
learnscore[0,2,2] = 0
learnscore[0,3,3] = 0

randscore[0,0,0] = 0
randscore[0,1,1] = 0
randscore[0,2,2] = 0
randscore[0,3,3] = 0

print(np.sum(randscore)/12)
print(np.sum(learnscore)/12)