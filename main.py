import numpy as np

def quantize_value(sl, sw, pl, pw):
  x1 = 0
  x2 = 0
  x3 = 0
  x4 = 0

  if sl < 5.5:
      x1 = 0
  elif 5.5 < sl < 6.5:
      x1 = 1
  elif 6.5 < sl:
      x1 = 2

  if sw < 2.8:
      x2 = 0
  elif 2.8 < sw < 3.3:
      x2 = 1
  elif 3.3 < sw:
      x2 = 2 

  if pl < 2.0:
      x3 = 0
  elif 2.0 < pl < 5.0:
      x3 = 1
  elif 5.0 < pl:
      x3 = 2

  if pw < 0.7:
      x4 = 0
  elif 0.7 < pw < 1.7:
      x4 = 1
  elif 1.7 < pw:
      x4 = 2

  return x1, x2, x3, x4

def single_neuron(x1, x2, x3, x4):
  S1 = -x1 +x2 -x4
  S2 = 2*x1 + x2 +x3
  S3 = x3 + x4
  S4 = -x1 - x2 - x3 - x4
  
  if S1 >= 1:
    S1 = 1
  else:
    S1 = 0
    
  if S2 >= 5:
    S2 = 1
  else:
    S2 = 0

  if S3 >= 2:
    S3 = 1
  else:
    S3 = 0
    
  if S4 >= 1:
    S4 = 1
  else:
    S4 = 0

  return S1, S2, S3, S4

def layer_two(S1, S2, S3, S4):
  S5 = 2*S1 + S2 - 2*S3 + S4
  S6 = -2*S1 - S2 + S3
  S7 = S1 + S2 + S3 - 2*S4

  X = [0, 0, 0] 
  
  if S5 >= 2:
    X[0] = 1
  else:
    X[0] = 0

  if S6 >= 1:
    X[1] = 1
  else:
    X[1] = 0

  if S7 >= 2:
    X[2] = 1
  else:
    X[2] = 0

  return X

def identify_iris(X):
  if X == [1, 0, 0]:
      return "Iris-setosa"
  elif X == [0, 1, 0]:
      return "Iris-versicolor"
  elif X == [0, 0, 1]:
      return "Iris-virginica"


correct_count = 0 
data = []

#(filename, letter according to file type)
with open('iris.txt', 'r') as file:
  lines = file.readlines() #read all content

data = np.array(data) #initialization

for line in lines:
  line = line.strip()#read by line
  if line:
    data = line.split(",")#被逗号分开的value，4个的统称
    
    values = [value.strip('"') for value in data[:4]]
    #create empty string(container) for values to go in, first 4
    
    if all(values):
      sl, sw, pl, pw = [float(value) for value in values]
      
      actual_answer = data[4]

      x1, x2, x3, x4 = quantize_value(sl, sw, pl, pw)
      
      S1, S2, S3, S4 = single_neuron(x1, x2, x3, x4)
      
      X = layer_two(S1, S2, S3, S4)
      
      predicted_result = identify_iris(X)
      
      if predicted_result == actual_answer:
        correct_count += 1

total_count = len(lines)
accuracy = correct_count / total_count

print(total_count)
print(correct_count)
print("Accuracy:", accuracy)