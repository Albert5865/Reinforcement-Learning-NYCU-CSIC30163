import matplotlib.pyplot as plt

# Initialize empty lists to store X and Y data
x_data = []
y_data = []

with open('record.txt', 'r') as file:
    for line in file:

        columns = line.split()
        x_value = int(columns[0])
        y_value = int(columns[1])
        

        x_data.append(x_value)
        y_data.append(y_value)

#plt.figure(figsize=(8, 6))
plt.plot(x_data, y_data, marker='o', linestyle='-') 
plt.xlabel('episode')
plt.ylabel('score')  
#plt.title('Data from record.txt') 
plt.grid(True)  
plt.show()
