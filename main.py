
import perceptron


x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0, 0], [1, 0], [1, 0], [0, 1]]

n = 0
perceptron = perceptron.Perceptron(x[n])

for i in range(5000):
    n = i % 4
    perceptron.backPropagation(x[n], y[n])


entrada = -1

print("For QUIT insert 0\n")
while entrada != '0':
    entrada = input("Enter input: ")
    if entrada != '0':
        entrada = entrada.split(",")
        entrada[0] = int(entrada[0])
        entrada[1] = int(entrada[1])
        print("INPUT: ", entrada)
        print("OUTPUT:", end= " ")
        perceptron.calculateFinalOutput(entrada)
    else:
        print("Bye")


